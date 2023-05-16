import argparse
import os
import pickle
import random
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from d3rlpy.base import LearnableBase

from recsys_mdp.generators.scenarios.mdp_next_item_integration import NextItemEnvironment, TypesResolver
from recsys_mdp.generators.utils.config import (
    GlobalConfig
)

from als_model import ALSRecommender
def generate_episode(env, model, framestack_size=10, user_id = None):
    env, model = env, model
    if user_id is None:
        user_id = env.reset()
    else:
        user_id = env.reset(user_id)
    trajectory = []

    # [N last item_ids] + [user_id]
    fake_obs = np.random.randint(0, env.n_items, framestack_size).tolist() + [user_id]
    obs = np.asarray(fake_obs)
    item_id = 0
    while True:
        try:
            item_id = model.predict(obs.reshape(1, -1))[0]
        except:
            item_id = model.predict(obs[:framestack_size].reshape(1, -1))[0]

        obs[:framestack_size - 1] = obs[1:framestack_size]
        obs[-2] = item_id

        timestamp = env.timestamp

        relevance, terminated = env.step(item_id)
        continuous_relevance, discrete_relevance, items_top = relevance
        #item_id = random.choice(items_top[:10])

        trajectory.append((
            timestamp,
            user_id, item_id,
            continuous_relevance, discrete_relevance,
            terminated,
            items_top[:10]
        ))
        if terminated:
            break
    return trajectory

def poor_model_from_dataset(dataset_path):
    model = ALSRecommender()
    data = pd.read_csv(dataset_path)
    model.fit(data)
    return model


def generate_dataset(gen_conf, env, model):
    config = gen_conf
    samples = []
    for episode in count():
        samples.extend(generate_episode(env, model))
        if config['episodes_per_epoch'] is not None and episode >= config['episodes_per_epoch']:
            break
        if config['samples_per_epoch'] is not None and len(samples) >= config['samples_per_epoch']:
            break
    return samples


def make_env_setting(config_path="", env_name='default', dataset_path = None):
    # read configs
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    path_object = Path(config_path)
    config_class = GlobalConfig(
        config=config, config_path=path_object, type_resolver=TypesResolver()
    )
    # make conrollable environment
    env_conf = config['env']
    env: NextItemEnvironment = config_class.resolve_object(
        env_conf, object_type_or_factory=NextItemEnvironment
    )

    ### One of user have one stable interest
    env.states[0].satiation = np.ones_like(env.states[0].satiation)*1000
    most_relevant = np.argmax(env.states[0].tastes)
    env.states[0].satiation[most_relevant] = 0.001
    env.states[0].satiation_speed[most_relevant] = 0.0000001

    ### One two interest with interaction [need to balance]
    env.states[6].satiation = np.ones_like(env.states[1].satiation)*1000
    top_1, top_2 = np.argsort(env.states[1].tastes)[:2]

    env.states[6].satiation[top_1] = 0.00000001
    env.states[6].satiation[top_2] = 0.00000001

    env.states[6].satiation_speed[top_1] = 0.25
    env.states[6].satiation_speed[top_2] = 0.5

    # make model
    if dataset_path:
        model = poor_model_from_dataset(dataset_path)
    else:
        model_conf = config['model']
        model: LearnableBase = config_class.resolve_object(
            model_conf | dict(use_gpu=False),
            n_actions=env.n_items
        )
    gen_conf = config['generation']

    # save environment object
    with open(f'environments/{env_name}/env.pkl', 'wb') as f:
        pickle.dump(env, f)

    return gen_conf, env, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--base_data', type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        args.config = "recsys_mdp/generators/configs/mdp_next_item_integration.yaml"

    directory = f'environments/{args.env_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    gen_conf, env, model = make_env_setting(config_path=args.config, env_name=args.env_name, dataset_path=args.base_data)
    trajectories = generate_dataset(gen_conf, env, model)

    # Define the column names
    column_names = ['timestamp', 'user_idx', 'item_idx', 'relevance_cont', 'relevance_int', 'terminated', 'true_top']

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(trajectories, columns=column_names)
    file_name = f"{directory}/{args.env_name}.csv"

    # Save the DataFrame to a CSV file
    print("Data generated: ", df.shape)
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
