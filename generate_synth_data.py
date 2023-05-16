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
        continuous_relevance, discrete_relevance = relevance
        items_top = env.state.ranked_items(with_satiation=True, discrete=True)
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


def make_user_with_stable_interest(user):
    user.satiation[:] = 100
    # most relevant item
    item = user.ranked_items(with_satiation=False, discrete=False)[0]
    item_cluster = user.embeddings.item_cluster_ind[item]

    user.satiation[item_cluster] = 0.1
    user.satiation_speed[item_cluster] = 0.0


def make_user_with_two_interests(user):
    user.satiation[:] = 1000
    static_tastes, _ = user.relevance(with_satiation=False, consume=False)
    sorted_tastes = user.ranked_items(with_satiation=False, discrete=False)

    top_1 = sorted_tastes[0]
    top_1_cluster_ind = user.embeddings.item_cluster_ind[top_1]
    top_1_emb = user.embeddings.items[top_1]

    top_2, top_2_emb, top_2_cluster_ind = None, None, None
    for item in sorted_tastes[1:]:
        item_cluster_ind = user.embeddings.item_cluster_ind[item]

        if item_cluster_ind != top_1_cluster_ind:
            from recsys_mdp.generators.datasets.synthetic.relevance import similarity
            emb = user.embeddings.items[item]
            sim = similarity(top_1_emb, emb, metric='l2')
            if sim > .9:
                continue

            top_2 = item
            top_2_emb = emb
            top_2_cluster_ind = user.embeddings.item_cluster_ind[top_2]
            break

    print(static_tastes[top_1], top_1_emb)
    print(static_tastes[top_2], top_2_emb)
    user.satiation[top_1_cluster_ind] = 0.1
    user.satiation[top_2_cluster_ind] = 0.1

    user.satiation_speed[top_1_cluster_ind] = 0.25
    user.satiation_speed[top_2_cluster_ind] = 0.5


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
    make_user_with_stable_interest(env.states[0])

    ### One two interest with interaction [need to balance]
    make_user_with_two_interests(env.states[6])

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
    print(f"Data generated to {file_name}: {df.shape}")
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
