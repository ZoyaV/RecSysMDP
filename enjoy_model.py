import os

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import count

from d3rlpy.base import LearnableBase

from recsys_mdp.generators.scenarios.mdp_next_item_integration import NextItemEnvironment, TypesResolver
from recsys_mdp.generators.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from constructors.mdp_constructor import load_data, make_mdp
from constructors.algorithm_constuctor import init_algo,init_model
from recsys_mdp.mdp_former.utils import to_d3rlpy_form_ND

import yaml
import pickle


def load_pretrained_model(conf_name):
    with open(f"./checkpoints/{conf_name}/cfg.yaml") as f:
        config = yaml.load(f)
    prediction_type = True if config['experiment']['scorer']['prediction_type'] == "discrete" else False

    # Load train data
    data, data_mapping, train_values = load_data(data_path=config['experiment']['data_path'],
                                                 return_values=True,
                                                 col_mapping=config['experiment']['col_mapping'])
    mdp_preparator = make_mdp(data=data, data_mapping=data_mapping, **config['experiment']['mdp_settings'])
    states, rewards, actions, termations, state_tail = mdp_preparator.create_mdp()
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations, discrete=prediction_type)

    model = init_model(train_values, **config['experiment']['algo_settings']['model_parametrs'])
    algo = init_algo(model, **config['experiment']['algo_settings']['general_parametrs'])
    algo.build_with_dataset(train_mdp)
    algo.load_model(f'checkpoints/{conf_name}/{conf_name}.pt')
    return algo


def get_enjoy_setting(pretrain_conf, config_path="recsys_mdp/generators/configs/mdp_next_item_integration.yaml"):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    path_object = Path(config_path)
    config_class = GlobalConfig(
        config=config, config_path=path_object, type_resolver=TypesResolver()
    )
    # Unpickle the object
    with open('env.pkl', 'rb') as f:
        env = pickle.load(f)

    print(type(env))
    with open('env.pkl', 'wb') as f:
        pickle.dump(env, f)

    model = load_pretrained_model(pretrain_conf)
    gen_conf = config['generation']
    return gen_conf, env, model

def generate_episode(env, model):
    env, model = env, model
    user_id = env.reset()
    trajectory = []

    # [10 last item_ids] + [user_id]
    fake_obs = np.random.randint(0, 100, 10).tolist() + [user_id]
    obs = np.asarray(fake_obs)

    while True:
        try:
            item_id = model.predict(obs.reshape(1, -1))[0]
        except:
            item_id = model.predict(obs[:10].reshape(1, -1))[0]
        obs[:9] = obs[1:10]
        obs[-2] = item_id

        timestamp = env.timestamp

        relevance, terminated = env.step(item_id)
        continuous_relevance, discrete_relevance = relevance
        trajectory.append((
            timestamp,
            user_id, item_id,
            continuous_relevance, discrete_relevance,
            terminated
        ))
        if terminated:
            break
    return trajectory

def eval_returns(env, model):
    cont_returns, disc_returns = [], []
    for ep in range(20):
        trajectory = generate_episode(env, model)
        cont_returns.append(np.sum([step[2] for step in trajectory]))
        disc_returns.append(np.sum([step[3] for step in trajectory]))
    return {
        'continuous_return': np.mean(cont_returns),
        'discrete_return': np.mean(disc_returns),
    }

if __name__ == "__main__":
    gen_conf, env, model = get_enjoy_setting(pretrain_conf = "mmdhcrqdqr")
    print(eval_returns(env, model))