import pickle
from pathlib import Path
import argparse
import numpy as np
import yaml

from constructors.algorithm_constuctor import init_algo, init_model
from constructors.mdp_constructor import load_data, make_mdp
from generate_synth_data import generate_episode
from recsys_mdp.generators.scenarios.mdp_next_item_integration import TypesResolver
from recsys_mdp.generators.utils.config import (
    GlobalConfig
)
from recsys_mdp.mdp_former.utils import to_d3rlpy_form_ND


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


def get_enjoy_setting(pretrain_conf, env_path, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # Unpickle the object
    with open(f'{env_path}/env.pkl', 'rb') as f:
        env = pickle.load(f)

    model = load_pretrained_model(pretrain_conf)
    gen_conf = config['generation']
    return gen_conf, env, model


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--pretrain_conf', type=str)
    parser.add_argument('--env', type=str)

    args = parser.parse_args()

    if args.config is None:
        args.config = "recsys_mdp/generators/configs/mdp_next_item_integration.yaml"
    gen_conf, env, model = get_enjoy_setting(pretrain_conf=args.pretrain_conf, env_path=args.env,
                                             config_path=args.config)
    print(eval_returns(env, model))

if __name__ == "__main__":
    main()
