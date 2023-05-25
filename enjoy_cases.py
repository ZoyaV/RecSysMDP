import argparse
import pickle

import numpy as np
import yaml

from constructors.algorithm_constuctor import init_algo, init_model
from constructors.mdp_constructor import load_data, make_mdp
from generate_synth_data import generate_episode
from recsys_mdp.utils.run.config import read_config
from recsys_mdp.utils.lazy_imports import lazy_import
from recsys_mdp.mdp.utils import to_d3rlpy_form_ND

wandb = lazy_import('wandb')


def load_pretrained_model(conf_name, step = -1):
    with open(f"./checkpoints/{conf_name}/cfg.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    prediction_type = config['experiment']['scorer']['prediction_type'] == "discrete"

    # Load train data
    data = load_data(
        filepath=config['experiment']['data_path'],
        relevance_ratings_column=config['experiment']['ratings_column'],
    )
    mdp_preparator = make_mdp(
        data=data, **config['experiment']['mdp_settings']
    )
    states, rewards, actions, termations, state_tail = mdp_preparator.create_mdp()
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations, discrete=prediction_type)

    model = init_model(data, **config['experiment']['algo_settings']['model_parametrs'])
    algo = init_algo(model, **config['experiment']['algo_settings']['general_parametrs'])
    algo.build_with_dataset(train_mdp)
    if step == -1:
        algo.load_model(f'checkpoints/{conf_name}/{conf_name}.pt')
    else:
        algo.load_model(f'checkpoints/{conf_name}/{conf_name}_{step}.pt')
    return algo


def get_enjoy_setting(pretrain_conf, env_path, config_path, model_epoch = -1):
    config = read_config(config_path)
    # Unpickle the object
    with open(f'{env_path}/env.pkl', 'rb') as f:
        env = pickle.load(f)

    model = load_pretrained_model(pretrain_conf, model_epoch)
    gen_conf = config['generation']
    return gen_conf, env, model


def eval_returns(env, model, user_id=None, logger=None):
    cont_returns, disc_returns, steps_hit_rate, coverages = [], [], [], []
    true_discrete_return = []
    n_episodes = 20 if user_id is not None else 50
    for ep in range(20):
        trajectory = generate_episode(env, model, user_id=user_id, log_sat=True, logger=logger)
        coverage = len({step[2] for step in trajectory})
        step_hit_rate = [step[2] in step[-1] for step in trajectory]
        cont_returns.append(np.mean([step[3] for step in trajectory]))
        disc_returns.append(np.mean([step[4] for step in trajectory]))
        true_discrete_return.append(np.sum([step[4] for step in trajectory]))
        coverages.append(coverage)
        steps_hit_rate.append(np.mean(step_hit_rate))

        # from temp_utils import log_distributions
        # log_distributions(true_items, predicted_items, "True best items", "Predicted best items")
    return {
        'continuous_return': np.mean(cont_returns),
        'discrete_return': np.mean(disc_returns),
        'true_discrete_return': np.mean(true_discrete_return),
        'coverage': np.mean(coverages),
        'step_hit_rate': np.mean(steps_hit_rate),
    }


def main():
    wandb_run = wandb.init(
        project="log_cases"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--pretrain_conf', type=str)
    parser.add_argument('--env', type=str)

    args = parser.parse_args()

    if args.config is None:
        args.config = "recsys_mdp/generators/configs/mdp_next_item_integration.yaml"

    for i in range(10, 500, 10):
        gen_conf, env, model = get_enjoy_setting(
            pretrain_conf=args.pretrain_conf, env_path=args.env,
            config_path=args.config, model_epoch=i
        )
        print(f"epoch {i} - {eval_returns(env, model)}")
        wandb_run.log(eval_returns(env, model))


if __name__ == "__main__":
    main()
