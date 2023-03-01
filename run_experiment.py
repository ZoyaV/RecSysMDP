import os
import argparse
import yaml
import pandas as pd

from metrics.metrics import true_ndcg
from experiment_constuctors import load_data, make_mdp, init_algo
from recsys_mdp.utils import to_d3rlpy_form_ND

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    if config['use_wandb']:
        import wandb
        wandb.init(project=config['name'])

    # Load train data
    data, data_mapping = load_data(config=config)
    mdp_preparator = make_mdp(config=config, data=data, data_mapping=data_mapping)
    states, rewards, actions, termations = mdp_preparator.create_mdp()
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations)

    # Load test data
    test_data, _ = load_data(config=config, data_path=config['experiment']['test_data_path'])
    test_mdp_preparator = make_mdp(config=config, data=test_data, data_mapping=data_mapping)
    states, rewards, actions, termations = test_mdp_preparator.create_mdp()
    test_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations)

    # Init RL algorithm
    algo = init_algo(config)

    # Init scorer
    # TODO: neen scorer costructor
    top_k = config['experiment']['top_k']
    users = test_mdp.observations[:, -1]
    observations = test_mdp.observations[:]
    scorer = true_ndcg(users, observations, top_k=top_k)

    # Run experiment
    batch_size = config['experiment']['algo_settings']['n_epochs']
    algo.fit(train_mdp, n_epochs = batch_size, eval_episodes=test_mdp, scorers={'ndcg_sorer': scorer})