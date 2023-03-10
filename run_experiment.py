import os
import argparse
import yaml
import pandas as pd
import numpy as np

from experiment_constuctors import load_data, make_mdp, init_algo, init_next_step_scorer
from recsys_mdp.utils import to_d3rlpy_form_ND

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--experiment_name', type = str, default="default_exp")
    parser.add_argument('--framestack', type = int, default = 0)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    if args.framestack!=0:
        config['experiment']['algo_settings']['model_parametrs']['memory_size'] = args.framestack
        config['experiment']['mdp_settings']['framestack_size'] = args.framestack
        print("Update framestack: ", config['experiment']['mdp_settings']['framestack_size'])

    if config['use_wandb']:
        import wandb
        wandb.init(project=f"{config['name']}", \
                   group = args.experiment_name,
                   name=f"top_k_{config['experiment']['top_k']}_framestack_size_\
                        {config['experiment']['mdp_settings']['framestack_size']}")


    # Load train data
    data, data_mapping = load_data(config=config)
    items_count = set(np.unique(data[config['experiment']['col_mapping']['item_col_name']].values))
    users_count = set(np.unique(data[config['experiment']['col_mapping']['user_col_name']].values))
    print("Count of items: ", len(items_count), "Count of users: ", len(users_count))
    mdp_preparator = make_mdp(config=config, data=data, data_mapping=data_mapping)
    states, rewards, actions, termations, state_tail = mdp_preparator.create_mdp()

    prediction_type = True if config['experiment']['scorer']['prediction'] == "discrete" else False
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations, discrete = prediction_type)

    # Load test data
    test_data, _ = load_data(config=config, data_path=config['experiment']['test_data_path'])
    test_users_unique = np.unique(test_data[config['experiment']['col_mapping']['user_col_name']].values)
    test_users = test_data[config['experiment']['col_mapping']['user_col_name']].values
    test_items = test_data[config['experiment']['col_mapping']['item_col_name']].values
    # Init RL algorithm
    algo = init_algo(config)

    # Init scorer
    top_k = config['experiment']['top_k']
    # test_mdp, _ = init_interactive_scorer(config, test_data, data_mapping, top_k)
    tresh = config['experiment']['scorer']['tresh']
    rating_scorer = init_next_step_scorer(state_tail, test_users_unique, test_users, \
                                          test_items, top_k, tresh, discrete=prediction_type)

    # Run experiment
    batch_size = config['experiment']['algo_settings']['n_epochs']

    algo.fit(train_mdp, n_epochs = batch_size, eval_episodes=train_mdp, scorers={'rating_scorer': rating_scorer})