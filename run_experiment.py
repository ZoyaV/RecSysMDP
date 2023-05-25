import argparse
import pickle
import random

import numpy as np
import torch
import yaml

from constructors.algorithm_constuctor import init_algo, init_model
from constructors.mdp_constructor import load_data, make_mdp
from constructors.scorers_constructor import init_logger
from enjoy_cases import eval_returns
from recsys_mdp.experiments.scenarios.mdp_next_item import USER_RESET_MODE_INIT
from recsys_mdp.utils.lazy_imports import lazy_import
from recsys_mdp.mdp.utils import to_d3rlpy_form_ND

wandb = lazy_import('wandb')


def episode_statistics(states):
    sizes = []
    for state in states:
        sizes.append(len(state))
    print("Mean: ", np.mean(sizes))
    print("Median: ", np.median(sizes))
    print("Max: ", np.max(sizes))
    print("Min: ", np.min(sizes))
    print("STD: ", np.std(sizes))


def eval_algo(algo, logger, train_logger, env = None, looking_for = None):
    if env:
        env.hard_reset(mode=USER_RESET_MODE_INIT)

        online_res = dict()
        looking_for.append(None)
        for i in looking_for:
            online_res[f"user {i}"] = eval_returns(
                env, algo, user_id=i, logger=logger.wandb_logger
            )
    else:
        online_res =None
    print("END")

   # print(online_res)
    logger.visual_log(algo, {
        "test_STAT": logger.static_log(algo),
        "test_INTERECT": logger.interactive_log(algo),
        "ONLINE": online_res
    })

    train_logger.visual_log(algo, {
        "train_STAT": train_logger.static_log(algo),
        "train_INTERECT": train_logger.interactive_log(algo),
    })


def fit(
        algo, n_epochs, train_mdp, test_mdp, train_logger, test_logger,
        model_name, eval_schedule=5, env = None, looking_for=None
):
    fitter = algo.fitter(train_mdp, n_epochs=n_epochs, save_metrics=False,)

    for epoch, metrics in fitter:
        if epoch % eval_schedule == 0:
            eval_algo(algo, test_logger, train_logger, env, looking_for)
          #  algo.save_model(f'checkpoints/{model_name}/{model_name}_{epoch}.pt')

    # algo.save_model(f'checkpoints/{model_name}/{model_name}_final.pt')
    return algo


def run_experiment(
        *,
        top_k, data_path, test_data_path, ratings_column,
        mdp_settings, scorer, algo_settings,
        model_name, wandb_logger=None, env_path = None, looking_for = None
):
    prediction_type = scorer['prediction_type'] == "discrete"

    # Load train data
    data = load_data(filepath=data_path, relevance_ratings_column=ratings_column)
    mdp_preparator = make_mdp(data=data, **mdp_settings)
    states, rewards, actions, terminations, state_tail = mdp_preparator.create_mdp()
    print("Train episode-statistics")
    episode_statistics(states)
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, terminations, discrete=prediction_type)

    # Load test data
    test_data = load_data(filepath=test_data_path, relevance_ratings_column=ratings_column)

    mdp_settings['episode_splitter_name'] = "interaction_interruption"
    test_mdp_preparator = make_mdp(data=test_data, **mdp_settings)
    states, rewards, actions, terminations, _ = test_mdp_preparator.create_mdp()
    print("Test episode-statistics")
    episode_statistics(states)

    test_mdp = to_d3rlpy_form_ND(states, rewards, actions, terminations, discrete=prediction_type)

    # Init RL algorithm
    model = init_model(data, **algo_settings['model_parametrs'])
    algo = init_algo(model, **algo_settings['general_parametrs'])

    test_logger = init_logger(
        test_mdp, state_tail, test_data, top_k,
        wandb_logger=wandb_logger,
        **scorer
    )

    train_logger = init_logger(
        train_mdp, state_tail, data, top_k,
        wandb_logger=wandb_logger,
        **scorer
    )
    # Init online env if cheked
    with open(f'{env_path}/env.pkl', 'rb') as f:
        env = pickle.load(f)
    # Run experiment
    n_epochs = algo_settings['n_epochs']
    fit(
        algo, n_epochs, train_mdp=train_mdp, test_mdp=test_mdp,
        train_logger=train_logger, test_logger=test_logger,
        model_name=model_name, eval_schedule=25, env=env, looking_for=looking_for
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    parser.add_argument('--folder_name', type = str, default = "aboba")
    parser.add_argument('--experiment_name', type = str, default="default_exp")
    parser.add_argument('--framestack', type = int, default = 0)
    parser.add_argument('--model_parametrs', nargs='+',default = [])
    parser.add_argument('--freeze_emb', type=int)
    parser.add_argument('--state_repr', type=str)
    parser.add_argument('--use_als', type=int)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    np.random.seed(int(config['seed']))
    random.seed(int(config['seed']))
    torch.manual_seed(int(config['seed']))

    env_path = None
    if bool(config["eval_online"]):
        env_path = config['env_path']

    looking_for = [int(user_id) for user_id in config['looking_for'].split(",")]
    checkpoints_name = 'checkpoints'
    if args.folder_name is not None:
        checkpoints_name = args.folder_name

    experiment = config['experiment']

    if args.framestack != 0:
        experiment['algo_settings']['model_parametrs']['memory_size'] = args.framestack
        experiment['mdp_settings']['framestack_size'] = args.framestack
        print("Update framestack: ", experiment['mdp_settings']['framestack_size'])

    top_k = experiment['top_k']
    framestack_size = experiment['mdp_settings']['framestack_size']

    if args.experiment_name:
        config['name'] = args.experiment_name

    if checkpoints_name is None:
        model_name = f"{args.experiment_name}_top_k_{top_k}_framestack_size_{framestack_size}"
    else:
        model_name = checkpoints_name

    if args.freeze_emb:
        experiment['algo_settings']['model_parametrs']['freeze_emb'] = args.freeze_emb
    if args.state_repr:
        experiment['algo_settings']['model_parametrs']['state_repr_name'] = args.state_repr
    if args.use_als:
        experiment['algo_settings']['use_als'] = args.use_als

    prm = []
    if len(args.model_parametrs)>1:
        prm = [int(p) for p in args.model_parametrs]
        experiment['algo_settings']['model_parametrs']['user_num'] = prm[0]
        experiment['algo_settings']['model_parametrs']['item_num'] = prm[1]
        experiment['algo_settings']['model_parametrs']['emb_dim'] = prm[2]
        experiment['algo_settings']['model_parametrs']['hid_dim'] = prm[3]
        experiment['algo_settings']['model_parametrs']['memory_size'] = prm[4]
        experiment['algo_settings']['model_parametrs']['feature_size'] = prm[5]
        experiment['algo_settings']['model_parametrs']['attention_hidden_size'] = prm[6]

    wandb_run = None
    if config['use_wandb']:
        run_name = f"model_{prm}_top_k_{top_k}_framestack_size_{framestack_size}"
        wandb_run = wandb.init(
            project=f"{config['name']}",
            group=config['group_name'],
            name=run_name
        )

    run_experiment(
        model_name=model_name, wandb_logger=wandb_run,
        env_path = env_path, looking_for = looking_for, **experiment
    )


if __name__ == "__main__":
    main()
