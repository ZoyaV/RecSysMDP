import numpy as np
import random
import torch
import pickle
from enjoy_cases import eval_returns
import argparse

import yaml

from recsys_mdp.generators.utils.lazy_imports import lazy_import
from recsys_mdp.mdp_former.utils import to_d3rlpy_form_ND
from constructors.algorithm_constuctor import init_algo, init_model
from constructors.mdp_constructor import load_data, make_mdp
from constructors.scorers_constructor import init_scorers, init_logger

wandb = lazy_import('wandb')


def eval_algo(algo, logger, train_logger, env = None, looking_for = None):
    if env:
        online_res = dict()
        looking_for.append(-1)
        for i in looking_for:
            online_res[f"user {i}"] = eval_returns(env,algo,i)
    else:
        online_res =None
   # print(online_res)
    logger.visual_log(algo, {
        "STAT": logger.static_log(algo),
        "INTERECT": logger.interactive_log(algo),
        "ONLINE": online_res
    })

    train_logger.visual_log(algo, {
        "train_STAT": train_logger.static_log(algo),
        "train_INTERECT": train_logger.interactive_log(algo),
    })


def fit(
        algo, train_mdp, test_mdp, n_epochs, scorers,
        logger, train_scorers, train_loggers,
        model_name, eval_schedule=5, env = None, looking_for=None
):
    fitter = algo.fitter(
        train_mdp,
        n_epochs=n_epochs,
        eval_episodes=test_mdp,
        scorers=scorers,
        save_metrics=False
    )

    for epoch, metrics in fitter:
        if epoch % eval_schedule == 0:
            eval_algo(algo, logger, train_loggers, env, looking_for)
          #  algo.save_model(f'checkpoints/{model_name}/{model_name}_{epoch}.pt')

    algo.save_model(f'checkpoints/{model_name}/{model_name}_final.pt')
    return algo


def run_experiment(
        *,
        top_k, data_path, test_data_path, col_mapping,
        mdp_settings, scorer, algo_settings,
        model_name, wandb_logger=None, env_path = None, looking_for = None
):
    prediction_type = scorer['prediction_type'] == "discrete"

    # Load train data
    data, data_mapping, train_values = load_data(
        data_path=data_path,
        return_values=True,
        col_mapping=col_mapping
    )
    mdp_preparator = make_mdp(
        data=data, data_mapping=data_mapping, **mdp_settings
    )
    states, rewards, actions, terminations, state_tail = mdp_preparator.create_mdp()
  #  print(states)
    train_mdp = to_d3rlpy_form_ND(states, rewards, actions, terminations, discrete=prediction_type)

    # Load test data
    test_data, _, test_values = load_data(
        data_path=test_data_path,
        return_values=True,
        col_mapping=col_mapping
    )

    mdp_settings['episode_splitter_name'] = "interaction_interruption"
    test_mdp_preparator = make_mdp(
        data=test_data, data_mapping=data_mapping, **mdp_settings
    )
    states, rewards, actions, terminations, _ = test_mdp_preparator.create_mdp()

    test_mdp = to_d3rlpy_form_ND(states, rewards, actions, terminations, discrete=prediction_type)

    # Init RL algorithm
    model = init_model(train_values, **algo_settings['model_parametrs'])
    algo = init_algo(model, **algo_settings['general_parametrs'])

    # Init scorer
    scorers = init_scorers(state_tail, test_values, top_k, **scorer)
    logger = init_logger(
        test_mdp, state_tail, test_values, top_k,
        wandb_logger=wandb_logger,
        **scorer
    )

    train_scorers = init_scorers(state_tail, train_values, top_k, **scorer)
    train_logger = init_logger(
        train_mdp, state_tail, train_values, top_k,
        wandb_logger=wandb_logger,
        **scorer
    )
    # Init online env if cheked
    with open(f'{env_path}/env.pkl', 'rb') as f:
        env = pickle.load(f)
    # Run experiment
    n_epochs = algo_settings['n_epochs']
    fit(
        algo, train_mdp, test_mdp, n_epochs,
        scorers, logger,train_scorers, train_logger,
        model_name=model_name, eval_schedule=25, env = env, looking_for = looking_for
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

    run_experiment(model_name=model_name, wandb_logger=wandb_run, env_path = env_path, looking_for = looking_for, **experiment)


if __name__ == "__main__":
    main()
