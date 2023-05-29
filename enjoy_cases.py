import pickle

import yaml

from recsys_mdp.experiments.utils.algorithm_constuctor import init_algo, init_model
from recsys_mdp.experiments.utils.mdp_constructor import load_data, make_mdp
from recsys_mdp.utils.run.config import read_config
from recsys_mdp.mdp.utils import to_d3rlpy_form_ND


def load_pretrained_model(conf_name, step=-1):
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


def get_enjoy_setting(pretrain_conf, env_path, config_path, model_epoch=-1):
    config = read_config(config_path)
    # Unpickle the object
    with open(f'{env_path}/env.pkl', 'rb') as f:
        env = pickle.load(f)

    model = load_pretrained_model(pretrain_conf, model_epoch)
    gen_conf = config['generation']
    return gen_conf, env, model


