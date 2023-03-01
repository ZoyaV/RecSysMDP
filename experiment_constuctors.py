import os
import argparse
import yaml
import pandas as pd
def load_data(config, data_path = None):
    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    if data_path is None:
        data_path = config['experiment']['data_path']
    col_mapping = config['experiment']['col_mapping']
    data = pd.read_csv(data_path, sep = "\t")
    return data, col_mapping

def load_reward_function(config):
    # Load reward function
    if config['experiment']['mdp_settings']['reward_function'] == 'condition_reward':
        from recsys_mdp.reward_functions import condition_reward
        reward_function = condition_reward
    elif config['experiment']['mdp_settings']['reward_function'] == 'relevance_based_reward':
        from recsys_mdp.reward_functions import relevance_based_reward
        reward_function = relevance_based_reward
    elif config['experiment']['mdp_settings']['reward_function'] == 'monotony_reward':
        from recsys_mdp.reward_functions import monotony_reward
        reward_function = monotony_reward
    return reward_function

def load_action_function(config):
    # Load action function
    if config['experiment']['mdp_settings']['action_function'] == 'next_item_action':
        from recsys_mdp.action_function import next_item_action
        action_function = next_item_action
    elif config['experiment']['mdp_settings']['action_function'] == 'continuous_relevance_action':
        from recsys_mdp.action_function import continuous_relevance_action
        action_function = continuous_relevance_action
    elif config['experiment']['mdp_settings']['action_function'] == 'discrete_relevance_action':
        from recsys_mdp.action_function import discrete_relevance_action
        action_function = discrete_relevance_action
    return action_function

def make_mdp(config, data, data_mapping):
    framestask_size = config['experiment']['mdp_settings']['framestack_size']
    reward_function = load_reward_function(config)
    action_function = load_action_function(config)
    history_keys = config['experiment']['mdp_settings']['history_keys']
    if config['experiment']['mdp_settings']['mdp_type'] == "WindowBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import WindowBasedRecSysMDP
        window_size = config['experiment']['mdp_settings']['window_size']
        step_size = config['experiment']['mdp_settings']['step_size']

        mdp_preparator_class = WindowBasedRecSysMDP(load_from_file=False, dataframe=data,
                                              data_mapping=data_mapping, framestack=framestask_size,
                                              history_keys=history_keys,
                                              reward_function=reward_function,
                                              action_function=action_function,
                                              window_size=window_size,
                                              step_size=step_size)

    if config['experiment']['mdp_settings']['mdp_type'] == "ConditionBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import ConditionBasedRecSysMDP
        mdp_preparator_class = ConditionBasedRecSysMDP(load_from_file=False, dataframe=data,
                                              data_mapping=data_mapping, framestack=framestask_size,
                                              history_keys=history_keys,
                                              reward_function=reward_function,
                                              action_function=action_function,
                                              condition=None)

    if config['experiment']['mdp_settings']['mdp_type'] == "FullUserHistoryBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import FullUserHistoryBasedRecSysMDP
        mdp_preparator_class = FullUserHistoryBasedRecSysMDP(load_from_file=False, dataframe=data,
                                              data_mapping=data_mapping, framestack=framestask_size,
                                              history_keys=history_keys,
                                              reward_function=reward_function,
                                              action_function=action_function)
    return mdp_preparator_class


def init_algo(config):
    from models.models import ActorEncoderFactory
    user_num = config['experiment']['algo_settings']['model_parametrs']['user_num']
    item_num = config['experiment']['algo_settings']['model_parametrs']['item_num']
    emb_dim = config['experiment']['algo_settings']['model_parametrs']['emb_dim']
    hid_dim = config['experiment']['algo_settings']['model_parametrs']['hid_dim']
    memory_size = config['experiment']['algo_settings']['model_parametrs']['memory_size']
    feature_size = config['experiment']['algo_settings']['model_parametrs']['feature_size']
    model_params = [user_num, item_num, emb_dim, hid_dim, memory_size, feature_size]
    actor_encoder_factory = ActorEncoderFactory(
        *model_params
    )
    use_gpu = config['experiment']['algo_settings']['use_gpu']
    batch_size = config['experiment']['algo_settings']['batch_size']
    algo = config['experiment']['algo_settings']['algo']
    if algo == 'CQL':
        from d3rlpy.algos import CQL
        algo = CQL(use_gpu=use_gpu, encoder_factory=actor_encoder_factory, batch_size=batch_size)

    if algo == 'DCQL':
        from d3rlpy.algos import DiscreteCQL
        algo = DiscreteCQL(use_gpu=use_gpu, encoder_factory=actor_encoder_factory, batch_size=batch_size)
    return algo

