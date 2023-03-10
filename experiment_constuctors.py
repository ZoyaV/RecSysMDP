import os
import argparse
import yaml
import pandas as pd
import numpy as np
from recsys_mdp.utils import to_d3rlpy_form_ND
from metrics.metrics import base_ndcg
def load_data(config, data_path = None):
    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    if data_path is None:
        data_path = config['experiment']['data_path']
    col_mapping = config['experiment']['col_mapping']
    data = pd.read_csv(data_path, sep = "\t")
    print(len(set(data['user_idx'].values)))
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

# def init_interactive_scorer(config, test_data, data_mapping, top_k):
#     test_mdp_preparator = make_mdp(config=config, data=test_data, data_mapping=data_mapping)
#     states, rewards, actions, termations, _ = test_mdp_preparator.create_mdp()
#     test_mdp = to_d3rlpy_form_ND(states, rewards, actions, termations)
#
#     users = np.unique(test_mdp.observations[:, -1])
#     observations = test_mdp.observations[:]
#     observations = np.unique(observations, axis=0)
#     scorer = true_ndcg(users, observations, top_k=top_k)
#     return test_mdp, scorer

def init_next_step_scorer(state_tail, test_users, test_users_or, test_items, \
                          top_k, tresh, discrete = True):
    state_tail = np.asarray(state_tail)
    test_states = []
    true_items = dict()
    if discrete:
        for user in test_users:
            true_items[user] = test_items[test_users_or == user]
            user_mask = state_tail[:, -1] == user
            try:
                new_obs = state_tail[user_mask][0]
            except:
                continue
            test_states.append(new_obs)
    else:
        for user in test_users:
            user_mask = state_tail[:, -1] == user
            true_items[user] = test_items[test_users_or == user]
            test_items_unique = np.unique(test_items)
            for item in test_items_unique:
                try:
                    new_obs = state_tail[user_mask][0]
                except:
                    continue
                new_obs[0:-2] = new_obs[1:-1]
                new_obs[-2] = item
                test_states.append(new_obs)
    test_states = np.asarray(test_states)
    scorer = base_ndcg(test_states, true_items, tresh,top_k, discrete)
    return scorer