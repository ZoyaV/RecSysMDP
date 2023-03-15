import pandas as pd
import numpy as np

def load_data(config, data_path = None, return_values = False):
    col_mapping = {'user_col_name': 'user_idx',
                   'item_col_name': 'item_idx',
                   'reward_col_name': 'rating',
                   'timestamp_col_name': 'ts'}
    if data_path is None:
        data_path = config['experiment']['data_path']
    col_mapping = config['experiment']['col_mapping']
    data = pd.read_csv(data_path, sep = "\t")
    if return_values:
        users_unique = np.unique(data[config['experiment']['col_mapping']['user_col_name']].values)
        items_unique = np.unique(data[config['experiment']['col_mapping']['item_col_name']].values)

        full_users = data[config['experiment']['col_mapping']['user_col_name']].values
        full_items = data[config['experiment']['col_mapping']['item_col_name']].values
        values = {'users_unique': users_unique, 'items_unique':items_unique,
                  'full_users': full_users, 'full_items':full_items}
        return data, col_mapping, values
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
        def split_by_time(df):
            ts_name = config['experiment']['col_mapping']['timestamp_col_name']
            ts = pd.to_datetime(df[ts_name]).astype(int) / 10**9#pd.to_datetime(df['datetime_col'])
           # print(ts)
            condition_music = lambda A: A > 100
            result = (ts[1:].values - ts[:-1].values).astype(int)
            indx = np.where(condition_music(result))
            if len(indx[0]) == 0:
                return [0,-1]
            return indx[0]

        mdp_preparator_class = ConditionBasedRecSysMDP(load_from_file=False, dataframe=data,
                                              data_mapping=data_mapping, framestack=framestask_size,
                                              history_keys=history_keys,
                                              reward_function=reward_function,
                                              action_function=action_function,
                                              condition=split_by_time)

    if config['experiment']['mdp_settings']['mdp_type'] == "FullUserHistoryBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import FullUserHistoryBasedRecSysMDP

        mdp_preparator_class = FullUserHistoryBasedRecSysMDP(load_from_file=False, dataframe=data,
                                              data_mapping=data_mapping, framestack=framestask_size,
                                              history_keys=history_keys,
                                              reward_function=reward_function,
                                              action_function=action_function)
    return mdp_preparator_class
