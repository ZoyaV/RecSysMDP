import pandas as pd
import numpy as np

def load_data(data_path, return_values, col_mapping):
    data = pd.read_csv(data_path, sep = "\t")
    if return_values:
        full_users = data[col_mapping['user_col_name']].values
        full_items = data[col_mapping['item_col_name']].values

        users_unique = np.unique(data[col_mapping['user_col_name']].values)
        items_unique = np.unique(data[col_mapping['item_col_name']].values)

        rating = data[col_mapping['reward_col_name']].values

        values = {'users_unique': users_unique, 'items_unique':items_unique,
                  'full_users': full_users, 'full_items':full_items, 'rating':rating}
        return data, col_mapping, values
    return data, col_mapping

def load_reward_function(reward_function_name):
    # Load reward function
    if reward_function_name == 'condition_reward':
        from recsys_mdp.reward_functions import condition_reward
        reward_function = condition_reward
    elif reward_function_name == 'relevance_based_reward':
        from recsys_mdp.reward_functions import relevance_based_reward
        reward_function = relevance_based_reward
    elif reward_function_name == 'monotony_reward':
        from recsys_mdp.reward_functions import monotony_reward
        reward_function = monotony_reward
    return reward_function

def load_action_function(action_function_name):
    # Load action function
    if action_function_name == 'next_item_action':
        from recsys_mdp.action_function import next_item_action
        action_function = next_item_action
    elif action_function_name == 'continuous_relevance_action':
        from recsys_mdp.action_function import continuous_relevance_action
        action_function = continuous_relevance_action
    elif action_function_name == 'discrete_relevance_action':
        from recsys_mdp.action_function import discrete_relevance_action
        action_function = discrete_relevance_action
    return action_function

def make_mdp(data, data_mapping, framestack_size, history_keys,
             action_function_name, reward_function_name, mdp_type,
             window_size, step_size):
    reward_function = load_reward_function(reward_function_name)
    action_function = load_action_function(action_function_name)
    if mdp_type == "WindowBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import WindowBasedRecSysMDP
        mdp_preparator_class = WindowBasedRecSysMDP(load_from_file=False, dataframe=data,
                                                    data_mapping=data_mapping, framestack=framestack_size,
                                                    history_keys=history_keys,
                                                    reward_function=reward_function,
                                                    action_function=action_function,
                                                    window_size=window_size,
                                                    step_size=step_size)

    if mdp_type == "ConditionBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import ConditionBasedRecSysMDP
        from constructors.episode_split_fucntions import split_by_time

        mdp_preparator_class = ConditionBasedRecSysMDP(load_from_file=False, dataframe=data,
                                                       data_mapping=data_mapping, framestack=framestack_size,
                                                       history_keys=history_keys,
                                                       reward_function=reward_function,
                                                       action_function=action_function,
                                                       condition=split_by_time)

    if mdp_type == "FullUserHistoryBasedRecSysMDP":
        from recsys_mdp.recsys_mdp import FullUserHistoryBasedRecSysMDP

        mdp_preparator_class = FullUserHistoryBasedRecSysMDP(load_from_file=False, dataframe=data,
                                                             data_mapping=data_mapping, framestack=framestack_size,
                                                             history_keys=history_keys,
                                                             reward_function=reward_function,
                                                             action_function=action_function)
    return mdp_preparator_class
