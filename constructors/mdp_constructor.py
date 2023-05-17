import pandas as pd
import numpy as np

def load_data(data_path, return_values, col_mapping):
    data = pd.read_csv(data_path)#, sep = "\t")
    data = data.sort_values(col_mapping['timestamp_col_name'])
    if return_values:
        full_users = data[col_mapping['user_col_name']].values
        full_items = data[col_mapping['item_col_name']].values

        users_unique = np.unique(data[col_mapping['user_col_name']].values)
        items_unique = np.unique(data[col_mapping['item_col_name']].values)

        rating = data[col_mapping['reward_col_name']].values

        values = {'users_unique': users_unique, 'items_unique':items_unique,
                  'full_users': full_users, 'full_items':full_items, 'rating':rating}
        print("////////////////")
        print(values)
        return data, col_mapping, values
    print("////////////////")
    print(data)
    return data, col_mapping

def load_reward_function(reward_function_name):
    # Load reward function
    if reward_function_name == 'condition_reward':
        from recsys_mdp.mdp_former.rewarding import condition_reward
        reward_function = condition_reward
    elif reward_function_name == 'relevance_based_reward':
        from recsys_mdp.mdp_former.rewarding import relevance_based_reward
        reward_function = relevance_based_reward
    elif reward_function_name == 'monotony_reward':
        from recsys_mdp.mdp_former.rewarding import monotony_reward
        reward_function = monotony_reward
    return reward_function

def load_action_function(action_function_name):
    # Load action function
    if action_function_name == 'next_item_action':
        from recsys_mdp.mdp_former.acting import next_item_action
        action_function = next_item_action
    elif action_function_name == 'continuous_relevance_action':
        from recsys_mdp.mdp_former.acting import continuous_relevance_action
        action_function = continuous_relevance_action
    elif action_function_name == 'discrete_relevance_action':
        from recsys_mdp.mdp_former.acting import discrete_relevance_action
        action_function = discrete_relevance_action
    return action_function

def load_episode_splitter(episode_splitter_name):
    # Load action function
    if episode_splitter_name == 'interaction_interruption':
        from recsys_mdp.mdp_former.episode_splitting import split_by_time
        episode_splitter = split_by_time
    elif episode_splitter_name == 'full_user_interaction':
        from recsys_mdp.mdp_former.episode_splitting import split_by_user
        episode_splitter = split_by_user

    return episode_splitter

def make_mdp(data, data_mapping, framestack_size, history_keys,
             action_function_name, reward_function_name, episode_splitter_name):
    reward_function = load_reward_function(reward_function_name)
    action_function = load_action_function(action_function_name)
    episode_splitter = load_episode_splitter(episode_splitter_name)

    from recsys_mdp.mdp_former.mdp_former import MDPFormer
    mdp_preparator_class = MDPFormer(load_from_file=False, dataframe=data,
                                     data_mapping=data_mapping, framestack=framestack_size,
                                     history_keys=history_keys,
                                     reward_function=reward_function,
                                     action_function=action_function,
                                     episode_splitter=episode_splitter)
    return mdp_preparator_class
