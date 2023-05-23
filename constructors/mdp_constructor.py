from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def save_data(
        data: list[tuple], columns: list[str], save_dir: str | Path, dataset_name: str,
        train_test_split: float | None = 0.7
):
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True)
    filepath = save_dir / f'{dataset_name}.csv'
    # Save the DataFrame to a CSV file
    print(f"Data generated to {filepath}: {df.shape}")
    df.to_csv(filepath, index=False)

    if train_test_split is not None:
        # calc timestamp for time split
        split_timestamp = df['timestamp'].quantile(train_test_split)

        # train/test split
        train_data = df[df['timestamp'] <= split_timestamp]
        test_data = df[df['timestamp'] > split_timestamp]

        # save train/test parts of the entire dataset
        train_data.to_csv(save_dir/'train_data.csv', index=False)
        test_data.to_csv(save_dir/'test_data.csv', index=False)
        print("Save train/test splits.")


def load_data(data_path, return_values, col_mapping: dict[str, str] = None):
    data = pd.read_csv(data_path)
    if col_mapping is not None:
        if set(col_mapping.keys()) & set(data.columns):
            data.rename(col_mapping)
            data.to_csv(data_path, index=False)

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


def load_reward_function(func_name):
    # Load reward function
    if func_name == 'condition_reward':
        from recsys_mdp.mdp_former.rewarding import condition_reward
        return condition_reward
    elif func_name == 'relevance_based_reward':
        from recsys_mdp.mdp_former.rewarding import relevance_based_reward
        return relevance_based_reward
    elif func_name == 'monotony_reward':
        from recsys_mdp.mdp_former.rewarding import monotony_reward
        return monotony_reward
    else:
        raise ValueError(f'Unknown reward func name {func_name}')


def load_action_function(func_name):
    # Load action function
    if func_name == 'next_item_action':
        from recsys_mdp.mdp_former.acting import next_item_action
        return next_item_action
    elif func_name == 'continuous_relevance_action':
        from recsys_mdp.mdp_former.acting import continuous_relevance_action
        return continuous_relevance_action
    elif func_name == 'discrete_relevance_action':
        from recsys_mdp.mdp_former.acting import discrete_relevance_action
        return discrete_relevance_action
    else:
        raise ValueError(f'Unknown action func name {func_name}')


def load_episode_splitter(splitter_name):
    # Load action function
    if splitter_name == 'interaction_interruption':
        from recsys_mdp.mdp_former.episode_splitting import split_by_time
        return split_by_time
    elif splitter_name == 'full_user_interaction':
        from recsys_mdp.mdp_former.episode_splitting import split_by_user
        return split_by_user
    else:
        raise ValueError(f'Unknown episode splitter name {splitter_name}')


def make_mdp(
        data, data_mapping, framestack_size, history_keys,
        action_function_name, reward_function_name, episode_splitter_name
):
    reward_function = load_reward_function(reward_function_name)
    action_function = load_action_function(action_function_name)
    episode_splitter = load_episode_splitter(episode_splitter_name)

    from recsys_mdp.mdp_former.mdp_former import MDPFormer
    mdp_preparator_class = MDPFormer(
        load_from_file=False, dataframe=data,
        data_mapping=data_mapping, framestack=framestack_size,
        history_keys=history_keys,
        reward_function=reward_function,
        action_function=action_function,
        episode_splitter=episode_splitter
    )
    return mdp_preparator_class
