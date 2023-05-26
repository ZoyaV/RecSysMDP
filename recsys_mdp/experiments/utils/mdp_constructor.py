from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, RATING_COL, USER_ID_COL, ITEM_ID_COL,
    RELEVANCE_INT_COL
)


def save_data(
        data: list[tuple], columns: list[str], save_dir: str | Path, dataset_name: str,
        train_test_split: float | None = 0.7
):
    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=columns)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True)
    filepath = save_dir / f'{dataset_name}.csv'
    # Save the DataFrame to a CSV file
    print(f"Data generated to {filepath}: {df.shape}")
    df.to_csv(filepath, index=False)

    if train_test_split is not None:
        # calc timestamp for time split
        split_timestamp = df[TIMESTAMP_COL].quantile(train_test_split)

        # train/test split
        train_data = df[df[TIMESTAMP_COL] <= split_timestamp]
        test_data = df[df[TIMESTAMP_COL] > split_timestamp]

        # save train/test parts of the entire dataset
        train_data.to_csv(save_dir/'train_data.csv', index=False)
        test_data.to_csv(save_dir/'test_data.csv', index=False)
        print("Save train/test splits.")


def load_data(filepath, relevance_ratings_column: str = None):
    data = pd.read_csv(filepath)

    # ensure correct column types
    data[TIMESTAMP_COL] = pd.to_datetime(data[TIMESTAMP_COL])
    data[USER_ID_COL] = data[USER_ID_COL].astype(int)
    data[ITEM_ID_COL] = data[ITEM_ID_COL].astype(int)
    data[RELEVANCE_INT_COL] = data[RELEVANCE_INT_COL].astype(int)

    # ensure correct time-based sorting
    data = data.sort_values(TIMESTAMP_COL)

    # add column with user ratings based on one of the relevance columns
    if relevance_ratings_column is not None:
        data[RATING_COL] = data[relevance_ratings_column]

    return data


def load_reward_function(func_name):
    # Load reward function
    if func_name == 'condition_reward':
        from recsys_mdp.mdp.rewarding import condition_reward
        return condition_reward
    elif func_name == 'relevance_based_reward':
        from recsys_mdp.mdp.rewarding import relevance_based_reward
        return relevance_based_reward
    elif func_name == 'monotony_reward':
        from recsys_mdp.mdp.rewarding import monotony_reward
        return monotony_reward
    elif func_name == 'summary_reward':
        from recsys_mdp.mdp.rewarding import summary_reward
        return summary_reward
    elif func_name == 'ones_reward':
        from recsys_mdp.mdp.rewarding import ones_reward
        return ones_reward
    else:
        raise ValueError(f'Unknown reward func name {func_name}')


def load_action_function(func_name):
    # Load action function
    if func_name == 'next_item_action':
        from recsys_mdp.mdp.acting import next_item_action
        return next_item_action
    elif func_name == 'continuous_relevance_action':
        from recsys_mdp.mdp.acting import continuous_relevance_action
        return continuous_relevance_action
    elif func_name == 'discrete_relevance_action':
        from recsys_mdp.mdp.acting import discrete_relevance_action
        return discrete_relevance_action

    else:
        raise ValueError(f'Unknown action func name {func_name}')


def load_episode_splitter(splitter_name):
    # Load action function
    if splitter_name == 'interaction_interruption':
        from recsys_mdp.mdp.episode_splitting import split_by_time
        return split_by_time
    elif splitter_name == 'full_user_interaction':
        from recsys_mdp.mdp.episode_splitting import split_by_user
        return split_by_user
    else:
        raise ValueError(f'Unknown episode splitter name {splitter_name}')


def make_mdp(
        data, framestack_size,
        action_function_name, reward_function_name, episode_splitter_name
):
    reward_function = load_reward_function(reward_function_name)
    action_function = load_action_function(action_function_name)
    episode_splitter = load_episode_splitter(episode_splitter_name)

    from recsys_mdp.mdp.mdp_former import MDPFormer
    mdp_preparator_class = MDPFormer(
        load_from_file=False, dataframe=data,
        framestack=framestack_size,
        reward_function=reward_function,
        action_function=action_function,
        episode_splitter=episode_splitter,
    )
    return mdp_preparator_class
