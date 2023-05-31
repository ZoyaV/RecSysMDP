from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, RATING_COL, USER_ID_COL, ITEM_ID_COL,
    RELEVANCE_INT_COL
)


def prepare_log_df(log_df: pd.DataFrame, column_for_rating: str = None) -> pd.DataFrame:
    # ensure correct column types
    log_df[TIMESTAMP_COL] = pd.to_datetime(log_df[TIMESTAMP_COL])
    log_df[USER_ID_COL] = log_df[USER_ID_COL].astype(int)
    log_df[ITEM_ID_COL] = log_df[ITEM_ID_COL].astype(int)
    log_df[RELEVANCE_INT_COL] = log_df[RELEVANCE_INT_COL].astype(int)

    if column_for_rating is not None:
        log_df[RATING_COL] = log_df[column_for_rating]

    # ensure correct time-based sorting
    log_df = log_df.sort_values(TIMESTAMP_COL)
    return log_df


def cache_log_df(path: Path, log_df: pd.DataFrame):
    if not path.exists():
        # noinspection PyTypeChecker
        log_df.to_pickle(path)


def restore_log_df(path: Path):
    if path.exists():
        return pd.read_pickle(path)


def save_data(
        data: list[tuple], columns: list[str], save_dir: str | Path, dataset_name: str
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


def split_dataframe(df: pd.DataFrame, split_ratio: float = .8, time_sorted: bool = False):
    if time_sorted:
        # DataFrame is already sorted by time, therefore we can split by indices
        split_ind = int(df.shape[0] * split_ratio)
        return df.iloc[:split_ind], df.iloc[split_ind:]

    # calc timestamp for time split
    split_timestamp = df[TIMESTAMP_COL].quantile(split_ratio)

    # train/test split
    return df[df[TIMESTAMP_COL] <= split_timestamp], df[df[TIMESTAMP_COL] > split_timestamp]


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
