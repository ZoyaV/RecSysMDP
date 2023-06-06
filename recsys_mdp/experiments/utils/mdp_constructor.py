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
    else:
        assert RATING_COL in log_df.columns

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


def resolve_reward_function(reward):
    if reward == 'irrelevant':
        from recsys_mdp.mdp.rewarding import irrelevant
        return irrelevant
    elif reward == 'relevant':
        from recsys_mdp.mdp.rewarding import relevant
        return relevant
    elif reward == 'one':
        from recsys_mdp.mdp.rewarding import const_one
        return const_one
    elif reward == 'sparse_return':
        from recsys_mdp.mdp.rewarding import sparse_return
        return sparse_return
    elif reward == 'sparse_length':
        from recsys_mdp.mdp.rewarding import sparse_episode_length
        return sparse_episode_length
    else:
        raise ValueError(f'Unknown reward func name {reward}')


def resolve_action_function(action):
    if action == 'item':
        from recsys_mdp.mdp.acting import item
        return item
    elif action == 'continuous_rating':
        from recsys_mdp.mdp.acting import continuous_rating
        return continuous_rating
    elif action == 'discrete_rating':
        from recsys_mdp.mdp.acting import discrete_rating
        return discrete_rating

    else:
        raise ValueError(f'Unknown action func name {action}')


def load_episode_splitter(episode):
    # Load action function
    if episode == 'pause':
        from recsys_mdp.mdp.episode_splitting import by_pause
        return by_pause
    elif episode == 'user':
        from recsys_mdp.mdp.episode_splitting import by_user
        return by_user
    else:
        raise ValueError(f'Unknown episode splitter name {episode}')


def make_mdp(data, framestack_size, action, reward, episode):
    reward_function = resolve_reward_function(reward)
    action_function = resolve_action_function(action)
    episode_splitter = load_episode_splitter(episode)

    from recsys_mdp.mdp.mdp_former import MDPFormer
    mdp_preparator_class = MDPFormer(
        load_from_file=False, dataframe=data,
        framestack=framestack_size,
        reward_function=reward_function,
        action_function=action_function,
        episode_splitter=episode_splitter,
    )
    return mdp_preparator_class
