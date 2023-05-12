from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd


def split_by_time(user_log: pd.DataFrame, col_mapping: dict, threshold_minutes: int = 20):
    """
    Divides the user's log into separate episodes where the pause duration
    between two consecutive interactions in an episode is under the passed threshold.

    :param user_log: pandas array of a single user interaction history
    :param col_mapping: dict with names of columns
    :param threshold_minutes: int with the pause in minutes required for a new episode starting
    :return: indices of transitions to a new episode
    """
    def pause_condition(col: pd.Series):
        pause_minutes = (col.iloc[1:] - col.iloc[:-1]).dt.total_seconds() / 60

        split_mask = np.empty_like(col.values)
        split_mask[0] = False
        split_mask[1:] = pause_minutes.values > threshold_minutes
        return split_mask

    return split_by_column_condition(
        user_log, col_name=col_mapping['timestamp_col_name'], condition=pause_condition
    )


def split_by_user(user_log: pd.DataFrame, col_mapping):
    """
    Divides story by episodes - entire user story = 1 episode

    :param user_log: pandas array of one user interaction history
    :param col_mapping: dict with names of columns
    :return: indices of transitions to a new episode
    """
    return split_by_column_condition(
        user_log, col_name=col_mapping['timestamp_col_name'], condition=no_split_const_condition
    )


def split_by_generated_episodes(user_log: pd.DataFrame, col_mapping):
    """Divides story into episodes as they were originally generated."""
    terminal_col = 'terminal'
    if terminal_col not in user_log.columns:
        # no split
        return split_by_column_condition(
            user_log, col_name=col_mapping['timestamp_col_name'],
            condition=no_split_const_condition
        )

    def terminal_condition(col: pd.Series):
        return col.values

    return split_by_column_condition(user_log, col_name=terminal_col, condition=terminal_condition)


def no_split_const_condition(col: pd.Series):
    return np.full_like(col.values, False)


def to_episode_ranges(
        user_log: pd.DataFrame, split_indices: list[int]
) -> Iterable[tuple[int, int]]:
    # noinspection GrazieInspection
    """
    Transform indices declaring splits of user's log into episodes
    to a list of tuples [start_ind, end_ind) declaring each episode range.
    """
    ep_start_ind = 0
    for ep_end_ind in split_indices:
        yield ep_start_ind, ep_end_ind
        ep_start_ind = ep_end_ind

    ep_end_ind = len(user_log)
    yield ep_start_ind, ep_end_ind


def split_by_column_condition(
        user_log: pd.DataFrame,
        col_name: str,
        condition: Callable[[pd.Series], np.ndarray | list[bool]]
):
    """
    Split the user's log into episodes by the condition set on one of the log's columns.

    :param user_log: pandas array of one user interaction history
    :param col_name: str name of a column by which the split is implemented
    :param condition: Callable[[pd.Series], bool] that applies condition logic for splitting
    :return: np.ndarray with indices of transitions to a new episode
    """
    split_mask = condition(user_log[col_name])
    split_indices = np.argwhere(split_mask)
    return split_indices
