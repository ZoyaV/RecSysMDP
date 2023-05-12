from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd


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
        # проверка формата столбца с временем, в некоторых датасетка не unix
        is_not_unix_timestamp = ~pd.to_datetime(col, errors='coerce').notnull().all()
        is_string_column = col.dtype == 'object'

        if is_not_unix_timestamp or is_string_column:
            timestamp = pd.to_datetime(col)
            col = pd.to_datetime(timestamp).astype(int) // 10**9
        col_values = col.values
        # Calculate time differences in minutes using np.diff
        time_diff = np.diff(col_values).astype('timedelta64[s]')
        pause_minutes = time_diff / 60
        threshold_timedelta = np.timedelta64(int(threshold_minutes), 'm')

        # Compare the pause_minutes array with the threshold using np.greater
        is_above_threshold = np.greater(pause_minutes, threshold_timedelta)
        return is_above_threshold

    return split_by_column_condition(
        user_log, col_name=col_mapping['timestamp_col_name'], condition=pause_condition
    )


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


def split_by_user(user_log, col_mapping):
    """
    Divides story by episodes - entire user story = 1 episode

    :param user_log: pandas array of one user interaction history
    :param col_mapping: dict with names of columns
    :return: indices of transitions to a new episode
    """

    def entire_log_condition(col: pd.Series):
        return np.full_like(col.values, False)

    return split_by_column_condition(
        user_log, col_name=col_mapping['timestamp_col_name'], condition=entire_log_condition
    )
