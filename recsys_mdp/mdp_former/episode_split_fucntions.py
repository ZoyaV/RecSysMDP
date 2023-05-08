from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def split_by_time(user_log, col_mapping):
    """
    Divides the story into episodes, taking into account the interruption
    of the user's interaction with the service

    :param user_log: pandas array of one user interaction history
    :param col_mapping: dict with names of columns
    :return: indices of transitions to a new episode
    """
    def pause_condition(col: pd.Series):
        pause_minutes = (col.iloc[1:] - col.iloc[:-1]).dt.total_seconds() / 60
        return pause_minutes.values > 20

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
    :return: indices of transitions to a new episode
    """
    split_mask = condition(user_log[col_name])
    split_indices = np.argwhere(split_mask).tolist() + [-1]
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
