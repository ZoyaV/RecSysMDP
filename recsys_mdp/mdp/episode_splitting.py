from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from recsys_mdp.mdp.base import TERMINATE_COL, TIMESTAMP_COL


def by_pause(user_log: pd.DataFrame, threshold_minutes: int = 20):
    """
    Divides the user's log into separate episodes where the pause duration
    between two consecutive interactions in an episode is under the passed threshold.

    :param user_log: pandas array of a single user interaction history
    :param threshold_minutes: int with the pause in minutes required for a new episode starting
    :return: indices of transitions to a new episode
    """
    def pause_condition(col: pd.Series):
        pause_minutes = col.diff(1).dt.total_seconds().div(60).values

        # NB: [0] element is NaN
        split_mask = pause_minutes > threshold_minutes
        split_mask[0] = False
        return split_mask

    return split_by_column_condition(
        user_log, col_name=TIMESTAMP_COL, condition=pause_condition
    )


def by_user(user_log: pd.DataFrame):
    """
    Divides story by episodes - entire user story = 1 episode

    :param user_log: pandas array of one user interaction history
    :return: indices of transitions to a new episode
    """
    return split_by_column_condition(
        user_log, col_name=TIMESTAMP_COL, condition=no_split_const_condition
    )


def by_terminate(user_log: pd.DataFrame):
    """Divides story into episodes as they were originally generated."""
    if TERMINATE_COL not in user_log.columns:
        # no split
        return split_by_column_condition(
            user_log, col_name=TIMESTAMP_COL, condition=no_split_const_condition
        )

    def terminate_condition(col: pd.Series):
        return col.values

    return split_by_column_condition(
        user_log, col_name=TERMINATE_COL, condition=terminate_condition
    )


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
    split_indices = np.argwhere(split_mask).flatten()
    return split_indices
