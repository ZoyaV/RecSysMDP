from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from recsys_mdp.mdp.base import TERMINATE_COL, TIMESTAMP_COL, TRUNCATED_COL


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

    return split_by_condition(user_log[TIMESTAMP_COL], condition=pause_condition)


def by_user(user_log: pd.DataFrame):
    """
    Divides story by episodes - entire user story = 1 episode

    :param user_log: pandas array of one user interaction history
    :return: indices of transitions to a new episode
    """
    return split_by_condition(user_log, condition=no_split_const_condition)


def by_terminate_or_truncate(user_log: pd.DataFrame):
    """Divides story into episodes as they were originally generated."""
    if TERMINATE_COL not in user_log.columns:
        # no split
        return split_by_condition(user_log[TIMESTAMP_COL], condition=no_split_const_condition)

    def terminate_condition(log_df: pd.DataFrame):
        return np.logical_or(log_df[TERMINATE_COL], log_df[TRUNCATED_COL])

    return split_by_condition(user_log, condition=terminate_condition)


def no_split_const_condition(log_df: pd.Series | pd.DataFrame):
    return np.full(log_df.shape[0], False)


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


def split_by_condition(
        log_df: pd.DataFrame,
        condition: Callable[[pd.Series | pd.DataFrame], np.ndarray | list[bool]]
):
    """
    Split the user's log into episodes by the condition set on one of the log's columns.

    :param log_df: pandas array or dataframe of a single user interaction history
    :param condition: Callable that applies condition logic for splitting
    :return: np.ndarray with indices of transitions to a new episode
    """
    split_mask = condition(log_df)
    split_indices = np.argwhere(split_mask).flatten()
    return split_indices
