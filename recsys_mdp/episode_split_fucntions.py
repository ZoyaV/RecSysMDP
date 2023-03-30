import pandas as pd
import numpy as np
def split_by_time(user_log, col_mapping):
    """
    Divides the story into episodes, taking into account the interruption of the user's interaction with the service

    :param user_log: pandas array of one user interaction history
    :param col_mapping: dict with names of columns
    :return: indices of transitions to a new episode
    """
    ts_name = col_mapping['timestamp_col_name']
    ts = pd.to_datetime(user_log[ts_name]).astype(int) // 10 ** 9
    condition_music = lambda A: A > 100
    result = (ts[1:].values - ts[:-1].values).astype(int)
    indx = np.where(condition_music(result))
    if len(indx[0]) == 0:
        return [0, -1]
    return indx[0]


def split_by_user(user_log, col_mapping):
    """
    Divides story by episodes - entire user story = 1 episode

    :param user_log: pandas array of one user interaction history
    :param col_mapping: dict with names of columns
    :return: indices of transitions to a new episode
    """
    return [0, len(user_log)]
