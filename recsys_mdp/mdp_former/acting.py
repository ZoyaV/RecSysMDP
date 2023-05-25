import numpy as np

from recsys_mdp.mdp_former.base import RATING_COL, ITEM_ID_COL


def discrete_relevance_action(df):
    actions = df[RATING_COL]
    return actions.values


def next_item_action(df):
    actions = df[ITEM_ID_COL]
    return actions.values


def continuous_relevance_action(df):
    # TODO: scale should be customizable
    actions = df[RATING_COL]
    noise = np.random.normal(scale=0.01, size=actions.shape)
    return ((actions.values + noise) - 2) / 5
