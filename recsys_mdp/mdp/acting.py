import numpy as np

from recsys_mdp.mdp.base import ITEM_ID_COL, RELEVANCE_CONT_COL, RELEVANCE_INT_COL


def discrete_rating(episode):
    actions = episode[RELEVANCE_INT_COL]
    return actions.values


def continuous_rating(df):
    # TODO: scale should be customizable
    actions = df[RELEVANCE_CONT_COL]
    # TODO: make random deterministic
    noise = np.random.normal(scale=0.01, size=actions.shape)
    return ((actions.values + noise) - 2) / 5


def item(episode):
    actions = episode[ITEM_ID_COL]
    return actions.values
