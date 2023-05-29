import numpy as np

from recsys_mdp.mdp.base import ITEM_ID_COL, RELEVANCE_CONT_COL, RELEVANCE_INT_COL


def discrete_relevance_action(df):
    actions = df[RELEVANCE_INT_COL]
    return actions.values


def next_item_action(df):
    actions = df[ITEM_ID_COL]
    return actions.values


def continuous_relevance_action(df):
    # TODO: scale should be customizable
    actions = df[RELEVANCE_CONT_COL]
    # TODO: make random deterministic
    noise = np.random.normal(scale=0.01, size=actions.shape)
    return ((actions.values + noise) - 2) / 5
