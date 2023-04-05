import pandas as pd
import numpy as np

def discrete_relevance_action(df, key_mapping=None):
    actions = df[key_mapping['reward_col_name']]
    return actions.values
def next_item_action(df, key_mapping=None):
    actions = df[key_mapping['item_col_name']].values.astype(int)
    return actions
def continuous_relevance_action(df, key_mapping=None):
    # TODO: scale shuld be customizable
    actions = df[key_mapping['reward_col_name']]
    noise = np.random.normal(scale=0.01, size=actions.shape)
    return (((actions.values + noise) - 2) / 5)
