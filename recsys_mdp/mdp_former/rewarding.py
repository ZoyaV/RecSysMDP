import pandas as pd
import numpy as np

def monotony_reward(df, key_mapping=None):
    rewards = np.ones(df.shape[0])
    return rewards
def condition_reward(df, key_mapping=None):
    #TODO: condition shuld be customizable
    rewards = np.zeros(df.shape[0])
    rewards[df[key_mapping['reward_col_name']] <= 3] = 1
    rewards[df[key_mapping['reward_col_name']] > 3] = -0.1
    return rewards

def relevance_based_reward(df, key_mapping=None):
    rewards = np.zeros(df.shape[0])
    rewards[df[key_mapping['reward_col_name']] <= 3] = -1
    rewards[df[key_mapping['reward_col_name']] > 3] = 1
    rewards[df[key_mapping['reward_col_name']] == 5] = 2

    return rewards

#добавить загрузку ембедингов
#поставить экспериментов