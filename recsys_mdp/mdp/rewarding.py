import numpy as np

from recsys_mdp.mdp.base import RATING_COL


def monotony_reward(df):
    rewards = np.ones(df.shape[0])
    return rewards


def condition_reward(df):
    # TODO: condition should be customizable
    rewards = np.zeros(df.shape[0])
    rewards[df[RATING_COL] <= 3] = 1
    rewards[df[RATING_COL] > 3] = -0.1
    return rewards

def summary_reward(df):
    rewards = relevance_based_reward(df)
    sum_reward = np.zeros(df.shape[0])
    sum_reward[-1] = np.sum(rewards)
    return sum_reward

def ones_reward(df):
    sum_reward = np.ones(df.shape[0])
    return sum_reward
def relevance_based_reward(df):
    rewards = np.zeros(df.shape[0])
    rewards[df[RATING_COL] <= 3] = -1
    rewards[df[RATING_COL] > 3] = 1
    rewards[df[RATING_COL] == 5] = 2
    return rewards

# добавить загрузку ембедингов
# поставить экспериментов