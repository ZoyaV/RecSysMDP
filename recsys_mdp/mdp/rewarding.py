import numpy as np

from recsys_mdp.mdp.base import RATING_COL

# TODO: convert to class objects with __call__(episode)
# FIXME: check that relevance based rewards are for discrete ratings


def irrelevant(episode):
    # TODO: condition should be customizable
    rewards = np.zeros(episode.shape[0])
    rewards[episode[RATING_COL] <= 3] = 1
    rewards[episode[RATING_COL] > 3] = -0.1
    return rewards


def relevant(episode):
    rewards = np.zeros(episode.shape[0])
    rewards[episode[RATING_COL] <= 3] = -1
    rewards[episode[RATING_COL] > 3] = 1
    rewards[episode[RATING_COL] == 5] = 2
    return rewards


def const_one(episode):
    rewards = np.ones(episode.shape[0])
    return rewards


def sparse_return(episode):
    dense_rewards = relevant(episode)

    sparse_rewards = np.zeros(episode.shape[0])
    sparse_rewards[-1] = np.sum(dense_rewards)
    return sparse_rewards


def sparse_episode_length(episode):
    sparse_rewards = np.zeros(episode.shape[0])
    sparse_rewards[-1] = episode.shape[0]
    return sparse_rewards
