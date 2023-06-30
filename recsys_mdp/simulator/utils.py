from __future__ import annotations

import numpy as np
from numpy.random import Generator

from recsys_mdp.utils.base import make_rng, sample_int


# noinspection PyPep8Naming
def boosting(
        relative_value: float | np.ndarray, k: float,
        softness: float = 1.5
) -> float:
    """
    Computes boosting scale coefficient depending on `relative_value`:
        0       --> k + 1
        1       --> 1
        +inf    --> 1 / (k + 1)

    Higher softness just makes the sigmoid curve smoother;
    default value is empirically optimized.
    """

    # relative value: value / target value  \in [0, +inf)
    # x = -log(relative_rate)
    #   0 1 +inf  -> +inf 0 -inf
    x = -np.log(relative_value)
    # zero k means "no boosting", that's why we use shifted value.
    K = k + 1

    # relative_rate -> x -> B:
    #   0 -> +inf -> K^tanh(+inf) = K^1 = K
    #   1 -> 0 -> K^tanh(0) = K^0 = 1
    #   +inf -> -inf -> K^tanh(-inf) = K^(-1) = 1 / K
    return np.power(K, np.tanh(x / softness))


def normalize(x: np.ndarray) -> np.ndarray:
    normalizer = x.sum(-1)
    assert normalizer > 1e-8, f'Normalization is dangerous for {x}'
    return x / normalizer


def softmax(x: np.ndarray, temp=.06) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)


class EpisodicRandomGenerator:
    """
    Seed and rng "live" only a single episode, then they're switched to the next determined pair.
    This class wraps this pair and is able to switch it to a new pair.
    """
    seed: int
    rng: Generator

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = make_rng(self.seed)

    def transit_to_next_episode(self):
        self.seed = sample_int(make_rng(self.seed))
        self.rng = make_rng(self.seed)
