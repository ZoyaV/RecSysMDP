from __future__ import annotations

import numpy as np


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


def softmax(x: np.ndarray, temp=.12) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)
