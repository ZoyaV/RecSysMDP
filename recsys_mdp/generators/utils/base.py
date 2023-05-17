import numpy as np
from numpy.random import Generator


def sample_int(rng: Generator):
    return rng.integers(100_000_000)


def sample_rng(rng: Generator):
    return np.random.default_rng(sample_int(rng))
