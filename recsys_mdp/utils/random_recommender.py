import numpy as np
from numpy.random import Generator


class RandomRecommender:
    n_actions: int
    batch_size: int
    rng: Generator

    def __init__(self, seed: int, n_actions: int, batch_size: int, **_):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.batch_size = batch_size

    @staticmethod
    def fitter(n_epochs: int, **_):
        for ep in range(n_epochs):
            yield ep, None

    def predict(self, batch: np.ndarray = None):
        if batch is not None:
            return self.rng.integers(0, self.n_actions, size=batch.shape[0])
        else:
            return self.rng.integers(0, self.n_actions)
