from __future__ import annotations

from collections import deque

import numpy as np

from recsys_mdp.simulator.env import NextItemEnvironment


class Framestack:
    size: int

    empty_item: int
    empty_rating: int | float
    discrete: bool

    items: deque[int]
    ratings: deque[int | float]
    user_id: int

    def __init__(
            self, size: int, discrete: bool,
            empty_item: int, empty_rating: tuple[float, int]
    ):
        self.size = size
        self.discrete = discrete
        self.empty_item = empty_item
        self.empty_rating = empty_rating[discrete]
        self.items = deque(maxlen=self.size)
        self.ratings = deque(maxlen=self.size)

    def reset(self, user_id: int):
        self.user_id = user_id
        for _ in range(self.size):
            self._append(self.empty_item, self.empty_rating)
        return self.observation()

    def step(self, item_id: int, continuous_relevance, discrete_relevance):
        rating = (continuous_relevance, discrete_relevance)[self.discrete]
        self._append(item_id, rating)
        return self.observation()

    def observation(self):
        return dict(
            items=np.array(self.items),
            ratings=np.array(self.ratings),
            user=np.array([self.user_id])
        )

    @staticmethod
    def compile_observation(observation: dict, keys):
        return np.concatenate([
            observation[key] for key in keys
        ])

    def _append(self, item_id, rating):
        self.items.append(item_id)
        self.ratings.append(rating)
