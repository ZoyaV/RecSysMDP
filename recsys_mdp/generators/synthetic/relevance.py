from __future__ import annotations

import numpy as np


class RelevanceCalculator:
    metric: str
    positive_ratio: float

    def __init__(self, metric: str, positive_ratio: float):
        self.metric = metric
        self.positive_ratio = positive_ratio
        self.relevant_threshold = None

    def calculate(self, users: np.ndarray, items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        continuous_relevance = similarity(users, items, metric=self.metric)
        if self.relevant_threshold is None:
            self.relevant_threshold = np.quantile(
                continuous_relevance, 1 - self.positive_ratio,
                interpolation='lower'
            )

        discrete_relevance = self.discretize(continuous_relevance)
        return continuous_relevance, discrete_relevance

    def discretize(self, relevance: np.ndarray) -> np.ndarray:
        return (relevance >= self.relevant_threshold).astype(int)


def similarity(users: np.ndarray, items: np.ndarray, metric: str) -> np.ndarray | float:
    if metric == 'l1':
        d = users - items
        return 1.0 - np.abs(d).mean(axis=-1)
    elif metric == 'l2':
        d = users - items
        avg_sq_d = (d ** 2).mean(axis=-1)
        return 1.0 - np.sqrt(avg_sq_d)
    elif metric == 'cosine':
        dot_product = np.sum(users * items, axis=-1)
        users_norm = np.linalg.norm(users, axis=-1)
        items_norm = np.linalg.norm(items, axis=-1)
        return dot_product / (users_norm * items_norm)
    raise ValueError(f'Unknown similarity metric: {metric}')
