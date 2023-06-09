from __future__ import annotations

from typing import Any

import numpy as np
from d3rlpy.algos import DiscreteRandomPolicy

from recsys_mdp.utils.run.config import resolve_absolute_quantity


class FastNearestNeighbor(DiscreteRandomPolicy):
    def __init__(
            self, *, env, seed: int,
            n_samples: int | float, kth_best: int | float,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.env = env
        self.rng = np.random.default_rng(seed)

        n_samples = resolve_absolute_quantity(n_samples, baseline=self.env.n_items)
        self.n_samples = max(1, n_samples)

        assert 0 < kth_best < self.n_samples
        self.kth_best = kth_best - 1

    def sample_action(self, x: np.ndarray | list) -> np.ndarray:
        x = np.asarray(x)
        if x.shape[0] > 1:
            return super().sample_action(x)

        env = self.env
        sampled_items = self.rng.choice(env.n_items, size=self.n_samples)
        relevance = np.array([
            env.state.relevance(item_id=item_id, with_satiation=True)[0]
            for item_id in sampled_items
        ])
        ranked_items = sampled_items[np.argsort(-relevance)]
        # take 1-element slice to return array instead of a single int value
        return ranked_items[self.kth_best:self.kth_best+1]

    def fitter(self, dataset=None, n_epochs: int = None, **kwargs):
        assert n_epochs is not None
        for epoch in range(1, n_epochs+1):
            yield epoch, 0.
