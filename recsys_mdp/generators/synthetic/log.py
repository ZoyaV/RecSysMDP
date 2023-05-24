from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator


class RandomLogGenerator:
    rng: Generator
    n_users: int
    n_items: int
    n_pairs: int

    def __init__(
            self, seed: int, n_users: int, n_items: int, n_pairs: int | float,
            duplicates: bool = False
    ):
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_pairs = self._to_pairs(n_pairs)
        self.duplicates = duplicates

    def generate(self, n_pairs: int | float = None, duplicates: bool = None) -> pd.DataFrame:
        n_pairs = self._to_pairs(n_pairs) if n_pairs is not None else self.n_pairs
        duplicates = duplicates if duplicates is not None else self.duplicates

        log_pairs = self.rng.choice(
            self.n_users * self.n_items,
            size=n_pairs,
            replace=duplicates
        )
        # timestamps denote the order of interactions, this is not the real timestamp
        timestamps = self.rng.uniform(size=n_pairs)
        log_users, log_items = np.divmod(log_pairs, self.n_items)
        log = pd.DataFrame({
            'user_id': log_users,
            'item_id': log_items,
            'timestamp': timestamps,
        })
        log.sort_values(
            ['user_id', 'timestamp'],
            inplace=True,
            ascending=[True, False]
        )
        return log

    def _to_pairs(self, n_pairs: int | float):
        return n_pairs if isinstance(n_pairs, int) else int(n_pairs * self.n_users * self.n_items)
