from __future__ import annotations

from typing import Any

import pandas as pd


class OfflineEmbeddings:
    n_dims: int
    learn: bool
    users: Any | None
    items: Any | None

    def __init__(self, n_dims: int, learn: bool, init_with_als: bool, **embeddings):
        self.n_dims = n_dims
        self.learn = learn
        self.users, self.items = None, None

        self.encoder = None
        if init_with_als:
            from recsys_mdp.models.als import AlsEmbeddings
            self.encoder = AlsEmbeddings(n_dims=n_dims, **embeddings)

    def fit(self, log_df: pd.DataFrame):
        if self.encoder is None:
            return
        self.users, self.items = self.encoder.fit_embeddings_on_dataframe(log_df)
