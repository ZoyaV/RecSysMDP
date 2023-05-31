from __future__ import annotations

from typing import Any

import pandas as pd


class OfflineEmbeddings:
    size: int
    users: Any | None
    items: Any | None

    def __init__(self, init_with_als, size, **embeddings):
        self.size = size
        self.embedder = None
        self.users, self.items = None, None

        if init_with_als:
            from recsys_mdp.models.als import AlsEmbeddings
            self.embedder = AlsEmbeddings(
                embeddings_size=size, **embeddings
            )

    def fit(self, log_df: pd.DataFrame):
        if self.embedder is None:
            return

        self.users, self.items = self.embedder.fit_embeddings_on_dataframe(log_df)
