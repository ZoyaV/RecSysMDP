from __future__ import annotations

import numpy as np
from numpy.random import Generator

from replay.models.rl.experiments.datasets.synthetic.clusters import generate_clusters


class RandomEmbeddingsGenerator:
    rng: Generator
    n_dims: int

    def __init__(self, seed: int, n_dims: int):
        self.rng = np.random.default_rng(seed)
        self.n_dims = n_dims
        self.n_clusters = 1
        self.clusters = np.full(n_dims, 0.5)

    def generate(self, n: int = None) -> np.ndarray:
        shape = (n, self.n_dims) if n is not None else (self.n_dims,)
        self.n_clusters = n
        self.clusters = self.rng.uniform(size=shape)
        return self.clusters


class RandomClustersEmbeddingsGenerator:
    rng: Generator
    n_dims: int
    intra_cluster_noise_scale: float

    clusters: np.ndarray

    def __init__(
            self, seed: int, n_dims: int, n_clusters: int | list[int],
            intra_cluster_noise_scale: float = 0.05,
            n_dissimilar_dims_required: int = 3,
            min_dim_delta: float = 0.3,
            min_l2_dist: float = 0.1,
            max_generation_tries: int = 10000
    ):
        self.rng = np.random.default_rng(seed)
        self.n_dims = n_dims
        self.intra_cluster_noise_scale = intra_cluster_noise_scale
        self.clusters = generate_clusters(
            self.rng, n_clusters, n_dims,
            n_dissimilar_dims_required=n_dissimilar_dims_required,
            min_dim_delta=min_dim_delta,
            min_l2_dist=min_l2_dist,
            max_tries=max_generation_tries,
        )
        self.n_clusters = len(self.clusters)

    def generate(self, n: int = None) -> np.ndarray:
        if n is None:
            return self.generate_one()
        return np.array([self.generate_one() for _ in range(n)])

    def generate_one(self) -> np.ndarray:
        cluster = self.rng.choice(self.clusters)
        embedding = self.rng.normal(
            loc=cluster, scale=self.intra_cluster_noise_scale, size=(self.n_dims,)
        )
        return np.clip(embedding, 0.0, 1.0)
