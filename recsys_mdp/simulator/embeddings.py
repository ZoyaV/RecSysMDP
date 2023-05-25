from __future__ import annotations

import numpy as np
from numpy.random import Generator

from recsys_mdp.simulator.clusters import generate_clusters


class RandomEmbeddingsGenerator:
    rng: Generator
    n_dims: int

    def __init__(self, seed: int, n_dims: int):
        self.rng = np.random.default_rng(seed)
        self.n_dims = n_dims
        self.n_clusters = 1
        _, self.clusters = np.full(n_dims, 0.5)

    def generate(self, n: int = None) -> tuple[int, np.ndarray] | tuple[np.ndarray, np.ndarray]:
        shape = (n, self.n_dims) if n is not None else (self.n_dims,)
        if n is None:
            n = 1
        self.n_clusters = n
        self.clusters = self.rng.uniform(size=shape)
        return np.arange(n), self.clusters


class RandomClustersEmbeddingsGenerator:
    rng: Generator
    n_dims: int
    intra_cluster_noise_scale: float

    clusters: np.ndarray

    def __init__(
            self, seed: int, n_dims: int, n_clusters: int | list[int],
            cluster_sampling_weight: dict[int, float] = None,
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
        self.cluster_sampling_weights = np.ones(self.n_clusters)
        if cluster_sampling_weight is not None:
            for cluster, weight in cluster_sampling_weight.items():
                self.cluster_sampling_weights[cluster] = weight
        self.cluster_sampling_weights /= self.cluster_sampling_weights.sum()

    def generate(self, n: int = None) -> tuple[int, np.ndarray] | tuple[np.ndarray, np.ndarray]:
        if n is None:
            return self.generate_one()

        result = [self.generate_one() for _ in range(n)]
        clusters = np.array([cluster_ind for cluster_ind, _ in result])
        embeddings = np.array([embedding for _, embedding in result])
        return clusters, embeddings

    def generate_one(self, cluster_ind=None) -> tuple[int, np.ndarray]:
        if cluster_ind is None:
            cluster_ind = self.rng.choice(self.n_clusters, p=self.cluster_sampling_weights)

        cluster = self.clusters[cluster_ind]
        embedding = self.rng.normal(
            loc=cluster, scale=self.intra_cluster_noise_scale, size=(self.n_dims,)
        )
        return cluster_ind, np.clip(embedding, 0.0, 1.0)
