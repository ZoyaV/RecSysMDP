from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from replay.models.rl.experiments.datasets.synthetic.relevance import RelevanceCalculator
from replay.models.rl.experiments.utils.config import GlobalConfig, TConfig


@dataclasses.dataclass
class ToyRatingsDataset:
    log: pd.DataFrame
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    @property
    def log_user_embeddings(self):
        return self.user_embeddings[self.log['user_id']]

    @property
    def log_item_embeddings(self):
        return self.item_embeddings[self.log['item_id']]

    @property
    def log_continuous_ratings(self):
        return self.log['continuous_rating']

    @property
    def log_discrete_ratings(self):
        return self.log['discrete_rating']

    @property
    def log_gt_continuous_ratings(self):
        return self.log['gt_continuous_rating']

    @property
    def log_gt_discrete_ratings(self):
        return self.log['gt_discrete_rating']

    @property
    def log_ground_truth(self):
        return self.log['ground_truth']


class ToyRatingsDatasetBuilder:
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray

    def __init__(
            self, global_config: GlobalConfig, seed: int, source: TConfig,
            embeddings_n_dims: int, user_embeddings: TConfig, item_embeddings: TConfig,
            ratings: TConfig
    ):
        self.global_config = global_config
        self.rng = np.random.default_rng(seed)
        self.source = global_config.resolve_object(source)
        self.embeddings_n_dims = embeddings_n_dims

        user_embeddings_generator = global_config.resolve_object(
            user_embeddings, n_dims=embeddings_n_dims
        )
        item_embeddings_generator = global_config.resolve_object(
            item_embeddings, n_dims=embeddings_n_dims
        )
        self.user_embeddings = user_embeddings_generator.generate(self.source.n_users)
        self.item_embeddings = item_embeddings_generator.generate(self.source.n_items)
        self.relevance = RelevanceCalculator(**ratings)

    def generate(self) -> ToyRatingsDataset:
        dataset = ToyRatingsDataset(
            self.source.generate(),
            user_embeddings=self.user_embeddings, item_embeddings=self.item_embeddings
        )
        continuous_ratings, discrete_ratings = self.relevance.calculate(
            dataset.log_user_embeddings, dataset.log_item_embeddings
        )
        dataset.log['continuous_rating'] = continuous_ratings
        dataset.log['discrete_rating'] = discrete_ratings
        dataset.log['gt_continuous_rating'] = continuous_ratings
        dataset.log['gt_discrete_rating'] = discrete_ratings
        dataset.log['ground_truth'] = True
        return dataset

    def generate_negative_samples(
            self, n_pairs: int | float, negate: str
    ) -> ToyRatingsDataset:
        dataset = ToyRatingsDataset(
            self.source.generate(n_pairs=n_pairs, duplicates=True),
            user_embeddings=self.user_embeddings, item_embeddings=self.item_embeddings
        )
        gt_continuous_ratings, gt_discrete_ratings = self.relevance.calculate(
            dataset.log_user_embeddings, dataset.log_item_embeddings
        )
        if negate == 'continuous':
            continuous_ratings = self.rng.uniform(size=gt_continuous_ratings.shape[0])
            discrete_ratings = self.relevance.discretize(continuous_ratings)
        elif negate == 'discrete':
            discrete_ratings = 1 - gt_discrete_ratings
            # sample continuous rating from the corresponding relevance range
            # precursors: just random values in [0,1]
            continuous_ratings = self.rng.uniform(size=gt_continuous_ratings.shape[0])
            # for irrelevant: [0, 1] -> [0, \theta]
            continuous_ratings[discrete_ratings == 0] *= self.relevance.relevant_threshold
            # for relevant: [0, 1] -> [0, 1 - \theta] -> [\theta, 1]
            continuous_ratings[discrete_ratings == 1] *= 1 - self.relevance.relevant_threshold
            continuous_ratings[discrete_ratings == 1] += self.relevance.relevant_threshold
        else:
            raise ValueError(f'Parameter `negate` has unsupported value {negate}')

        dataset.log['continuous_rating'] = continuous_ratings
        dataset.log['discrete_rating'] = discrete_ratings
        dataset.log['gt_continuous_rating'] = gt_continuous_ratings
        dataset.log['gt_discrete_rating'] = gt_discrete_ratings
        dataset.log['ground_truth'] = False
        return dataset

    def split(self, dataset: ToyRatingsDataset, split_by, train: float) -> ToyRatingsDataset:
        if split_by == 'user':
            n_train_users = int(train * self.source.n_users)
            train_users = self.rng.choice(self.source.n_users, size=n_train_users, replace=False)
            return ToyRatingsDataset(
                dataset.log[dataset.log['user_id'].isin(train_users)].copy(),
                user_embeddings=self.user_embeddings,
                item_embeddings=self.item_embeddings
            )
        raise ValueError(f'Unknown split-by type: {split_by}')
