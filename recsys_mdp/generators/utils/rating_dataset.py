from __future__ import annotations

import time

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset


class RatingDataset:
    init_time: float

    name: str
    k: int
    test_ratio: float
    use_negative_events: bool
    rating_based_reward: bool
    reward_top_k: bool
    action_randomization_scale: float
    rating_actions: bool

    raw_rating_to_reward_rescale: dict
    binary_rating_to_reward_rescale: dict

    def __init__(
            self, *, name: str, k: int, test_ratio: float,
            use_negative_events: bool,
            rating_based_reward: bool, reward_top_k: bool,
            action_randomization_scale: float,
            rating_actions: bool,
            raw_rating_to_reward_rescale: dict,
            binary_rating_to_reward_rescale: dict,
    ):
        self.init_time = time.time()
        self.print_with_timestamp('==> Dataset init')
        self.name = name
        self.k = k
        self.test_ratio = test_ratio
        self.use_negative_events = use_negative_events
        self.rating_based_reward = rating_based_reward
        self.reward_top_k = reward_top_k
        self.action_randomization_scale = action_randomization_scale
        self.rating_actions = rating_actions
        self.raw_rating_to_reward_rescale = raw_rating_to_reward_rescale
        self.binary_rating_to_reward_rescale = binary_rating_to_reward_rescale

    def prepare(self):
        self.print_with_timestamp('==> Prepare dataset')

        log = self.prepare_raw_dataset()
        dataset = self.prepare_mdp_dataset(log)

        self.print_with_timestamp('<==')
        return dataset

    def print_with_timestamp(self, text: str):
        print(f'[{time.time() - self.init_time:5.1f}] {text}')

    def prepare_mdp_dataset(self, log: pd.DataFrame):
        if not self.use_negative_events:
            # remove negative events
            log = log[log['rating'] >= 3]

        user_logs = log.sort_values(['user_id', 'timestamp'], ascending=True)

        if self.rating_based_reward:
            rescale = self.raw_rating_to_reward_rescale
        else:
            rescale = self.binary_rating_to_reward_rescale
        rewards = user_logs['rating'].map(rescale).to_numpy()

        if self.reward_top_k:
            # additionally reward top-K watched movies
            user_top_k_idxs = (

                user_logs
                .sort_values(['rating', 'timestamp'], ascending=[False, True])
                .groupby('user_id')
                .head(self.k)
                .index
            )
            # rescale positives and additionally reward top-K watched movies
            rewards[rewards > 0] /= 2
            rewards[user_top_k_idxs] += 0.5

        user_logs['rewards'] = rewards

        # every user has his own episode (the latest item is defined as terminal)
        user_terminal_idxs = (
            user_logs[::-1]
            .groupby('user_id')
            .head(1)
            .index
        )
        terminals = np.zeros(len(user_logs))
        terminals[user_terminal_idxs] = 1
        user_logs['terminals'] = terminals

        actions = user_logs['rating'].to_numpy()
        if not self.rating_actions:
            actions = (actions >= 3).astype(int)

        # train/test split
        split_idx = int(len(user_logs) * self.test_ratio)
        user_logs_train = user_logs.iloc[:-split_idx]
        user_logs_test = user_logs.iloc[-split_idx:]

        train_dataset = MDPDataset(
            observations=np.array(user_logs_train[['user_id', 'item_id']]),
            actions=actions[:, None] ,
            rewards=user_logs_train['rewards'],
            terminals=user_logs_train['terminals']
        )
        test_dataset = MDPDataset(
            observations=np.array(user_logs_test[['user_id', 'item_id']]),
            actions=actions[:, None],
            rewards=user_logs_test['rewards'],
            terminals=user_logs_test['terminals']
        )
        return train_dataset, user_logs_train, test_dataset, user_logs_test

    def prepare_raw_dataset(self):
        name, category = self.name.split('.')
        if name == 'MovieLens':
            from rs_datasets import MovieLens
            ds = MovieLens(version=category)
        elif name == 'Amazon':
            from rs_datasets import Amazon
            ds = Amazon(category=category)
        else:
            raise KeyError()
        return ds.ratings
