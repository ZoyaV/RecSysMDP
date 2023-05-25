from __future__ import annotations

import numpy as np
from d3rlpy.dataset import MDPDataset

from recsys_mdp.simulator.dataset import ToyRatingsDataset


class MdpDatasetBuilder:
    actions: str
    rewards: dict

    def __init__(self, actions: str, rewards: dict):
        self.actions = actions
        self.rewards = rewards

    def build(self, ds: ToyRatingsDataset, use_ground_truth: bool = False) -> MDPDataset:
        observations = np.concatenate((ds.log_user_embeddings, ds.log_item_embeddings), axis=-1)
        # FIXME: it's dangerous, better to pass already correct dataset
        actions = self._get_actions(ds, use_ground_truth=use_ground_truth)
        rewards = self._get_rewards(ds)
        terminals = self._get_terminals(ds)

        return MDPDataset(
            observations=observations,
            actions=np.expand_dims(actions, axis=-1),
            rewards=rewards,
            terminals=terminals,
        )

    def _get_rewards(self, ds):
        rewards = np.zeros(ds.log.shape[0])
        for name, value in self.rewards.items():
            if name == 'baseline':
                baseline = np.array(value)
                rewards += baseline[ds.log_ground_truth.values.astype(int)]
            elif name == 'continuous':
                weight: float = value
                rewards[ds.log_ground_truth] += (
                    ds.log_continuous_ratings[ds.log_ground_truth] * weight
                )
            elif name == 'discrete':
                weights = np.array(value)
                rewards[ds.log_ground_truth] += weights[
                    ds.log_discrete_ratings[ds.log_ground_truth]
                ]
            elif name == 'continuous_error':
                weight: float = value
                err = np.abs(ds.log_continuous_ratings - ds.log['gt_continuous_rating'])
                rewards[~ds.log_ground_truth] += err[~ds.log_ground_truth] * weight
            elif name == 'discrete_error':
                weight: float = value
                err = np.abs(ds.log_discrete_ratings - ds.log['gt_discrete_rating'])
                rewards[~ds.log_ground_truth] += err[~ds.log_ground_truth] * weight
        return rewards

    def _get_actions(self, ds, use_ground_truth: bool):
        if self.actions == 'continuous':
            if use_ground_truth:
                return ds.log_gt_continuous_ratings
            else:
                return ds.log_continuous_ratings.values
        elif self.actions == 'discrete':
            if use_ground_truth:
                return ds.log_gt_discrete_ratings
            else:
                return ds.log_discrete_ratings.values
        else:
            raise ValueError(f'Unknown actions type: {self.actions}')

    @staticmethod
    def _get_terminals(ds):
        terminals = np.zeros(ds.log.shape[0])
        terminals[1:] = ds.log['user_id'].values[1:] != ds.log['user_id'].values[:-1]
        terminals[-1] = True
        return terminals
