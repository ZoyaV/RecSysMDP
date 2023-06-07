from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator

from recsys_mdp.experiments.utils.phases import LearningPhaseParameters
from recsys_mdp.mdp.base import (
    OBSERVATION_COL, RELEVANCE_CONT_COL, RELEVANCE_INT_COL, RATING_COL,
    ITEM_ID_COL
)
from recsys_mdp.metrics.logger import log_satiation
from recsys_mdp.simulator.user_state import USER_RESET_MODE_DISCONTINUE, USER_RESET_MODE_INIT


def generate_episode(env, model, framestack, user_id=None, log_sat=False,):
    trajectory = []
    user_id, info = env.reset(user_id=user_id)
    obs = framestack.reset(**info)
    while True:
        item_id = model.predict(np.expand_dims(obs, axis=0))[0]
        rating, terminate, truncated, info = env.step(item_id)
        trajectory.append(info | {OBSERVATION_COL: obs})
        if terminate or truncated:
            break

        obs = framestack.step(**info)
    return trajectory


def eval_returns(
        env, model, framestack, eval_phase: LearningPhaseParameters,
        user_id=None, logger=None, rng: Generator = None
):
    n_episodes = eval_phase.eval_episodes if user_id is not None else eval_phase.eval_episodes_all
    interactions = []
    for ep in range(n_episodes):
        trajectory = generate_episode(
            env, model, framestack=framestack, user_id=user_id, log_sat=True
        )
        interactions.extend(trajectory)

    log_df = pd.DataFrame(interactions)
    n_samples, n_items = log_df.shape[0], env.n_items

    avg_cont_relevance = log_df[RELEVANCE_CONT_COL].mean()
    avg_int_relevance = log_df[RELEVANCE_INT_COL].mean()
    avg_return = log_df[RATING_COL].sum()
    avg_reward = avg_return / n_samples
    avg_episode_len = n_samples / n_episodes
    avg_coverage = log_df[ITEM_ID_COL].unique().shape[0]

    return {
        'cont_relevance': avg_cont_relevance,
        'int_relevance': avg_int_relevance,
        'return': avg_return,
        'reward': avg_reward,
        'episode_length': avg_episode_len,
        'coverage': avg_coverage,
        # FIXME: rework this metric calculation
        #  NB: hit_rate == recommended item is in actual top 10 items
        # 'step_hit_rate': np.mean(steps_hit_rate),
    }


def eval_algo(
        algo, logger, train_logger, eval_phase: LearningPhaseParameters,
        env=None, framestack=None, dataset_info=None, rng=None,
):
    if env:
        env.hard_reset(mode=USER_RESET_MODE_INIT)

        online_res = dict()
        # noinspection PyTypeChecker
        looking_for = eval_phase.eval_users + [None]
        for i in looking_for:
            online_res[f"user {i}"] = eval_returns(
                env, algo, framestack=framestack, eval_phase=eval_phase,
                user_id=i, logger=logger.wandb_logger, rng=rng
            )
        if dataset_info is not None:
            for i, name in enumerate(['mean', 'mean+', 'mean-', 'median']):
                online_res[f" dataset {name}"] = dataset_info[i]

    else:
        online_res = None

    # print(online_res)
    logger.visual_log(algo, {
        "test_STAT": logger.static_log(algo),
        "test_INTERECT": logger.interactive_log(algo),
        "ONLINE": online_res
    })

    train_logger.visual_log(algo, {
        "train_STAT": train_logger.static_log(algo),
        "train_INTERECT": train_logger.interactive_log(algo),
    })


def generate_episode_old(
        env, model, framestack_size=10, user_id=None, log_sat=False, logger=None,
        get_best_for_start=True, use_best=False,
        rng: Generator = None
):
    orig_user_id = user_id
    user_id = env.reset(user_id=user_id)
    trajectory = []

    # Get random items from best for framestack
    # TODO: How it will affect to episode lenghts?
    # TODO: Make framestack making as function

    if get_best_for_start:
        top_framestack = []

        for i in range(framestack_size):
            items_top = env.state.ranked_items(with_satiation=True, discrete=True)
            item_id = rng.choice(items_top[:10])
            top_framestack.append(item_id)
            _, _ = env.step(item_id)
        # add scores as all is best
        # [N last item_ids] + [[5] * N] +  [user_id]
        fake_obs = top_framestack + [5] * framestack_size + [user_id]

    else:
        # [N last item_ids] + [user_id]
        fake_obs = rng.integers(0, env.n_items, framestack_size).tolist() + [user_id]
    obs = np.asarray(fake_obs)
    item_id = 0
    # episode generation
    while True:
        if not use_best:
            try:
                item_id = model.predict(obs.reshape(1, -1))[0]
            except:
                item_id = model.predict(obs[:framestack_size].reshape(1, -1))[0]

        timestamp = env.timestamp
        # log satiation histogram
        if env.timestep % 4 == 0 and log_sat:
            log_satiation(logger, env.state.satiation, orig_user_id)

        relevance, terminated = env.step(item_id)

        continuous_relevance, discrete_relevance = relevance

        # TODO: add addtional function for framestack support

        obs[:framestack_size - 1] = obs[1:framestack_size]
        obs[framestack_size - 1] = item_id
        #  print(obs[:framestack_size])
        obs[framestack_size:framestack_size * 2 - 1] = obs[framestack_size + 1:framestack_size * 2]
        obs[framestack_size * 2 - 1] = discrete_relevance

        # print(obs)
        #   print(obs[framestack_size:framestack_size * 2])
        items_top = env.state.ranked_items(with_satiation=True, discrete=True)

        if use_best:
            item_id = rng.choice(items_top[:10])

        trajectory.append(
            (
                timestamp,
                user_id, item_id,
                continuous_relevance, discrete_relevance,
                terminated,
                items_top[:10]
            )
        )
        if terminated:
            break
    env.reset(user_id, USER_RESET_MODE_DISCONTINUE)
    if log_sat:
        log_satiation(logger, env.state.satiation, orig_user_id)

    return trajectory

def _generate_episode(
        self, cold_start=False, user_id=None,
        use_env_actions=False, log_sat=False, first_run=False
):
    env, model = self.env, self.model
    orig_user_id = user_id
    user_id = env.reset(user_id=user_id)
    trajectory = []
    N_BEST_ITEMS = 10
    RANGE_SIZE= 15
    # Get random items from best for framestack
    # TODO: How it will affect to episode lenghts?
    # TODO: Make framestack making as function
    if not cold_start:
        fake_obs = self._framestack_from_last_best(user_id, N_BEST_ITEMS)
    else:
        fake_obs = self._framestack_random_act(user_id)

    obs = np.asarray(fake_obs)
    item_id = 0
    # episode generation
    while True:
        if not use_env_actions:
            item_id = model.predict(obs.reshape(1, -1))[0]
        relevance, terminated = env.step(item_id)
        continuous_relevance, discrete_relevance = relevance
        timestamp = env.timestamp
        if not first_run:
            # generate new observation with framestack
            _, obs = self.preparator.make_interaction(
                relevance=discrete_relevance, user=user_id, item=item_id,
                ts=timestamp, obs_prev=obs, relevance2reward=False
            )

        items_top = env.state.ranked_items(with_satiation=True, discrete=True)
        if use_env_actions:
            item_id = self.rng.choice(items_top[:RANGE_SIZE])
        trajectory.append((
            timestamp,
            user_id, item_id,
            continuous_relevance, discrete_relevance,
            terminated,
            items_top[:N_BEST_ITEMS]
        ))
        if terminated:
            break

        if env.timestep % 4 == 0 and log_sat:
            log_satiation(self.logger, env.state.satiation, orig_user_id)
    env.reset(user_id, USER_RESET_MODE_DISCONTINUE)
    return trajectory
