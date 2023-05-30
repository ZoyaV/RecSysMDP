from __future__ import annotations

import numpy as np
from numpy.random import Generator

from recsys_mdp.metrics.logger import log_satiation
from recsys_mdp.simulator.user_state import USER_RESET_MODE_DISCONTINUE, USER_RESET_MODE_INIT


def generate_episode(
        env, model, framestack, rng, logger, cold_start=False, user_id=None,
        use_env_actions=False, log_sat=False, first_run=False
):
    orig_user_id = user_id
    trajectory = []
    N_BEST_ITEMS = 10
    RANGE_SIZE= 15

    user_id = env.reset(user_id=user_id)
    # FIXME: obs keys
    obs_keys = ['items', 'user']
    obs = framestack.compile_observation(framestack.reset(user_id), keys=obs_keys)
    item_id = 0
    # episode generation
    while True:
        items_top = env.state.ranked_items(with_satiation=True, discrete=True)
        if use_env_actions:
            item_id = rng.choice(items_top[:RANGE_SIZE])
        else:
            item_id = model.predict(obs.reshape(1, -1))[0]

        (continuous_relevance, discrete_relevance), terminated = env.step(item_id)
        timestamp = env.timestamp
        obs = framestack.compile_observation(
            framestack.step(item_id, continuous_relevance, discrete_relevance), keys=obs_keys
        )

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
            log_satiation(logger, env.state.satiation, orig_user_id)
    return trajectory


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


def eval_returns(env, model, framestack, user_id=None, logger=None, rng: Generator = None):
    cont_returns, disc_returns, steps_hit_rate, coverages = [], [], [], []
    true_discrete_return = []
    episode_lenghts= []
    n_episodes = 20 if user_id is not None else 50
    for ep in range(20):
        trajectory = generate_episode(
            env, model, framestack=framestack, user_id=user_id, log_sat=True, logger=logger, rng=rng
        )
        episode_lenghts.append(len(trajectory))
        coverage = len({step[2] for step in trajectory})
        step_hit_rate = [step[2] in step[-1] for step in trajectory]
        cont_returns.append(np.mean([step[3] for step in trajectory]))
        disc_returns.append(np.mean([step[4] for step in trajectory]))
        true_discrete_return.append(np.sum([step[4] for step in trajectory]))
        coverages.append(coverage)
        steps_hit_rate.append(np.mean(step_hit_rate))

        # from temp_utils import log_distributions
        # log_distributions(true_items, predicted_items, "True best items", "Predicted best items")
    return {
        'continuous_return': np.mean(cont_returns),
        'discrete_return': np.mean(disc_returns),
        'true_discrete_return': np.mean(true_discrete_return),
        'coverage': np.mean(coverages),
        'step_hit_rate': np.mean(steps_hit_rate),
        'trajectory_len': np.mean(episode_lenghts)
    }


def eval_algo(
        algo, logger, train_logger, env=None, framestack=None,
        looking_for=None, dataset_info=None, rng=None
):
    if env:
        env.hard_reset(mode=USER_RESET_MODE_INIT)

        online_res = dict()
        looking_for.append(None)
        for i in looking_for:
            online_res[f"user {i}"] = eval_returns(
                env, algo, framestack=framestack, user_id=i, logger=logger.wandb_logger, rng=rng
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
