from __future__ import annotations

import numpy as np
from numpy.random import Generator

from recsys_mdp.experiments.utils.phases import LearningPhaseParameters
from recsys_mdp.metrics.logger import log_satiation
from recsys_mdp.simulator.user_state import USER_RESET_MODE_DISCONTINUE, USER_RESET_MODE_INIT

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import numpy as np
import os
import cv2
import wandb
def generate_item_colors(trajectory):
    item_colors = []  # Пустой словарь для хранения цветов айтемов

    for i, entry in enumerate(trajectory):
        items_top = list(entry[6] ) # Получаем items_top для текущего шага траектории

        # Определяем позицию порекомендованного айтема в топе
        recommended_item = entry[2]  # item_id, который мы порекомендовали
        position = items_top.index(recommended_item)

        # Вычисляем пропорцию позиции в топе
        proportion = position / (len(items_top) - 1)

        # Создаем градиентный переход от голубого к красному
        cmap = LinearSegmentedColormap.from_list("custom_colormap", ["#00BFFF", "#FF0000"], N=len(items_top))

        # Получаем цвет для порекомендованного айтема
        color = cmap(proportion)

        # Преобразуем цвет к трехкомпонентному формату RGB
        color_rgb = np.array(color)[:3]  # Оставляем только первые три компонента RGB
        # print(color_rgb)
        # Обновляем словарь item_colors для каждого item_id в items_top
        item_colors.append({item_id: color_rgb for item_id in items_top})
        # print(item_colors)
    return item_colors


def generate_image(trajectory, item_colors):
    # Определите размеры изображения на основе длины траектории
    width = len(trajectory)
    height = 1

    # Создайте пустой массив numpy для хранения цветов пикселей
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for i, entry in enumerate(trajectory):
        item_id = entry[2]  # Получите item_id из элемента trajectory
        color = item_colors[i][item_id] * 255  # Получите цвет по item_id
        # print(color)
        # Задайте цвет пикселя в массиве изображения
        image[0, i] = color

    return image


def save_trajectory(trajectory, save_path):
    # Генерируем словарь item_colors на основе массива trajectory
    item_colors = generate_item_colors(trajectory)
    # Создаем массив numpy-изображение
    image = generate_image(trajectory, item_colors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matplotlib.use('Agg')
    num_items = len(set(item_id for _, _, item_id, _, _, _, _ in trajectory))

    N = 500
    image = cv2.resize(image, (N, N), interpolation=cv2.INTER_NEAREST)

    if 'run' not in save_path:
        images = wandb.Image(
            image,
            caption="action accuracy"
        )

        wandb.log({"episode-progress": images})

    #  image_resized = Image.fromarray((image * 255).astype(np.uint8)).resize((N, N))
    # print(image)
    # Отображаем изображение
    plt.imshow(image)
    plt.title(f"Number of items: {len(trajectory)}, \n set items {num_items}", fontsize=14)
    # Save the plot
    plt.savefig(save_path)


def generate_episode(
        env, model, framestack, rng, logger, cold_start=False, user_id=None,
        use_env_actions=False, log_sat=False, first_run=False
):
    orig_user_id = user_id
    trajectory = []
    N_BEST_ITEMS = 10
    RANGE_SIZE= np.random.choice([15, 25, -10])

    user_id = env.reset(user_id=user_id)
    # FIXME: obs keys
    obs_keys = ['items', 'user']
    obs = framestack.compile_observation(framestack.reset(user_id), keys=obs_keys)
    item_id = 0
    # episode generation
    while True:
        items_top = env.state.ranked_items(with_satiation=True, discrete=True)
        if use_env_actions:
            if RANGE_SIZE < 0:
                item_id = rng.choice(items_top[RANGE_SIZE:])
            else:
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
            items_top
        ))
        if terminated:
            break



        if env.timestep % 4 == 0 and log_sat:
            log_satiation(logger, env.state.satiation, orig_user_id)

    if env.timestep % 4 == 0:
        if use_env_actions:
            save_trajectory(trajectory, f"trajectory_log/env_best_100/{user_id}_{env.timestamp}.png")
        else:

            save_trajectory(trajectory, f"trajectory_log/run/{user_id}_{env.timestamp}.png")
    return trajectory


def eval_returns(
        env, model, framestack, eval_phase: LearningPhaseParameters,
        user_id=None, logger=None, rng: Generator = None
):
    cont_returns, disc_returns, steps_hit_rate, coverages = [], [], [], []
    true_discrete_return = []
    episode_lenghts= []
    n_episodes = eval_phase.eval_episodes if user_id is not None else eval_phase.eval_episodes_all
    for ep in range(n_episodes):
        trajectory = generate_episode(
            env, model, framestack=framestack, user_id=user_id, log_sat=True,
            logger=logger, rng=rng
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
        algo, logger, train_logger, eval_phase: LearningPhaseParameters,
        env=None, framestack=None, dataset_info=None, rng=None,
):
    if env:
        env.hard_reset(mode=USER_RESET_MODE_INIT)

        online_res = dict()
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
            item_id = rng.choice(items_top[:60])

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
    RANGE_SIZE= 500
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
