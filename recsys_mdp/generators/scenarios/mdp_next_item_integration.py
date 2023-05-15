from __future__ import annotations

import datetime
import logging
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from d3rlpy.base import LearnableBase
from numpy.random import Generator

from recsys_mdp.generators.datasets.mdp.utils import boosting
from recsys_mdp.generators.datasets.synthetic.relevance import similarity
from recsys_mdp.generators.run.wandb import get_logger
from recsys_mdp.generators.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from recsys_mdp.generators.utils.timer import timer, print_with_timestamp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class MdpGenerationProcessParameters:
    epochs: int

    episodes_per_epoch: int | None
    samples_per_epoch: int | None

    def __init__(
            self,
            epochs: int,
            episodes_per_epoch: int | None = None,
            samples_per_epoch: int | None = None,
    ):
        self.epochs = epochs
        assert episodes_per_epoch is not None or samples_per_epoch is not None
        self.episodes_per_epoch = episodes_per_epoch
        self.samples_per_epoch = samples_per_epoch


class LearningProcessParameters:
    epochs: int
    eval_schedule: int
    eval_episodes: int

    def __init__(self, epochs: int, eval_schedule: int = 1, eval_episodes: int = 1):
        self.epochs = epochs
        self.eval_schedule = eval_schedule
        self.eval_episodes = eval_episodes


class Embeddings:
    n_dims: int
    users: np.ndarray
    items: np.ndarray

    def __init__(
            self, global_config: GlobalConfig,
            n_users: int, n_items: int,
            n_dims: int, users: TConfig, items: TConfig
    ):
        self.n_dims = n_dims
        self.users = (
            global_config
            .resolve_object(users, n_dims=self.n_dims)
            .generate(n_users)
        )
        item_embeddings_generator = global_config.resolve_object(items, n_dims=self.n_dims)
        self.items = item_embeddings_generator.generate(n_items)
        self.item_clusters = item_embeddings_generator.clusters
        self.n_item_clusters = item_embeddings_generator.n_clusters


class UserState:
    rng: Generator
    user_id: int
    tastes: np.ndarray

    # all projected onto clusters
    # volatile, it also correlates to user's mood
    satiation: np.ndarray
    # changes with reset
    satiation_speed: np.ndarray

    relevance_boosting_k: tuple[float, float]
    metric: str
    embeddings: Embeddings

    def __init__(
            self, user_id: int, embeddings: Embeddings,
            base_satiation: float,
            base_satiation_speed: float | tuple[float, float],
            similarity_metric: str,
            relevance_boosting: tuple[float, float],
            boosting_softness: tuple[float, float],
            discrete_actions: list[tuple[float, float]],
            rng: Generator
    ):
        self.user_id = user_id
        self.tastes = embeddings.users[user_id]
        self.rng = np.random.default_rng(rng.integers(100_000_000))

        n_clusters = embeddings.n_item_clusters
        self.satiation = base_satiation + self.rng.uniform(size=n_clusters)

        if isinstance(base_satiation_speed, float):
            self.satiation_speed = np.full(embeddings.n_item_clusters, base_satiation_speed)
        else:
            # tuple[float, float]
            base_satiation_speed, k = base_satiation_speed
            k = 1.0 + k
            min_speed, max_speed = 1/k * base_satiation_speed, k * base_satiation_speed
            self.satiation_speed = np.clip(
                self.rng.uniform(min_speed, max_speed, n_clusters),
                0., 1.0
            )

        self.relevance_boosting_k = tuple(relevance_boosting)
        self.boosting_softness = tuple(boosting_softness)
        self.discrete_actions_distr = discrete_actions
        self.metric = similarity_metric
        self.embeddings = embeddings

    def step(self, item_id: int):
        # 1) find similarity between item embedding and item clusters
        item_embedding = self.embeddings.items[item_id]
        clusters = self.embeddings.item_clusters
        item_to_cluster_relevance = similarity(item_embedding, clusters, metric=self.metric)
        item_to_cluster_relevance /= item_to_cluster_relevance.sum(-1)

        # 2) increase satiation via similarity and speed
        self.satiation *= 1.0 + item_to_cluster_relevance * self.satiation_speed

        # 3) get item similarity to user preferences and compute boosting
        #       from the aggregate weighted cluster satiation
        base_item_to_user_relevance = similarity(self.tastes, item_embedding, metric=self.metric)
        aggregate_item_satiation = np.sum(self.satiation * item_to_cluster_relevance)

        boosting_k = self.relevance_boosting_k[aggregate_item_satiation > 1.0]
        boosting_softness = self.boosting_softness[aggregate_item_satiation > 1.0]
        relevance_boosting = boosting(
            aggregate_item_satiation, k=boosting_k, softness=boosting_softness
        )

        # 4) compute continuous and discrete relevance as user feedback
        relevance = base_item_to_user_relevance * relevance_boosting
        discrete_relevance = self.sample_user_response(relevance)

        # print(
        #     f'AggSat {aggregate_item_satiation:.2f} '
        #     f'| RB {relevance_boosting:.2f} '
        #     f'| Rel {relevance:.3f}'
        # )
        return relevance, discrete_relevance

    def sample_user_response(self, relevance):
        marks = np.array([self.rng.normal(*distr) for distr in self.discrete_actions_distr])
        return 2 + np.argmin(np.abs(marks - relevance))


class NextItemEnvironment:
    n_users: int
    n_items: int

    embeddings: Embeddings

    max_episode_len: tuple[int, int]
    timestep: int
    timestamp: datetime.datetime
    state: UserState
    states: list[UserState]
    current_max_episode_len: int

    def __init__(
            self, global_config: GlobalConfig, seed: int,
            n_users: int, n_items: int,
            embeddings: TConfig,
            max_episode_len: int | tuple[int, int],
            user_state: TConfig,
    ):
        self.global_config = global_config
        self.rng = np.random.default_rng(seed)

        self.n_users = n_users
        self.n_items = n_items
        self.embeddings = global_config.resolve_object(
            embeddings | dict(global_config=self.global_config, n_users=n_users, n_items=n_items),
            object_type_or_factory=Embeddings
        )
        self.states = [
            UserState(user_id, embeddings=self.embeddings, rng=self.rng, **user_state)
            for user_id in range(self.n_users)
        ]
        if isinstance(max_episode_len, int):
            self.max_episode_len = (max_episode_len, max_episode_len)
        else:
            # tuple[int, int]: (avg_len, delta)
            self.max_episode_len = (
                max_episode_len[0] - max_episode_len[1],
                max_episode_len[0] + max_episode_len[1]
            )
        self.timestep = 0
        self.timestamp = random_datetime(self.rng, end_year=2019)

    def reset(self, user_id: int = None):
        if user_id is None:
            user_id = self.rng.integers(self.n_users)

        self.state = self.states[user_id]
        self.timestep = 0
        self.timestamp += pause_random_duration(self.rng)
        self.current_max_episode_len = self.rng.integers(*self.max_episode_len)

        # print(f'Sat: {self.state.satiation}')
        # print(f'SSp: {self.state.satiation_speed}')
        return self.state.user_id

    def step(self, item_id: int):
        relevance = self.state.step(item_id)
        self.timestep += 1
        self.timestamp += track_random_duration(self.rng)

        terminated = self.timestep >= self.current_max_episode_len
        # if terminated:
        #     print(f'Sat: {self.state.satiation}')
        #     print(f'SSp: {self.state.satiation_speed}')
        return relevance, terminated


class MdpNextItemExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int
    rng: Generator

    generation_config: MdpGenerationProcessParameters
    learning_config: LearningProcessParameters

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            generation: TConfig, learning: TConfig,
            zoya_settings: TConfig,
            model: TConfig, env: TConfig,
            log: bool, cuda_device: bool | int | None,
            project: str = None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )
        self.logger = get_logger(config, log=log, project=project)

        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.generation_config = MdpGenerationProcessParameters(**generation)
        self.learning_config = LearningProcessParameters(**learning)
        self.zoya_settings = zoya_settings

        self.env: NextItemEnvironment = self.config.resolve_object(
            env, object_type_or_factory=NextItemEnvironment
        )
        self.model: LearnableBase = self.config.resolve_object(
            model | dict(use_gpu=get_cuda_device(cuda_device)),
            n_actions=self.env.n_items
        )
        self.learnable_model = False

    def run(self):
        logging.disable(logging.DEBUG)
        self.set_metrics()

        self.print_with_timestamp('==> Run')
        total_epoch = 0
        for generation_epoch in range(self.generation_config.epochs):
            self.print_with_timestamp(f'Epoch: {generation_epoch} ==> generation')
            dataset = self._generate_dataset()
            self.print_with_timestamp(f'Epoch: {generation_epoch} ==> learning')
            total_epoch += self._learn_on_dataset(
                total_epoch, dataset,
                **self.zoya_settings
            )

        self.print_with_timestamp('<==')

    def _generate_dataset(self):
        config = self.generation_config
        samples = []
        for episode in count():
            samples.extend(self._generate_episode())

            if config.episodes_per_epoch is not None and episode >= config.episodes_per_epoch:
                break
            if config.samples_per_epoch is not None and len(samples) >= config.samples_per_epoch:
                break

        return samples

    def _generate_episode(self):
        env, model = self.env, self.model
        user_id = env.reset()
        trajectory = []
        # FIXME: set special item_id for EMPTY_ITEM token
        # [10 last item_ids] + [user_id]
        fake_obs = np.random.randint(0, 3521, 10).tolist() + [user_id]
        obs = np.asarray(fake_obs)

        while True:
            # FIXME: separate
            # TODO: batched observations vs item_id as obs with env wrapper
            try:
                item_id = model.predict(obs.reshape(1, -1))[0]
            except:
                item_id = model.predict(obs[:10].reshape(1, -1))[0]
            obs[:9] = obs[1:10]
            obs[-2] = item_id

            timestamp = env.timestamp

            relevance, terminated = env.step(item_id)
            continuous_relevance, discrete_relevance = relevance
            trajectory.append((
                timestamp,
                user_id, item_id,
                continuous_relevance, discrete_relevance,
                terminated
            ))
            if terminated:
                break
        return trajectory

    def _learn_on_dataset(
            self, total_epoch: int, dataset: list[tuple],
            top_k: int,
            mdp_settings: TConfig, scorer: TConfig, algo_settings: TConfig
    ):
        from constructors.mdp_constructor import make_mdp
        from recsys_mdp.mdp_former.utils import to_d3rlpy_form_ND
        from constructors.algorithm_constuctor import init_model
        from constructors.algorithm_constuctor import init_algo
        from constructors.scorers_constructor import init_scorers
        from constructors.scorers_constructor import init_logger
        from run_experiment import eval_algo

        log = pd.DataFrame(dataset, columns=[
            'timestamp',
            'user_id', 'item_id',
            'continuous_rating', 'discrete_rating',
            'terminal'
        ])
        data_mapping = dict(
            user_col_name='user_id',
            item_col_name='item_id',
            reward_col_name='discrete_rating',
            timestamp_col_name='timestamp'
        )
        train_values = get_values_fixme(log, data_mapping)
        mdp_preparator = make_mdp(data=log, data_mapping=data_mapping, **mdp_settings)
        states, rewards, actions, terminations, state_tail = mdp_preparator.create_mdp()
        train_mdp = to_d3rlpy_form_ND(
            states, rewards, actions, terminations,
            discrete=scorer['prediction_type'] == "discrete"
        )

        # Init RL algorithm
        if not self.learnable_model:
            model = init_model(train_values, **algo_settings['model_parameters'])
            algo = init_algo(model, **algo_settings['general_parameters'])
            self.model = algo
            self.learnable_model = True

        # Init scorer
        scorers = init_scorers(state_tail, train_values, top_k, **scorer)
        logger = init_logger(
            train_mdp, state_tail, train_values, top_k, wandb_logger=self.logger, **scorer
        )

        # Run experiment
        config = self.learning_config
        fitter = self.model.fitter(
            dataset=train_mdp,
            n_epochs=config.epochs,
            eval_episodes=train_mdp,
            scorers=scorers,
            verbose=False, save_metrics=False, show_progress=False,
        )
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % config.eval_schedule == 0:
                eval_algo(self.model, logger)
                # self._eval_and_log(total_epoch)
            total_epoch += 1
        return total_epoch

    def _eval_and_log(self, epoch):
        metrics = self._eval_returns()

        self.print_with_timestamp(
            f'Epoch {epoch:03}: '
            f'ContRet {metrics["continuous_return"]:.3f} '
            f'| DiscRet {metrics["discrete_return"]:.3f}'
        )
        if self.logger:
            metrics |= dict(epoch=epoch)
            self.logger.log(metrics)

    def _eval_returns(self):
        cont_returns, disc_returns = [], []
        for ep in range(self.learning_config.eval_episodes):
            trajectory = self._generate_episode()
            cont_returns.append(np.sum([step[2] for step in trajectory]))
            disc_returns.append(np.sum([step[3] for step in trajectory]))
        return {
            'continuous_return': np.mean(cont_returns),
            'discrete_return': np.mean(disc_returns),
        }

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)

    def set_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('mae', step_metric='epoch')


class TypesResolver(LazyTypeResolver):
    def resolve(self, type_name: str, **kwargs):
        if type_name == 'dataset.toy_ratings':
            from recsys_mdp.generators.datasets.synthetic.dataset import \
                ToyRatingsDatasetBuilder
            return ToyRatingsDatasetBuilder
        if type_name == 'ds_source.random':
            from recsys_mdp.generators.datasets.synthetic.log import RandomLogGenerator
            return RandomLogGenerator
        if type_name == 'embeddings.random':
            from recsys_mdp.generators.datasets.synthetic.embeddings import \
                RandomEmbeddingsGenerator
            return RandomEmbeddingsGenerator
        if type_name == 'embeddings.clusters':
            from recsys_mdp.generators.datasets.synthetic.embeddings import \
                RandomClustersEmbeddingsGenerator
            return RandomClustersEmbeddingsGenerator
        if type_name == 'model.random':
            from recsys_mdp.generators.random_recommender import RandomRecommender
            return RandomRecommender
        if type_name == 'd3rlpy.cql':
            from d3rlpy.algos import CQL
            return CQL
        if type_name == 'd3rlpy.sac':
            from d3rlpy.algos import SAC
            return SAC
        if type_name == 'd3rlpy.ddpg':
            from d3rlpy.algos import DDPG
            return DDPG
        if type_name == 'd3rlpy.discrete_cql':
            from d3rlpy.algos import DiscreteCQL
            return DiscreteCQL
        # if type_name == 'd3rlpy.sdac':
        #     from replay.models.rl.sdac.sdac import SDAC
        #     return SDAC
        if type_name == 'd3rlpy.discrete_sac':
            from d3rlpy.algos import DiscreteSAC
            return DiscreteSAC
        if type_name == 'd3rlpy.bc':
            from d3rlpy.algos.bc import BC
            return BC
        raise ValueError(f'Unknown type: {type_name}')


def get_values_fixme(data, col_mapping):
    full_users = data[col_mapping['user_col_name']].values
    full_items = data[col_mapping['item_col_name']].values

    users_unique = np.unique(data[col_mapping['user_col_name']].values)
    items_unique = np.unique(data[col_mapping['item_col_name']].values)

    rating = data[col_mapping['reward_col_name']].values
    return {
        'users_unique': users_unique,
        'items_unique': items_unique,
        'full_users': full_users,
        'full_items': full_items,
        'rating': rating
    }


def random_datetime(
        rng: Generator, start_year: int = 2019, end_year: int = 2021
) -> datetime.datetime:
    return datetime.datetime(
        year=rng.integers(start_year, end_year, endpoint=True),
        month=rng.integers(1, 12, endpoint=True),
        day=rng.integers(1, 28, endpoint=True),
        hour=rng.integers(1, 24, endpoint=True),
        minute=rng.integers(1, 60, endpoint=True),
        second=rng.integers(1, 60, endpoint=True)
    )


def pause_random_duration(rng: Generator) -> datetime.timedelta:
    return datetime.timedelta(minutes=float(rng.integers(15, 600, endpoint=True)))


def track_random_duration(rng: Generator) -> datetime.timedelta:
    return datetime.timedelta(seconds=float(rng.integers(120, 260, endpoint=True)))


def get_cuda_device(cuda_device: int | None) -> int | bool:
    if cuda_device is not None:
        import torch.cuda
        cuda_available = torch.cuda.is_available()
        print(f'CUDA available: {cuda_available}; device: {cuda_device}')
        if not cuda_available:
            cuda_device = False
    return cuda_device
