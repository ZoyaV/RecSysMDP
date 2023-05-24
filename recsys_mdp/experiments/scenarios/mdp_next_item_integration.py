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
from recsys_mdp.experiments.run.wandb import get_logger
from recsys_mdp.generators.utils.base import sample_rng, sample_int, lin_sum, update_exp_trace
from recsys_mdp.experiments.run.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from recsys_mdp.experiments.run.timer import timer, print_with_timestamp
from recsys_mdp.mdp_former.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RELEVANCE_INT_COL, TERMINATE_COL, RATING_COL
)

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


USER_RESET_MODE_CONTINUE = 'continue'
USER_RESET_MODE_INIT = 'init'
USER_RESET_MODE_DISCONTINUE = 'discontinue'


def normalize(x: np.ndarray) -> np.ndarray:
    normalizer = x.sum(-1)
    assert normalizer > 1e-8, f'Normalization is dangerous for {x}'
    return x / normalizer


def softmax(x: np.ndarray, temp=.12) -> np.ndarray:
    """Computes softmax values for a vector `x` with a given temperature."""
    temp = np.clip(temp, 1e-5, 1e+3)
    e_x = np.exp((x - np.max(x, axis=-1)) / temp)
    return e_x / e_x.sum(axis=-1)


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

    n_users: int
    users: np.ndarray
    # mapping user ind -> cluster ind
    user_cluster_ind: np.ndarray
    # user clusters' embeddings
    user_clusters: np.ndarray
    n_user_clusters: int

    n_items: int
    items: np.ndarray
    # mapping item ind -> cluster ind
    item_cluster_ind: np.ndarray
    # item clusters' embeddings
    item_clusters: np.ndarray
    n_item_clusters: int

    def __init__(
            self, global_config: GlobalConfig, seed: int,
            n_users: int, n_items: int,
            n_dims: int, users: TConfig, items: TConfig
    ):
        self.n_dims = n_dims

        rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.user_embeddings_generator = global_config.resolve_object(
            users, n_dims=self.n_dims, seed=rng.integers(100_000_000)
        )
        self.user_cluster_ind, self.users = self.user_embeddings_generator.generate(n_users)
        self.user_clusters = self.user_embeddings_generator.clusters
        self.n_user_clusters = self.user_embeddings_generator.n_clusters

        self.n_items = n_items
        self.item_embeddings_generator = global_config.resolve_object(
            items, n_dims=self.n_dims, seed=rng.integers(100_000_000)
        )
        self.item_cluster_ind, self.items = self.item_embeddings_generator.generate(n_items)
        self.item_clusters = self.item_embeddings_generator.clusters
        self.n_item_clusters = self.item_embeddings_generator.n_clusters


class UserState:
    rng: Generator
    user_id: int

    base_satiation: float
    # all projected onto clusters
    # volatile, it also correlates to user's mood
    satiation: np.ndarray
    # changes with reset
    satiation_speed: np.ndarray
    satiation_drift: float
    init_mood_seed: int

    item_listening_trace_decay: float
    item_listening_trace: np.ndarray
    item_repeat_penalty_power: float

    satisfaction: float
    early_stop_delta: float
    initial_satisfaction: float = 4

    relevance_boosting_k: tuple[float, float]
    metric: str
    embeddings: Embeddings

    def __init__(
            self, user_id: int, embeddings: Embeddings,
            base_satiation: float,
            base_satiation_speed: float | tuple[float, float],
            satiation_drift: float,
            item_listening_trace_decay: float,
            item_repeat_penalty_power: float,
            early_stop_delta: float,
            similarity_metric: str,
            item_to_cluster_classification: str,
            relevance_boosting: tuple[float, float],
            boosting_softness: tuple[float, float],
            discrete_actions: list[tuple[float, float]],
            rng: Generator
    ):
        self.user_id = user_id
        self.rng = sample_rng(rng)

        self.base_satiation = base_satiation
        self.satiation_drift = satiation_drift
        self.item_listening_trace_decay = item_listening_trace_decay
        self.item_repeat_penalty_power = item_repeat_penalty_power
        self.early_stop_delta = early_stop_delta
        self.relevance_boosting_k = tuple(relevance_boosting)
        self.boosting_softness = tuple(boosting_softness)
        self.discrete_actions_distr = discrete_actions
        self.metric = similarity_metric
        self.embeddings = embeddings

        # how to normalize item-to-item_clusters similarity to attention/probability
        self.item_to_cluster_classification = dict(
            softmax=softmax,
            normalize=normalize
        )[item_to_cluster_classification]

        self.init_mood_seed = sample_int(self.rng)
        self.satiation = self.sample_satiation(self.init_mood_seed)
        self.satiation_speed = self.sample_satiation_speed(base_satiation_speed)
        self.item_listening_trace = np.zeros(self.embeddings.n_items)
        self.satisfaction = self.sample_initial_satisfaction()

    @property
    def tastes(self) -> np.ndarray:
        return self.embeddings.users[self.user_id]

    def reset(self, mode: str = USER_RESET_MODE_CONTINUE):
        self.satisfaction = self.sample_initial_satisfaction()
        self.item_listening_trace *= self.item_listening_trace_decay**7

        if mode == USER_RESET_MODE_CONTINUE:
            self.drift_satiation()
        elif mode == USER_RESET_MODE_INIT:
            # return to the initial mood
            self.satiation = self.sample_satiation(self.init_mood_seed)
            self.item_listening_trace[:] = 0
        elif mode == USER_RESET_MODE_DISCONTINUE:
            # re-sample new mood
            self.satiation = self.sample_satiation(sample_int(self.rng))
        else:
            raise ValueError(f'User reset mode "{mode}" does not supported.')

    def step(self, item_id: int):
        # 1) update item listening trace
        update_exp_trace(
            traces=self.item_listening_trace, tr=item_id, decay=self.item_listening_trace_decay
        )

        # 2) increase satiation via similarity and speed
        item_to_cluster_relevance = self.item_to_cluster_relevance(item_id)
        self.satiation *= 1.0 + item_to_cluster_relevance * self.satiation_speed
        np.clip(self.satiation, 1e-4, 1e+4, out=self.satiation)

        # 3) calculate relevance
        continuous_relevance, discrete_relevance = self.relevance(
            item_id, item_to_cluster_relevance=item_to_cluster_relevance
        )

        # 4) update current satisfaction
        self.satisfaction = lin_sum(self.satisfaction, .9, discrete_relevance)
        return continuous_relevance, discrete_relevance

    def sample_stop_listening(self):
        min_prob = 0.001
        k = self.early_stop_delta
        dissatisfaction = 5 - self.satisfaction

        # early stop increases with increasing speed based on dissatisfaction
        probability = min_prob + k * dissatisfaction * (dissatisfaction + 1) / 2
        return self.rng.random() < probability

    def sample_satiation(self, seed: int) -> np.ndarray:
        n_clusters = self.embeddings.n_item_clusters
        rng = np.random.default_rng(seed)
        return self.base_satiation + rng.uniform(size=n_clusters)

    def drift_satiation(self):
        # continue based on previous mood, but slightly drift to another mood
        mood = self.satiation
        new_mood = self.sample_satiation(sample_int(self.rng))
        alpha = self.satiation_drift
        # calculate new satiation vector
        new_satiation = lin_sum(x=mood, lr=self.satiation_drift, y=new_mood)

        # but update only those parts that are object to satiation
        # to prevent specific scenarios stop working expected way
        update_mask = self.satiation_speed > 0
        self.satiation[update_mask] = new_satiation[update_mask]

    def relevance(
            self, item_id: int = None, with_satiation: bool = True,
            item_to_cluster_relevance: np.ndarray = None
    ):
        if item_id is None:
            # compute relevance for all items without consuming
            all_relevance = [
                self.relevance(item_id, with_satiation=with_satiation)
                for item_id in range(self.embeddings.n_items)
            ]
            continuous_relevance = np.array([cont for cont, discr in all_relevance])
            discrete_relevance = np.array([discr for cont, discr in all_relevance])
            return continuous_relevance, discrete_relevance

        # get item similarity to user preferences
        relevance = similarity(self.tastes, self.embeddings.items[item_id], metric=self.metric)

        if with_satiation:
            if item_to_cluster_relevance is None:
                item_to_cluster_relevance = self.item_to_cluster_relevance(item_id)
            # compute and apply relevance boosting based on item-to-clusters satiation
            relevance *= self.relevance_boosting(item_to_cluster_relevance)
            # compute and apply boosting penalty based on repeating tracks
            relevance *= self.repeat_boosting(item_id)

        # discrete relevance as user feedback
        discrete_relevance = self.sample_discrete_response(relevance)
        return relevance, discrete_relevance

    def ranked_items(self, discrete: bool, with_satiation: bool) -> np.ndarray:
        continuous_relevance, discrete_relevance = self.relevance(with_satiation=with_satiation)
        relevance = discrete_relevance if discrete else continuous_relevance

        ranked_items = np.argsort(relevance)[::-1]
        return ranked_items

    def sample_discrete_response(self, relevance) -> int:
        marks = np.array([self.rng.normal(*distr) for distr in self.discrete_actions_distr])
        return 2 + np.argmin(np.abs(marks - relevance))

    def relevance_boosting(self, item_to_cluster_relevance: np.ndarray = None) -> float:
        # compute boosting from the aggregate weighted cluster satiation
        aggregate_item_satiation = np.sum(self.satiation * item_to_cluster_relevance)

        boosting_k = self.relevance_boosting_k[aggregate_item_satiation > 1.0]
        boosting_softness = self.boosting_softness[aggregate_item_satiation > 1.0]
        return boosting(aggregate_item_satiation, k=boosting_k, softness=boosting_softness)

    def repeat_boosting(self, item_id: int) -> float:
        repeat = self.item_listening_trace[item_id]
        if repeat < 2:
            return 1.
        return (repeat - 1) ** (-self.item_repeat_penalty_power)

    def sample_satiation_speed(self, base_satiation_speed) -> np.ndarray:
        n_clusters = self.embeddings.n_item_clusters
        if isinstance(base_satiation_speed, float):
            return np.full(n_clusters, base_satiation_speed)

        # tuple[float, float]
        base_satiation_speed, k = base_satiation_speed
        k = 1.0 + k
        min_speed, max_speed = 1/k * base_satiation_speed, k * base_satiation_speed
        return np.clip(
            self.rng.uniform(min_speed, max_speed, n_clusters),
            0., 1.0
        )

    def item_to_cluster_relevance(self, item_id: int) -> np.ndarray:
        item_to_cluster_relevance = similarity(
            self.embeddings.items[item_id], self.embeddings.item_clusters, metric=self.metric
        )
        # normalize relevance to [0, 1]
        return self.item_to_cluster_classification(item_to_cluster_relevance)

    def sample_initial_satisfaction(self):
        return self.rng.normal(self.initial_satisfaction, 0.15)


class NextItemEnvironment:
    n_users: int
    n_items: int

    embeddings: Embeddings

    max_episode_len: tuple[int, int]
    global_timestep: int
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
            embeddings | dict(
                global_config=self.global_config, seed=seed,
                n_users=n_users, n_items=n_items
            ),
            object_type_or_factory=Embeddings
        )
        self.states = [
            UserState(user_id, embeddings=self.embeddings, rng=self.rng, **user_state)
            for user_id in range(self.n_users)
        ]

        if isinstance(max_episode_len, int):
            max_episode_len = (max_episode_len, 0)
        avg_len, delta = max_episode_len
        self.max_episode_len = (avg_len - delta, avg_len + delta)

        self.global_timestep = self.timestep = 0
        self.timestamp = random_datetime(self.rng, end_year=2019)

    def hard_reset(self, mode: str = USER_RESET_MODE_INIT):
        self.global_timestep = self.timestep = 0

        assert mode in [USER_RESET_MODE_INIT, USER_RESET_MODE_DISCONTINUE], \
            f'Env hard reset mode "{mode}" does not supported.'

        for user in self.states:
            user.reset(mode)

    def reset(self, user_id: int = None, mode: str = USER_RESET_MODE_CONTINUE):
        if user_id is None:
            user_id = self.rng.integers(self.n_users)

        self.state = self.states[user_id]

        assert mode in [USER_RESET_MODE_CONTINUE, USER_RESET_MODE_DISCONTINUE], \
            f'Env reset mode "{mode}" does not supported.'
        self.state.reset(mode)

        self.timestep = 0
        self.timestamp += pause_random_duration(self.rng)
        self.current_max_episode_len = self.rng.integers(*self.max_episode_len)
        return self.state.user_id

    def step(self, item_id: int):
        relevance = self.state.step(item_id)

        self.timestep += 1
        self.global_timestep += 1
        self.timestamp += track_random_duration(self.rng)

        terminated = self.timestep >= self.current_max_episode_len
        terminated |= self.state.sample_stop_listening()

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
            top_k: int, ratings_column,
            mdp_settings: TConfig, scorer: TConfig, algo_settings: TConfig
    ):
        from constructors.mdp_constructor import make_mdp
        from recsys_mdp.mdp_former.utils import to_d3rlpy_form_ND
        from constructors.algorithm_constuctor import init_model
        from constructors.algorithm_constuctor import init_algo
        from constructors.scorers_constructor import init_logger
        from run_experiment import eval_algo

        log = pd.DataFrame(dataset, columns=[
            TIMESTAMP_COL,
            USER_ID_COL, ITEM_ID_COL,
            RELEVANCE_CONT_COL, RELEVANCE_INT_COL,
            TERMINATE_COL
        ])
        log[RATING_COL] = log[ratings_column]

        mdp_preparator = make_mdp(data=log, **mdp_settings)
        states, rewards, actions, terminations, state_tail = mdp_preparator.create_mdp()
        train_mdp = to_d3rlpy_form_ND(
            states, rewards, actions, terminations,
            discrete=scorer['prediction_type'] == "discrete"
        )

        # Init RL algorithm
        if not self.learnable_model:
            model = init_model(data=log, **algo_settings['model_parameters'])
            algo = init_algo(model, **algo_settings['general_parameters'])
            self.model = algo
            self.learnable_model = True

        # Init scorer
        logger = init_logger(
            train_mdp, state_tail, log, top_k, wandb_logger=self.logger, **scorer
        )

        # Run experiment
        config = self.learning_config
        fitter = self.model.fitter(
            dataset=train_mdp, n_epochs=config.epochs,
            verbose=False, save_metrics=False, show_progress=False,
        )
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % config.eval_schedule == 0:
                eval_algo(
                    self.model, logger, train_logger=logger, env=self.env,
                    looking_for=[0, 1, 6]
                )
            total_epoch += 1
        return total_epoch

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
            from recsys_mdp.generators.utils.random_recommender import RandomRecommender
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
