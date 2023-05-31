from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import d3rlpy
import numpy as np
import pandas as pd
import structlog
from d3rlpy.algos import AlgoBase
from d3rlpy.models.encoders import EncoderFactory
from numpy.random import Generator

from recsys_mdp.experiments.utils.algorithm_constuctor import (
    init_hidden_state_encoder,
)
from recsys_mdp.experiments.utils.cache import ExperimentCache
from recsys_mdp.experiments.utils.helper import eval_algo, generate_episode
from recsys_mdp.experiments.utils.mdp_constructor import (
    make_mdp, split_dataframe,
    prepare_log_df
)
from recsys_mdp.experiments.utils.phases import (
    GenerationPhaseParameters,
    LearningPhaseParameters
)
from recsys_mdp.experiments.utils.scorers_constructor import init_logger
from recsys_mdp.experiments.utils.type_resolver import TypesResolver
from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RELEVANCE_INT_COL, TERMINATE_COL, relevance_col_for_rating
)
from recsys_mdp.mdp.utils import to_d3rlpy_form_ND, isnone
from recsys_mdp.simulator.env import (
    NextItemEnvironment
)
from recsys_mdp.simulator.framestack import Framestack
from recsys_mdp.utils.base import get_cuda_device
from recsys_mdp.utils.run.config import (
    TConfig, GlobalConfig, extracted
)
from recsys_mdp.utils.run.timer import timer, print_with_timestamp
from recsys_mdp.utils.run.wandb import get_logger

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class NextItemExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int
    rng: Generator
    cuda_device: int | bool

    generation_phase: GenerationPhaseParameters
    learning_phase: LearningPhaseParameters
    cache: ExperimentCache | None

    env: NextItemEnvironment
    framestack: Framestack
    embeddings: Any
    hidden_state_encoder: EncoderFactory | TConfig

    generation_model: AlgoBase
    eval_model: AlgoBase | TConfig

    scoring: TConfig
    mdp: Any

    discrete: bool
    column_for_rating: str

    def __init__(
            self,
            config: TConfig, config_path: Path, seed: int,
            generation_phase: TConfig, learning_phase: TConfig,
            env: TConfig, framestack: TConfig,
            embeddings: TConfig, hidden_state_encoder: TConfig,
            generation_model: TConfig, eval_model: TConfig,
            mdp: TConfig, scoring: TConfig,
            log: bool, cuda_device: bool | int | None,
            project: str = None, wandb_init: TConfig = None,
            cache: TConfig = None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )

        # LOGGING
        self.init_time = timer()
        self.print_with_timestamp('==> Init')
        self.logger = self.config.resolve_object(
            isnone(wandb_init, {}),
            object_type_or_factory=get_logger,
            config=config, log=log, project=project
        )
        logging.disable(logging.DEBUG)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )

        # MATH: RANDOM SEEDING, CUDA DEVICES
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        d3rlpy.seed(seed)
        self.cuda_device = get_cuda_device(cuda_device)

        # PHASES
        self.generation_phase = GenerationPhaseParameters(**generation_phase)
        self.learning_phase = LearningPhaseParameters(**learning_phase)

        self.env = self.config.resolve_object(env)

        self.generation_model = self.config.resolve_object(
            generation_model, n_actions=self.env.n_items, use_gpu=self.cuda_device
        )
        self.discrete = (
            self.generation_model.get_action_type() == d3rlpy.constants.ActionSpace.DISCRETE
        )
        self.column_for_rating = relevance_col_for_rating(self.discrete)
        self.framestack = self.config.resolve_object(
            framestack | dict(discrete=self.discrete, empty_item=self.env.n_items),
            object_type_or_factory=Framestack,
        )
        self.embeddings = self.config.resolve_object(embeddings, size=self.framestack.size)
        self.hidden_state_encoder = hidden_state_encoder

        self.eval_model = eval_model
        self.learnable_model = False

        self.scoring = scoring
        self.mdp = mdp
        self.mdp_preparator = None

        # CACHING
        generation_minimal_config = self.generation_minimal_config(**self.config.config)
        self.cache = self.config.resolve_object(
            cache, object_type_or_factory=ExperimentCache,
            enable=self.generation_phase.use_cache, experiment_config=generation_minimal_config
        )
        if self.cache.enabled:
            self.print_with_timestamp(f'Initialized cache in {self.cache.root}')

    def run(self):
        self.print_with_timestamp('==> Run')
        self.set_metrics()
        total_epoch = 1
        generation_epoch = 1

        self.print_with_timestamp(f'Meta-Epoch: {generation_epoch} ==> generating')
        dataset = self.generate_dataset(generation_epoch)
        dataset_info = None
        # dataset_info = _get_df_info(dataset)

        self.print_with_timestamp(f'Meta-Epoch: {generation_epoch} ==> learning')
        fitter = self._init_rl_setting(dataset)
        total_epoch += self._learn_on_dataset(
            total_epoch, fitter, dataset_info
        )

        self.print_with_timestamp('<==')
        if self.logger:
            self.logger.config.update(self.config.config, allow_val_change=True)

    def generate_dataset(self, generation_epoch: int):
        log_df = self.cache.try_restore_log_df(generation_epoch, logger=self.print_with_timestamp)

        if log_df is None:
            self.print_with_timestamp("Generating dataset...")
            config = self.generation_phase
            samples = []
            for episode in count():
                trajectory = generate_episode(
                    env=self.env, model=self.generation_model, framestack=self.framestack,
                    rng=self.rng, logger=self.logger,
                    first_run=True, use_env_actions=True
                )
                samples.extend(trajectory)
                if episode >= config.episodes_per_epoch or len(samples) >= config.samples_per_epoch:
                    break

            log_df = pd.DataFrame(samples, columns=[
                TIMESTAMP_COL,
                USER_ID_COL, ITEM_ID_COL,
                RELEVANCE_CONT_COL, RELEVANCE_INT_COL,
                TERMINATE_COL, "best_from_env"
            ])

        log_df = prepare_log_df(log_df, column_for_rating=self.column_for_rating)
        self.cache.try_cache_log_df(
            log_df=log_df, generation_epoch=generation_epoch, logger=self.print_with_timestamp
        )
        return log_df

    def _init_rl_setting(self, log_df: pd.DataFrame):
        if not isinstance(self.eval_model, AlgoBase):
            # init RL model
            self.embeddings.fit(log_df)

            from recsys_mdp.models.models import ActorEncoderFactory
            self.hidden_state_encoder = ActorEncoderFactory(
                user_num=self.env.n_users, item_num=self.env.n_items + 1,
                initial_user_embeddings=self.embeddings.users,
                initial_item_embeddings=self.embeddings.items,
                embedding_dim=self.embeddings.size,
                **self.hidden_state_encoder
            )
            self.eval_model = self.config.resolve_object(
                self.eval_model | dict(
                    encoder_factory=self.hidden_state_encoder,
                    actor_encoder_factory=self.hidden_state_encoder,
                    critic_encoder_factory=self.hidden_state_encoder,
                ),
                n_actions=self.env.n_items, use_gpu=self.cuda_device,
            )

        train_log, test_log = split_dataframe(log_df, time_sorted=True)

        mdp_prep, train_mdp, algo_logger = self.data2mdp(train_log, self.mdp, self.scoring)
        test_mdp_config = self.mdp.copy()
        test_mdp_config['reward_function_name'] = "relevance_based_reward"
        test_mdp_config['episode_splitter_name'] = "interaction_interruption"
        _, _, algo_test_logger = self.data2mdp(test_log, test_mdp_config, self.scoring)

        self.mdp_preparator = mdp_prep
        self.algo_logger = algo_logger
        self.algo_test_logger = algo_test_logger

        # Run experiment
        config = self.learning_phase
        fitter = self.eval_model.fitter(
            dataset=train_mdp, n_epochs=config.epochs,
            verbose=False, save_metrics=False, show_progress=False,
        )
        return fitter

    def _framestack_from_last_best(self, user_id, N=10):
        top_framestack = []
        framestack_size = self.zoya_settings['mdp_settings']['framestack_size']
        for i in range(framestack_size):
            items_top = self.env.state.ranked_items(with_satiation=True, discrete=True)
            item_id = self.rng.choice(items_top[:N])
            top_framestack.append(item_id)
            _, _ = self.env.step(item_id)
        # add scores as all is best
        obs = top_framestack + [5] * framestack_size + [user_id]
        return obs

    def _framestack_random_act(self, user_id):
        framestack_size = self.zoya_settings['mdp_settings']['framestack_size']
        obs = self.rng.integers(0, self.env.n_items, framestack_size).tolist() + [user_id]
        return obs

    def _learn_on_dataset(self, total_epoch, fitter, dataset_info = None):
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % self.learning_phase.eval_schedule == 0:
                self.print_with_timestamp(f'Epoch: {epoch} | Total epoch: {total_epoch}')
                eval_algo(
                    self.eval_model, self.algo_test_logger,
                    train_logger=self.algo_logger,
                    env=self.env, framestack=self.framestack,
                    looking_for=[0, 1, 6], dataset_info=dataset_info,
                    rng=self.rng
                )
            total_epoch += 1
        return total_epoch

    def data2mdp(self, log_df, mdp_settings, scorer):
        # TODO: one preparator shuld transform different datasets?
        preparator = make_mdp(data=log_df, framestack_size=self.framestack.size, **mdp_settings)
        states, rewards, actions, terminations, state_tail = preparator.create_mdp()
        mdp = to_d3rlpy_form_ND(states, rewards, actions, terminations, discrete=self.discrete)
        algo_logger = init_logger(
            test_mdp=mdp, state_tail=state_tail, data=log_df,
            wandb_logger=self.logger, discrete=self.discrete, **scorer
        )
        return preparator, mdp, algo_logger

    def print_with_timestamp(self, *args):
        print_with_timestamp(self.init_time, *args)

    def set_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('mae', step_metric='epoch')

    def generation_minimal_config(self, seed, env, generation_phase, framestack, **_):
        env_config, _ = self.config.resolve_object_requirements(
            env, object_type_or_factory=NextItemEnvironment
        )
        # remove global config object
        env_config, _ = extracted(env_config, 'global_config')

        minimal_config = generation_phase | env_config | framestack
        minimal_config['seed'] = seed
        return minimal_config


def _get_df_info(log_df):
    mean_dreturn = np.mean(log_df[RELEVANCE_INT_COL])
    median_dreturn = np.median(log_df[RELEVANCE_INT_COL])
    std_dreturn = np.std(log_df[RELEVANCE_INT_COL])

    mean_return = np.mean(log_df[RELEVANCE_CONT_COL])
    median_return = np.median(log_df[RELEVANCE_CONT_COL])
    std_return = np.std(log_df[RELEVANCE_CONT_COL])

    dataset_info = [
        {
            "discrete_return": mean_dreturn,
            "continuous_return": mean_return,
        },
        {
            "discrete_return": mean_dreturn + std_dreturn,
            "continuous_return": mean_return + std_return,
        },
        {
            "discrete_return": mean_dreturn - std_dreturn,
            "continuous_return": mean_return - std_return,
        },
        {
            "discrete_return": median_dreturn,
            "continuous_return": median_return
        }
    ]
    return dataset_info
