from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import d3rlpy
import numpy as np
import pandas as pd
import structlog
from d3rlpy.base import LearnableBase
from numpy.random import Generator

from recsys_mdp.experiments.utils.algorithm_constuctor import (
    init_hidden_state_encoder, init_algo,
    init_als_embeddings
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
    RELEVANCE_INT_COL, TERMINATE_COL
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

    generation_phase: GenerationPhaseParameters
    learning_phase: LearningPhaseParameters
    cache: ExperimentCache | None

    env: NextItemEnvironment
    generation_model: LearnableBase
    eval_model: LearnableBase

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            generation_phase: TConfig, learning_phase: TConfig,
            env: TConfig, framestack: TConfig,
            generation_model: TConfig, eval_model: TConfig,
            zoya_settings: TConfig,
            log: bool, cuda_device: bool | int | None,
            project: str = None, wandb_init: TConfig = None,
            cache: TConfig = None,
            **_
    ):
        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )

        self.logger = self.config.resolve_object(
            isnone(wandb_init, {}),
            object_type_or_factory=get_logger,
            config=config, log=log, project=project
        )
        logging.disable(logging.DEBUG)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        d3rlpy.seed(seed)

        self.generation_phase = GenerationPhaseParameters(**generation_phase)
        self.learning_phase = LearningPhaseParameters(**learning_phase)
        self.zoya_settings = zoya_settings

        self.env = self.config.resolve_object(env)
        # TODO: call model.create_impl to init dimensions
        self.generation_model = self.config.resolve_object(
            generation_model, n_actions=self.env.n_items, use_gpu=get_cuda_device(cuda_device)
        )
        self.eval_model = self.config.resolve_object(
            eval_model, n_actions=self.env.n_items, use_gpu=get_cuda_device(cuda_device)
        )
        assert self.generation_model.get_action_type() == self.eval_model.get_action_type()
        self.discrete = (
                self.generation_model.get_action_type() == d3rlpy.constants.ActionSpace.DISCRETE
        )

        self.framestack = self.config.resolve_object(
            framestack | dict(discrete=self.discrete, empty_item=self.env.n_items),
            object_type_or_factory=Framestack,
        )
        self.model = self.generation_model

        self.learnable_model = False
        self.preparator = None

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
        total_epoch = 0
        generation_epoch = 0

        self.print_with_timestamp(f'Meta-Epoch: {generation_epoch} ==> generating')
        dataset = self._generate_dataset(generation_epoch, **self.zoya_settings)

        self.print_with_timestamp(f'Meta-Epoch: {generation_epoch} ==> learning')
        fitter, dataset_info = self._init_rl_setting(
             dataset, **self.zoya_settings
        )
        # TODO: need to be a parameter
        dataset_info = None
        for generation_epoch in range(self.generation_phase.epochs):
            self.print_with_timestamp(f'Epoch: {generation_epoch} ==> learning')
            total_epoch += self._learn_on_dataset(
                total_epoch, fitter, dataset_info
            )

        self.print_with_timestamp('<==')
        if self.logger:
            self.logger.config.update(self.config.config, allow_val_change=True)

    def _generate_dataset(self, generation_epoch: int, ratings_column: str = None, **_):
        config = self.generation_phase
        log_df = self.cache.try_restore_log_df(generation_epoch, logger=self.print_with_timestamp)

        if log_df is None:
            self.print_with_timestamp("Generating dataset...")
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

        log_df = prepare_log_df(log_df, ratings_column=ratings_column)
        self.cache.try_cache_log_df(
            log_df=log_df, generation_epoch=generation_epoch, logger=self.print_with_timestamp
        )
        return log_df

    def _init_rl_setting(
            self, log_df: pd.DataFrame,
            top_k: int, ratings_column,
            mdp_settings: TConfig, scorer: TConfig, algo_settings: TConfig
    ):
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
            }]

        train_log, test_log = split_dataframe(log_df, time_sorted=True)

        mdp_prep, train_mdp, algo_logger = self.data2mdp(train_log, top_k, mdp_settings, scorer)
        mdp_settings['reward_function_name'] = "relevance_based_reward"
        mdp_settings['episode_splitter_name'] = "interaction_interruption"
        _, _, algo_test_logger = self.data2mdp(test_log, top_k, mdp_settings, scorer)

        self.preparator = mdp_prep
        self.algo_logger = algo_logger
        self.algo_test_logger = algo_test_logger

        # Init RL algorithm
        if not self.learnable_model:
            initial_user_embeddings, initial_item_embeddings = None, None
            init_with_als = False
            if init_with_als:
                emb_dim = None
                initial_user_embeddings, initial_item_embeddings = init_als_embeddings(
                    data=log_df, emb_dim=emb_dim
                )

            model = init_hidden_state_encoder(
                user_num=self.env.n_users, item_num=self.env.n_items + 1,
                initial_user_embeddings=initial_user_embeddings,
                initial_item_embeddings=initial_item_embeddings,
                **algo_settings['model_parameters']
            )
            algo = init_algo(model, **algo_settings['general_parameters'])
            self.model = algo
            self.learnable_model = True

        # Run experiment
        config = self.learning_phase
        fitter = self.model.fitter(
            dataset=train_mdp, n_epochs=config.epochs,
            verbose=False, save_metrics=False, show_progress=False,
        )
        return fitter, dataset_info

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
                    self.model, self.algo_test_logger,
                    train_logger=self.algo_logger,
                    env=self.env, framestack=self.framestack,
                    looking_for=[0, 1, 6], dataset_info=dataset_info,
                    rng=self.rng
                )
            total_epoch += 1
        return total_epoch

    def data2mdp(self, log, top_k, mdp_settings, scorer):
        # TODO: one preparator shuld transform different datasets?
        preparator = make_mdp(data=log, **mdp_settings)
        states, rewards, actions, terminations, state_tail = preparator.create_mdp()
        mdp = to_d3rlpy_form_ND(
            states, rewards, actions, terminations,
            discrete=scorer['prediction_type'] == "discrete"
        )
        algo_logger = init_logger(
            mdp, state_tail, log, top_k, wandb_logger=self.logger, **scorer
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
