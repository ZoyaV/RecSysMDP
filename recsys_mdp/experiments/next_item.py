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

from generate_synth_data import log_satiation
from recsys_mdp.experiments.utils.algorithm_constuctor import init_model, init_algo
from recsys_mdp.experiments.utils.cache import ExperimentCache, hex_digest
from recsys_mdp.experiments.utils.mdp_constructor import (
    make_mdp, split_dataframe,
    prepare_log_df, restore_log_df, cache_log_df
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
from recsys_mdp.simulator.user_state import (
    USER_RESET_MODE_DISCONTINUE
)
from recsys_mdp.utils.base import get_cuda_device
from recsys_mdp.utils.run.config import (
    TConfig, GlobalConfig, extracted
)
from recsys_mdp.utils.run.timer import timer, print_with_timestamp
from recsys_mdp.utils.run.wandb import get_logger
from run_experiment import eval_algo

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

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            generation_phase: TConfig, learning_phase: TConfig,
            zoya_settings: TConfig,
            model: TConfig, env: TConfig,
            log: bool, cuda_device: bool | int | None,
            project: str = None, wandb_init: TConfig = None,
            cache: TConfig = None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )

        self.logger = self.config.resolve_object(
            dict(config=config, log=log, project=project) | isnone(wandb_init, {}),
            object_type_or_factory=get_logger
        )
        logging.disable(logging.DEBUG)
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )

        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        d3rlpy.seed(seed)

        self.generation_phase = GenerationPhaseParameters(**generation_phase)
        self.learning_phase = LearningPhaseParameters(**learning_phase)
        self.zoya_settings = zoya_settings

        self.env: NextItemEnvironment = self.config.resolve_object(
            env, object_type_or_factory=NextItemEnvironment
        )
        self.model: LearnableBase = self.config.resolve_object(
            model | dict(use_gpu=get_cuda_device(cuda_device)),
            n_actions=self.env.n_items
        )
        self.learnable_model = False
        self.preparator = None

        self.cache = None
        if self.generation_phase.use_cache:
            env_config, _ = self.config.resolve_object_requirements(
                env, object_type_or_factory=NextItemEnvironment
            )
            # remove global config object
            env_config, _ = extracted(env_config, 'global_config')

            generated_data_minimum_config = generation_phase | env_config
            self.cache = self.config.resolve_object(
                cache, object_type_or_factory=ExperimentCache,
                experiment_id=hex_digest(generated_data_minimum_config),
            )
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
        # TODO: need to be a parametr
        dataset_info = None
        for generation_epoch in range(self.generation_phase.epochs):
            self.print_with_timestamp(f'Epoch: {generation_epoch} ==> learning')
            total_epoch += self._learn_on_dataset(
                total_epoch, fitter, dataset_info
            )

        self.print_with_timestamp('<==')

    def log_df_cache_filepath(self, generation_epoch: int):
        if self.cache is None:
            return None
        return self.cache.root / f'epoch_{generation_epoch}.pkl.gzip'

    def _generate_dataset(self, generation_epoch: int, ratings_column: str = None, **_):
        config = self.generation_phase

        use_cache = self.cache is not None
        cache_filepath = self.log_df_cache_filepath(generation_epoch)

        log_df = restore_log_df(cache_filepath) if use_cache else None
        if log_df is None:
            self.print_with_timestamp("Not use caching or didn't find data, generating it then")
            samples = []
            for episode in count():
                trajectory = self._generate_episode(first_run=True, use_env_actions=True)
                samples.extend(trajectory)
                if episode >= config.episodes_per_epoch or len(samples) >= config.samples_per_epoch:
                    break

            log_df = pd.DataFrame(samples, columns=[
                TIMESTAMP_COL,
                USER_ID_COL, ITEM_ID_COL,
                RELEVANCE_CONT_COL, RELEVANCE_INT_COL,
                TERMINATE_COL, "best_from_env"
            ])
        else:
            self.print_with_timestamp(f'Data restored from {cache_filepath}')

        log_df = prepare_log_df(log_df, ratings_column=ratings_column)

        if use_cache:
            cache_log_df(path=cache_filepath, log_df=log_df)
            self.print_with_timestamp(f'Data cached to {cache_filepath}')

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

        self.mdp_prep = mdp_prep
        self.algo_logger = algo_logger
        self.algo_test_logger = algo_test_logger

        # Init RL algorithm
        if not self.learnable_model:
            model = init_model(
                data=log_df,
                user_num=self.env.n_users, item_num=self.env.n_items,
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

    def _learn_on_dataset(self, total_epoch, fitter, dataset_info = None):
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % self.learning_phase.eval_schedule == 0:
                self.print_with_timestamp(f'Epoch: {epoch} | Total epoch: {total_epoch}')
                eval_algo(
                    self.model, self.algo_test_logger,
                    train_logger=self.algo_logger, env=self.env,
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

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)

    def set_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('mae', step_metric='epoch')

