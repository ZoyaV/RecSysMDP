from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from d3rlpy.base import LearnableBase
from numpy.random import Generator

from recsys_mdp.experiments.utils.mdp_constructor import make_mdp
from recsys_mdp.experiments.utils.type_resolver import TypesResolver
from recsys_mdp.utils.run.wandb import get_logger
from recsys_mdp.utils.run.config import (
    TConfig, GlobalConfig
)
from recsys_mdp.utils.run.timer import timer, print_with_timestamp

from recsys_mdp.simulator.env import (
    NextItemEnvironment
)
from recsys_mdp.experiments.utils.phases import (
    GenerationPhaseParameters,
    LearningPhaseParameters
)
from recsys_mdp.utils.base import get_cuda_device
from recsys_mdp.experiments.utils.scorers_constructor import init_logger
from run_experiment import eval_algo

from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RELEVANCE_INT_COL, TERMINATE_COL, RATING_COL
)

from recsys_mdp.mdp.utils import to_d3rlpy_form_ND, isnone

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def log_satiation(logger, satiation, user_id):
    if logger is None:
        return
    hist = (satiation, np.arange(len(satiation)+1))
    histogram = logger.Histogram(np_histogram=hist)
    logger.log({f'user_{user_id}_satiation': histogram})


class NextItemExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int
    rng: Generator

    generation_phase: GenerationPhaseParameters
    learning_phase: LearningPhaseParameters

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            generation_phase: TConfig, learning_phase: TConfig,
            zoya_settings: TConfig,
            model: TConfig, env: TConfig,
            log: bool, cuda_device: bool | int | None,
            project: str = None, wandb_init: TConfig = None,
            **_
    ):
        self.config = GlobalConfig(
            config=config, config_path=config_path, type_resolver=TypesResolver()
        )
        self.logger = self.config.resolve_object(
            dict(config=config, log=log, project=project) | isnone(wandb_init, {}),
            object_type_or_factory=get_logger
        )

        self.init_time = timer()
        self.print_with_timestamp('==> Init')

        self.seed = seed
        self.rng = np.random.default_rng(seed)
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

    def run(self):
        logging.disable(logging.DEBUG)
        self.set_metrics()

        self.print_with_timestamp('==> Run')
        total_epoch = 0
        self.print_with_timestamp(f'Generate dataset')
        dataset = self._generate_dataset()
        fitter = self._init_rl_setting(
             dataset, **self.zoya_settings
        )
        for generation_epoch in range(self.generation_phase.epochs):
            self.print_with_timestamp(f'Epoch: {generation_epoch} ==> learning')
            total_epoch += self._learn_on_dataset(
                total_epoch, fitter
            )

        self.print_with_timestamp('<==')

    def _generate_dataset(self):
        config = self.generation_phase
        samples = []
        for episode in count():
            trajectory = self._generate_episode(first_run=True, use_env_actions=True)
            samples.extend(trajectory)
            if episode >= config.episodes_per_epoch or len(samples) >= config.samples_per_epoch:
                break
        return samples

    def _framestack_from_last_best(self, user_id, N = 10):
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
        obs = np.random.randint(0, self.env.n_items, framestack_size).tolist() + [user_id]
        return obs

    def _generate_episode(
            self, cold_start = False, user_id = None,
            use_env_actions = False, log_sat = False, first_run = False
    ):
        env, model = self.env, self.model
        orig_user_id = user_id
        user_id = env.reset(user_id=user_id)
        trajectory = []
        N_BEST_ITEMS = 10
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
                item_id = self.rng.choice(items_top[:N_BEST_ITEMS])
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
        return trajectory

    def _learn_on_dataset(self, total_epoch, fitter):
        for epoch, metrics in fitter:
            if epoch == 1 or epoch %  self.learning_phase.eval_schedule == 0:
                eval_algo(
                    self.model, self.algo_test_logger, train_logger=self.algo_logger, env=self.env,
                    looking_for=[0, 1, 6]
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
        return preparator,mdp, algo_logger


    def _init_rl_setting(
            self, dataset: list[tuple],
            top_k: int,ratings_column,
            mdp_settings: TConfig, scorer: TConfig, algo_settings: TConfig
    ):

        log = pd.DataFrame(dataset, columns=[
            TIMESTAMP_COL,
            USER_ID_COL, ITEM_ID_COL,
            RELEVANCE_CONT_COL, RELEVANCE_INT_COL,
            TERMINATE_COL, "best_from_env"
        ])
        log[RATING_COL] = log[ratings_column]

        split_timestamp = log[TIMESTAMP_COL].quantile(0.7)

        # train/test split
        train_log = log[log[TIMESTAMP_COL] <= split_timestamp]
        test_log = log[log[TIMESTAMP_COL] > split_timestamp]

        mdp_prep, train_mdp, algo_logger = self.data2mdp(train_log, top_k, mdp_settings, scorer)
        _, _, algo_test_logger = self.data2mdp(test_log, top_k, mdp_settings, scorer)

        self.mdp_prep = mdp_prep
        self.algo_logger = algo_logger
        self.algo_test_logger = algo_test_logger

        # Init RL algorithm
        if not self.learnable_model:
            from recsys_mdp.experiments.utils.algorithm_constuctor import init_model
            from recsys_mdp.experiments.utils.algorithm_constuctor import init_algo
            model = init_model(data=log, **algo_settings['model_parameters'])
            algo = init_algo(model, **algo_settings['general_parameters'])
            self.model = algo
            self.learnable_model = True

        # Run experiment
        config = self.learning_phase
        fitter = self.model.fitter(
            dataset=train_mdp, n_epochs=config.epochs,
            verbose=False, save_metrics=False, show_progress=False,
        )
        return fitter

    def print_with_timestamp(self, text: str):
        print_with_timestamp(text, self.init_time)

    def set_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('mae', step_metric='epoch')

