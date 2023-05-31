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
from recsys_mdp.experiments.utils.helper import eval_algo

from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RELEVANCE_INT_COL, TERMINATE_COL, RATING_COL
)

from recsys_mdp.mdp.utils import to_d3rlpy_form_ND, isnone

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class NextItemOnDataExperiment:
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
        train_log = pd.read_csv('./row_data/train.csv')
        test_log = pd.read_csv('./row_data/test.csv')
        fitter = self._init_rl_setting(
             train_log, test_log, **self.zoya_settings
        )

        total_epoch += self._learn_on_dataset(
            total_epoch, fitter
        )

        self.print_with_timestamp('<==')



    def _learn_on_dataset(self, total_epoch, fitter, dataset_info = None):
        for epoch, metrics in fitter:
            if epoch == 1 or epoch %  self.learning_phase.eval_schedule == 0:
                eval_algo(
                    self.model, self.algo_test_logger, train_logger=self.algo_logger, env=None,
                    dataset_info = dataset_info
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
            self, train_log, test_log,
            top_k: int,ratings_column,
            mdp_settings: TConfig, scorer: TConfig, algo_settings: TConfig
    ):
        train_log[TIMESTAMP_COL] = pd.to_datetime(train_log[TIMESTAMP_COL])
        test_log[TIMESTAMP_COL] = pd.to_datetime(test_log[TIMESTAMP_COL])
        mdp_prep, train_mdp, algo_logger = self.data2mdp(train_log, top_k, mdp_settings, scorer)
        mdp_settings['reward_function_name'] = "relevance_based_reward"
        mdp_settings['episode_splitter_name'] = "interaction_interruption"
        _, _, algo_test_logger = self.data2mdp(test_log, top_k, mdp_settings, scorer)

        self.mdp_prep = mdp_prep
        self.algo_logger = algo_logger
        self.algo_test_logger = algo_test_logger

        # Init RL algorithm
        if not self.learnable_model:
            from recsys_mdp.experiments.utils.algorithm_constuctor import init_hidden_state_encoder
            from recsys_mdp.experiments.utils.algorithm_constuctor import init_algo
            model = init_hidden_state_encoder(data=train_log, **algo_settings['model_parameters'])
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

    def print_with_timestamp(self, *args):
        print_with_timestamp(self.init_time, *args)

    def set_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        self.logger.define_metric('mae', step_metric='epoch')

