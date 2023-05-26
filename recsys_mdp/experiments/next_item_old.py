from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from d3rlpy.base import LearnableBase
from numpy.random import Generator

from recsys_mdp.experiments.utils.type_resolver import TypesResolver
from recsys_mdp.simulator.env import (
    NextItemEnvironment
)
from recsys_mdp.experiments.utils.phases import (
    GenerationPhaseParameters,
    LearningPhaseParameters
)
from recsys_mdp.utils.run.wandb import get_logger
from recsys_mdp.utils.base import get_cuda_device
from recsys_mdp.utils.run.config import (
    TConfig, GlobalConfig
)
from recsys_mdp.utils.run.timer import timer, print_with_timestamp
from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RELEVANCE_INT_COL, TERMINATE_COL, RATING_COL
)

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class MdpNextItemExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int
    rng: Generator

    generation_config: GenerationPhaseParameters
    learning_config: LearningPhaseParameters

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
        self.generation_config = GenerationPhaseParameters(**generation)
        self.learning_config = LearningPhaseParameters(**learning)
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
            if episode >= config.episodes_per_epoch or len(samples) >= config.samples_per_epoch:
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
        from utils.mdp_constructor import make_mdp
        from recsys_mdp.mdp.utils import to_d3rlpy_form_ND
        from utils.algorithm_constuctor import init_model
        from utils.algorithm_constuctor import init_algo
        from utils.scorers_constructor import init_logger
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
