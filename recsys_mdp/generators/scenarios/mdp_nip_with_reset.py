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
from recsys_mdp.generators.utils.base import sample_rng, sample_int, lin_sum, update_exp_trace
from recsys_mdp.generators.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from recsys_mdp.generators.utils.timer import timer, print_with_timestamp

from mdp_next_item_integration import (
    MdpGenerationProcessParameters, LearningProcessParameters,
    TypesResolver, NextItemEnvironment, get_cuda_device
)

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

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

