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

from  recsys_mdp.generators.scenarios.mdp_next_item_integration import (
    MdpGenerationProcessParameters, LearningProcessParameters,
    TypesResolver, NextItemEnvironment, get_cuda_device
)
### Rewrrite eval as part of Experiment class
from run_experiment import eval_algo

from recsys_mdp.mdp_former.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RELEVANCE_INT_COL, TERMINATE_COL, RATING_COL
)


import wandb

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

def log_satiation(logger, satiation, user_id):
    if logger is None:
        return
    hist = (satiation, np.arange(len(satiation)+1))
    histogram = logger.Histogram(np_histogram=hist)
    logger.log({f'user_{user_id}_satiation': histogram})

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



class NIP_with_reset:
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
        self.mdp_preparator = None

    def run(self):
        logging.disable(logging.DEBUG)
        self.set_metrics()

        self.print_with_timestamp('==> Run')
        total_epoch = 0
        self.print_with_timestamp(f'Generate dataset')
        dataset = self._generate_dataset()
        fitter = self._init_rl_setting(
             dataset,
            **self.zoya_settings
        )
        for generation_epoch in range(self.generation_config.epochs):
            self.print_with_timestamp(f'Epoch: {generation_epoch} ==> learning')
            total_epoch += self._learn_on_dataset(
                total_epoch, fitter
            )

        self.print_with_timestamp('<==')

    def _generate_dataset(self):
        config = self.generation_config
        samples = []
        for episode in count():
            samples.extend(self._generate_episode(first_run=True, use_env_actions=True))

            if config.episodes_per_epoch is not None and episode >= config.episodes_per_epoch:
                break
            if config.samples_per_epoch is not None and len(samples) >= config.samples_per_epoch:
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
    def _generate_episode(self, cold_start = False, user_id = None,
                          use_env_actions = False, log_sat = False, first_run = False):
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
            #generate new observation with framestack
                _, obs = self.mdp_preparator.make_interaction(relevance=discrete_relevance, user=user_id, item=item_id,
                                                     ts=timestamp,obs_prev=obs,relevance2reward=False)

            items_top = env.state.ranked_items(with_satiation=True, discrete=True)
            if use_env_actions:
                item_id =  self.rng.choice(items_top[:N_BEST_ITEMS])
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
            if epoch == 1 or epoch %  self.learning_config.eval_schedule == 0:
                eval_algo(
                    self.model, self.algo_logger, train_logger=self.algo_logger, env=self.env,
                    looking_for=[0, 1, 6]
                )
            total_epoch += 1
        return total_epoch
    def _init_rl_setting(
            self, dataset: list[tuple],
            top_k: int,ratings_column,
            mdp_settings: TConfig, scorer: TConfig, algo_settings: TConfig
    ):
        from constructors.mdp_constructor import make_mdp
        from recsys_mdp.mdp_former.utils import to_d3rlpy_form_ND
        from constructors.algorithm_constuctor import init_model
        from constructors.algorithm_constuctor import init_algo
        from constructors.scorers_constructor import init_scorers
        from constructors.scorers_constructor import init_logger

        log = pd.DataFrame(dataset, columns=[
            TIMESTAMP_COL,
            USER_ID_COL, ITEM_ID_COL,
            RELEVANCE_CONT_COL, RELEVANCE_INT_COL,
            TERMINATE_COL
        ])
        log[RATING_COL] = log[ratings_column]


      #  train_values = get_values_fixme(log, data_mapping)
        self.mdp_preparator = make_mdp(data=log, **mdp_settings)
        states, rewards, actions, terminations, state_tail = self.mdp_preparator.create_mdp()
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
       # scorers = init_scorers(state_tail, train_values, top_k, **scorer)
        self.algo_logger = init_logger(
            train_mdp, state_tail, log, top_k, wandb_logger=self.logger, **scorer
        )

        # Run experiment
        config = self.learning_config
        fitter = self.model.fitter(
            dataset=train_mdp, n_epochs=config.epochs,
            verbose=False, save_metrics=False, show_progress=False,
        )
        return fitter

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

