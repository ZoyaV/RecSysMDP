from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from d3rlpy.algos import AlgoBase
from d3rlpy.dataset import MDPDataset

from recsys_mdp.generators.datasets.synthetic.dataset import (
    ToyRatingsDataset,
    ToyRatingsDatasetBuilder
)
from recsys_mdp.generators.datasets.synthetic.embeddings import (
    RandomEmbeddingsGenerator,
    RandomClustersEmbeddingsGenerator
)
from recsys_mdp.generators.datasets.synthetic.log import RandomLogGenerator
from recsys_mdp.generators.mdp.ratings import MdpDatasetBuilder

from recsys_mdp.generators.run.wandb import get_logger
from recsys_mdp.generators.utils.config import (
    TConfig, GlobalConfig, LazyTypeResolver
)
from recsys_mdp.generators.utils.timer import timer, print_with_timestamp

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class BanditsRatingsExperiment:
    config: GlobalConfig
    logger: Run | None

    init_time: float
    seed: int

    top_k: int
    epochs: int
    eval_schedule: int

    def __init__(
            self, config: TConfig, config_path: Path, seed: int,
            top_k: int, epochs: int, dataset: TConfig, mdp: TConfig, model: TConfig,
            train_test_split: TConfig, negative_samples: TConfig,
            log: bool, eval_schedule: int,
            cuda_device: bool | int | None,
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
        self.top_k = top_k
        self.epochs = epochs
        self.eval_schedule = eval_schedule

        self.dataset_generator = self.config.resolve_object(dataset)
        full_dataset = self.dataset_generator.generate()
        train_dataset = self.dataset_generator.split(full_dataset, **train_test_split)
        negative_samples = self.dataset_generator.generate_negative_samples(**negative_samples)
        train_log = pd.concat([train_dataset.log, negative_samples.log], ignore_index=True)
        train_log.sort_values(
            ['user_id', 'timestamp'],
            inplace=True,
            ascending=[True, False]
        )

        mdp_builder = MdpDatasetBuilder(**mdp)
        self.test_mdp = mdp_builder.build(full_dataset, use_ground_truth=True)
        self.train_mdp = mdp_builder.build(ToyRatingsDataset(
            log=train_log,
            user_embeddings=train_dataset.user_embeddings,
            item_embeddings=train_dataset.item_embeddings,
        ), use_ground_truth=False)
        self.model = self.config.resolve_object(
            model | dict(use_gpu=get_cuda_device(cuda_device))
        )

    def run(self):
        logging.disable(logging.DEBUG)
        self.set_metrics()

        self.print_with_timestamp('==> Run')
        fitter = self.model.fitter(
            self.train_mdp,
            n_epochs=self.epochs, verbose=False,
            save_metrics=False, show_progress=False,
        )
        for epoch, metrics in fitter:
            if epoch == 1 or epoch % self.eval_schedule == 0:
                self._eval_and_log(self.model, epoch)

        self.print_with_timestamp('<==')

    def _eval_and_log(self, model, epoch):
        metrics = self._eval_mae(model, self.test_mdp)

        mae, discrete_mae = metrics['mae'], metrics['discrete_mae']
        self.print_with_timestamp(
            f'Epoch {epoch:03}: mae {mae:.4f} | dmae {discrete_mae:.4f}'
        )
        if self.logger:
            metrics |= dict(epoch=epoch)
            self.logger.log(metrics)

    def _eval_mae(self, model: AlgoBase, dataset: MDPDataset):
        batch_size = model.batch_size
        n_splits = dataset.observations.shape[0] // batch_size
        test_prediction = np.concatenate([
            model.predict(batch)
            for batch in np.array_split(dataset.observations, n_splits)
        ])
        mae = np.mean(np.abs(test_prediction - dataset.actions))

        discrete_predictions = self.dataset_generator.relevance.discretize(test_prediction)
        discrete_gt = self.dataset_generator.relevance.discretize(dataset.actions)
        discrete_mae = np.mean(np.abs(discrete_predictions - discrete_gt))
        return {
            'mae': mae,
            'discrete_mae': discrete_mae,
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
            return ToyRatingsDatasetBuilder
        if type_name == 'ds_source.random':
            return RandomLogGenerator
        if type_name == 'embeddings.random':
            return RandomEmbeddingsGenerator
        if type_name == 'embeddings.clusters':
            return RandomClustersEmbeddingsGenerator
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


def get_cuda_device(cuda_device: int | None) -> int | bool:
    if cuda_device is not None:
        import torch.cuda
        cuda_available = torch.cuda.is_available()
        print(f'CUDA available: {cuda_available}; device: {cuda_device}')
        if not cuda_available:
            cuda_device = False
    return cuda_device
