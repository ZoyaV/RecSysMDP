from __future__ import annotations

from recsys_mdp.experiments.utils.mdp_constructor import restore_log_df, cache_log_df
from recsys_mdp.utils.run.cache import CacheDirectory, hex_digest
from recsys_mdp.utils.run.config import TConfig


class ExperimentCache:
    cache: CacheDirectory | None

    def __init__(
            self, *, enable: bool, cache_root: str, experiment_config: TConfig,
            keep_last_n_experiments: int = None
    ):
        self.cache = None
        if enable:
            self.cache = CacheDirectory(
                cache_root=cache_root, unique_id=hex_digest(experiment_config),
                keep_last_n_entries=keep_last_n_experiments
            )

    @property
    def root(self):
        if not self.enabled:
            return None
        return self.cache.root

    @property
    def enabled(self):
        return self.cache is not None

    def try_restore_log_df(self, generation_epoch: int, logger=None):
        if not self.enabled:
            if logger is not None:
                logger("Dataset caching is disabled")
            return None

        path = self._log_df_cache_filepath(generation_epoch)
        log_df = restore_log_df(path)

        if log_df is None:
            msg = "Didn't find cached dataset"
        else:
            msg = f'Dataset restored from {path}'
        if logger is not None:
            logger(msg)

        return log_df

    def try_cache_log_df(self, log_df, generation_epoch: int, logger=None):
        if not self.enabled:
            return None
        path = self._log_df_cache_filepath(generation_epoch)
        if path.exists():
            return None

        cache_log_df(path=path, log_df=log_df)
        if logger is not None:
            logger(f'Data cached to {path}')
        return path

    def _log_df_cache_filepath(self, generation_epoch: int):
        if not self.enabled:
            return None
        return self.cache.root / f'epoch_{generation_epoch}.pkl.gzip'
