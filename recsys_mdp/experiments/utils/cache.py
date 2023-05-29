from recsys_mdp.utils.run.cache import CacheDirectory, hex_digest
from recsys_mdp.utils.run.config import TConfig


class ExperimentCache:
    cache: CacheDirectory

    def __init__(
            self, *, cache_root: str, config: TConfig, keep_last_n_experiments: int = None
    ):
        self.cache = CacheDirectory(
            cache_root=cache_root, unique_id=hex_digest(config),
            keep_last_n_entries=keep_last_n_experiments
        )

    @property
    def root(self):
        return self.cache.root
