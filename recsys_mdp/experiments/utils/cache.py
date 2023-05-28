import hashlib
from pathlib import Path
from typing import Any

from recsys_mdp.utils.run.config import TConfig


def cache_dataset(config: TConfig, keep_last_n_caches: int = 10):
    cache_root = Path('./cache')
    # clean up old cache, while keeping only the last N
    _clean_up_dir(cache_root, keep_last_n=keep_last_n_caches)

    unique_id = hex_digest(obj=config)
    dataset_root = cache_root / unique_id
    dataset_root.mkdir(exist_ok=True)



def _clean_up_dir(path: Path, keep_last_n: int = None):
    dir_content = [(f, f.stat().st_mtime) for f in path.iterdir()]
    dir_content = sorted(dir_content, key=lambda t: t[1], reverse=True)
    n = keep_last_n if keep_last_n is not None and keep_last_n > 1 else None
    for f, _ in dir_content[n:]:
        if f.is_dir():
            # clean up directory's content and remove it afterward
            _clean_up_dir(f)
        else:
            f.unlink()

    if n is None:
        # means: we've already removed any content, now remove the dir itself
        path.rmdir()


def hex_digest(path: str = None, obj: Any = None) -> str:
    # only one of the should be specified
    assert (path is None) != (obj is None)

    def get_hasher():
        return hashlib.blake2s(digest_size=14)

    if path is not None:
        with open(path, 'rb') as f:
            return hashlib.file_digest(f, get_hasher).hexdigest()

    hasher = get_hasher()
    hasher.update(repr(obj).encode('utf-8'))
    return hasher.hexdigest()
