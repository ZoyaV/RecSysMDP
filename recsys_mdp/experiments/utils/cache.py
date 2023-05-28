import hashlib
from pathlib import Path
from typing import Any


class ExperimentCache:
    root: Path

    def __init__(self, cache_root: str, experiment_id: str, keep_last_n_experiments: int):
        cache_root = Path(cache_root)
        self.root = cache_root / experiment_id

        # ensure all dirs along the path are created
        self.root.mkdir(exist_ok=True)
        # bump current cache folder modified time
        self.root.touch(exist_ok=True)

        # clean up old cache, while keeping only the last N
        _clean_up_dir(cache_root, keep_last_n=keep_last_n_experiments)


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


def hex_digest(obj: Any = None, path: str = None) -> str:
    """Provides a hash digest of an object or a file (provided via its path)."""

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
