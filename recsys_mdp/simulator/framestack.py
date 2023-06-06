from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from recsys_mdp.mdp.base import USER_ID_COL, ITEM_ID_COL, RATING_COL, TIMESTAMP_COL

SUPPORTED_KEYS = {USER_ID_COL, ITEM_ID_COL, RATING_COL, TIMESTAMP_COL}


class Framestack:
    size: int

    # keys attribute keeps the order of keys in observation (i.e. its binary structure)
    keys: list[str]
    dummy: dict[str, Any]
    memory: dict[str, deque]
    components: dict[str, tuple[int, int]]

    shape: tuple[int]

    def __init__(self, size: int, keys: list[str]):
        keys_with_memory = [_resolve_memory_for_key(key, size) for key in keys]

        self.size = size
        self.keys = [key for key, _ in keys_with_memory]
        self.memory = dict(keys_with_memory)
        self.components = self._get_observation_components()
        self.shape = (self.components[self.keys[-1]][1],)

    def reset(self, **raw_observation: dict[str, Any]):
        # NB: rewrite entire memory on reset
        for key in self.keys:
            value = raw_observation.get(key, None)
            if value is None:
                value = self.dummy[key]
            self._fill(key, value)
        return self.observation()

    def step(self, **raw_observation: dict[str, Any]):
        # NB: append to memory only passed values for specified keys and leave the rest unchanged
        # e.g. user_id is passed only on reset and is kept unchanged for the rest of it
        for key, value in raw_observation.items():
            if (memory := self.memory.get(key)) is not None:
                memory.append(value)
        return self.observation()

    def observation_dict(self) -> dict[str, np.ndarray]:
        """Return dict of arrays"""
        return {
            key: np.array(self.memory[key])
            for key in self.memory
        }

    def observation(self) -> np.ndarray:
        return self.compile_observation(observation=self.observation_dict())

    def compile_observation(self, observation: dict) -> np.ndarray:
        """Compile observation dict into np.array"""
        return np.concatenate([
            # it's crucial to equalize their type to float
            observation[key].astype(float)
            for key in self.keys
        ])

    def _get_observation_components(self):
        """Return dict of tuples (start, end), each defines a range of indices for a key."""
        components = {}
        start = 0
        for key in self.keys:
            end = start + self.memory[key].maxlen
            components[key] = start, end
            start = end
        return components

    def _fill(self, key, value):
        """Fill memory for specific `key` with passed value."""
        m = self.memory[key]
        for _ in range(m.maxlen):
            m.append(value)


def _resolve_memory_for_key(key: str, size: int):
    if key in SUPPORTED_KEYS:
        # keep only the current value
        return key, deque(maxlen=1)
    elif key[-1] == 's' and key[:-1] in SUPPORTED_KEYS:
        # keep `size` last items
        return key[:-1], deque(maxlen=size)
    else:
        raise ValueError(f'Key {key} is not supported!')
