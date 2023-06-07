from __future__ import annotations

import datetime
from typing import Any

import numpy as np
from numpy.random import Generator

from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RATING_COL, RELEVANCE_INT_COL, TERMINATE_COL, TRUNCATED_COL
)
from recsys_mdp.mdp.utils import isnone
from recsys_mdp.simulator.embeddings import Embeddings
from recsys_mdp.simulator.user_state import (
    USER_RESET_MODE_CONTINUE, USER_RESET_MODE_INIT,
    USER_RESET_MODE_DISCONTINUE, UserState
)
from recsys_mdp.utils.run.config import GlobalConfig, TConfig


class NextItemEnvironment:
    discrete: bool
    n_users: int
    n_items: int
    n_ratings: int

    embeddings: Embeddings

    max_episode_len: tuple[int, int]
    global_timestep: int
    timestep: int
    timestamp: datetime.datetime
    state: UserState
    states: list[UserState]
    current_max_episode_len: int

    dummy: dict[str, Any]
    compiled_dummies: dict[str, Any]

    def __init__(
            self, global_config: GlobalConfig, seed: int,
            n_users: int, n_items: int, max_rating: int,
            embeddings: TConfig,
            max_episode_len: int | tuple[int, int],
            user_state: TConfig,
            dummy: dict[str, Any]
    ):
        self.global_config = global_config
        self.rng = np.random.default_rng(seed)

        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = max_rating + 1

        self.embeddings = global_config.resolve_object(
            embeddings | dict(
                global_config=self.global_config, seed=seed,
                n_users=n_users, n_items=n_items
            ),
            object_type_or_factory=Embeddings
        )

        user_state, _ = self.global_config.resolve_object_requirements(user_state)
        self.states = [
            UserState(user_id=user_id, embeddings=self.embeddings, rng=self.rng, **user_state)
            for user_id in range(self.n_users)
        ]

        if isinstance(max_episode_len, int):
            max_episode_len = (max_episode_len, 0)
        avg_len, delta = max_episode_len
        self.max_episode_len = (avg_len - delta, avg_len + delta)

        self.global_timestep = self.timestep = 0
        self.timestamp = random_datetime(self.rng, end_year=2019)
        self.dummy = dummy
        self._extend_dummy()

    def set_rating_type(self, discrete: bool):
        self.discrete = discrete
        self.compiled_dummies = self.compile_dummy_values()

    def hard_reset(self, mode: str = USER_RESET_MODE_INIT):
        self.global_timestep = self.timestep = 0

        assert mode in [USER_RESET_MODE_INIT, USER_RESET_MODE_DISCONTINUE], \
            f'Env hard reset mode "{mode}" does not supported.'

        for user in self.states:
            user.reset(mode)

    def reset(
            self, user_id: int = None, mode: str = USER_RESET_MODE_CONTINUE,
            supply_info: set[str] = None
    ):
        if user_id is None:
            user_id = self.rng.integers(self.n_users)

        self.state = self.states[user_id]

        assert mode in [USER_RESET_MODE_CONTINUE, USER_RESET_MODE_DISCONTINUE], \
            f'Env reset mode "{mode}" does not supported.'
        self.state.reset(mode)

        self.timestep = 0
        self.timestamp += pause_random_duration(self.rng)
        self.current_max_episode_len = self.rng.integers(*self.max_episode_len)

        info = self.compiled_dummies | {
            TIMESTAMP_COL: self.timestamp,
            USER_ID_COL: self.state.user_id,
        }

        # if you need something in addition, add it here conditionally
        supply_info = isnone(supply_info, [])
        return self.state.user_id, info

    def step(self, item_id: int, supply_info: set[str] = None):
        relevance = self.state.step(item_id)

        self.timestep += 1
        self.global_timestep += 1
        self.timestamp += track_random_duration(self.rng)

        # stop by time-limit (user state independent effect)
        truncated = self.timestep >= self.current_max_episode_len
        # stop by preferences (user state dependent effect)
        terminate = self.state.sample_stop_listening(self.timestep)
        rating = relevance[self.discrete]

        info = {
            TIMESTAMP_COL: self.timestamp,
            USER_ID_COL: self.state.user_id,
            ITEM_ID_COL: item_id,
            RELEVANCE_CONT_COL: relevance[0],
            RELEVANCE_INT_COL: relevance[1],
            RATING_COL: rating,
            TRUNCATED_COL: truncated,
            TERMINATE_COL: terminate,
        }

        # if you need something in addition, add it here conditionally
        supply_info = isnone(supply_info, [])
        return rating, terminate, truncated, info

    def _extend_dummy(self):
        """Process passed dummy defaults and supply them with additional env-specific ones."""
        self.dummy.update({
            ITEM_ID_COL: self.n_items
        })

    def compile_dummy_values(self):
        return {
            # convention: dummy values passed as tuple/list treated as (cont_val, disc_val)
            key: value[self.discrete] if isinstance(value, (list, tuple)) else value
            for key, value in self.dummy.items()
        } | {
            RELEVANCE_CONT_COL: self.dummy[RATING_COL][0],
            RELEVANCE_INT_COL: self.dummy[RATING_COL][1],
            TRUNCATED_COL: False,
            TERMINATE_COL: False,
        }


def random_datetime(
        rng: Generator, start_year: int = 2019, end_year: int = 2021
) -> datetime.datetime:
    return datetime.datetime(
        year=rng.integers(start_year, end_year, endpoint=True),
        month=rng.integers(1, 12, endpoint=True),
        day=rng.integers(1, 28, endpoint=True),
        hour=rng.integers(1, 24, endpoint=True),
        minute=rng.integers(1, 60, endpoint=True),
        second=rng.integers(1, 60, endpoint=True)
    )


def pause_random_duration(rng: Generator) -> datetime.timedelta:
    return datetime.timedelta(minutes=float(rng.integers(15, 600, endpoint=True)))


def track_random_duration(rng: Generator) -> datetime.timedelta:
    return datetime.timedelta(seconds=float(rng.integers(120, 260, endpoint=True)))
