from __future__ import annotations

import datetime

import numpy as np
from numpy.random import Generator

from recsys_mdp.simulator.embeddings import Embeddings
from recsys_mdp.simulator.user_state import (
    USER_RESET_MODE_CONTINUE, USER_RESET_MODE_INIT,
    USER_RESET_MODE_DISCONTINUE, UserState
)
from recsys_mdp.utils.run.config import GlobalConfig, TConfig


class NextItemEnvironment:
    n_users: int
    n_items: int

    embeddings: Embeddings

    max_episode_len: tuple[int, int]
    global_timestep: int
    timestep: int
    timestamp: datetime.datetime
    state: UserState
    states: list[UserState]
    current_max_episode_len: int

    def __init__(
            self, global_config: GlobalConfig, seed: int,
            n_users: int, n_items: int,
            embeddings: TConfig,
            max_episode_len: int | tuple[int, int],
            user_state: TConfig,
    ):
        self.global_config = global_config
        self.rng = np.random.default_rng(seed)

        self.n_users = n_users
        self.n_items = n_items
        self.embeddings = global_config.resolve_object(
            embeddings | dict(
                global_config=self.global_config, seed=seed,
                n_users=n_users, n_items=n_items
            ),
            object_type_or_factory=Embeddings
        )
        self.states = [
            UserState(user_id, embeddings=self.embeddings, rng=self.rng, **user_state)
            for user_id in range(self.n_users)
        ]

        if isinstance(max_episode_len, int):
            max_episode_len = (max_episode_len, 0)
        avg_len, delta = max_episode_len
        self.max_episode_len = (avg_len - delta, avg_len + delta)

        self.global_timestep = self.timestep = 0
        self.timestamp = random_datetime(self.rng, end_year=2019)

    def hard_reset(self, mode: str = USER_RESET_MODE_INIT):
        self.global_timestep = self.timestep = 0

        assert mode in [USER_RESET_MODE_INIT, USER_RESET_MODE_DISCONTINUE], \
            f'Env hard reset mode "{mode}" does not supported.'

        for user in self.states:
            user.reset(mode)

    def reset(self, user_id: int = None, mode: str = USER_RESET_MODE_CONTINUE):
        if user_id is None:
            user_id = self.rng.integers(self.n_users)

        self.state = self.states[user_id]

        assert mode in [USER_RESET_MODE_CONTINUE, USER_RESET_MODE_DISCONTINUE], \
            f'Env reset mode "{mode}" does not supported.'
        self.state.reset(mode)

        self.timestep = 0
        self.timestamp += pause_random_duration(self.rng)
        self.current_max_episode_len = self.rng.integers(*self.max_episode_len)
        return self.state.user_id

    def step(self, item_id: int):
        relevance = self.state.step(item_id)

        self.timestep += 1
        self.global_timestep += 1
        self.timestamp += track_random_duration(self.rng)

        terminated = self.timestep >= self.current_max_episode_len
        terminated |= self.state.sample_stop_listening()

        return relevance, terminated


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
