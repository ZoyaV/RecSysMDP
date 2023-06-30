from __future__ import annotations

import datetime
from typing import Any

from numpy.random import Generator

from recsys_mdp.mdp.base import (
    TIMESTAMP_COL, USER_ID_COL, ITEM_ID_COL, RELEVANCE_CONT_COL,
    RATING_COL, RELEVANCE_INT_COL, TERMINATE_COL, TRUNCATED_COL
)
from recsys_mdp.mdp.utils import isnone
from recsys_mdp.simulator.embeddings import Embeddings
from recsys_mdp.simulator.user_state import (
    USER_RESET_MODE_CONTINUE, User, VolatileUserState
)
from recsys_mdp.simulator.utils import EpisodicRandomGenerator
from recsys_mdp.utils.base import sample_rng
from recsys_mdp.utils.run.config import GlobalConfig, TConfig


class EnvCheckpoint:
    global_timestep: int
    timestamp: datetime.datetime

    users: list[VolatileUserState]

    current_user: User
    current_max_episode_len: int

    def __init__(
            self, *, global_timestep: int, timestamp: datetime.datetime,
            users: list[VolatileUserState],
            current_user: User, current_max_episode_len: int
    ):
        self.global_timestep = global_timestep
        self.timestamp = timestamp
        self.users = users
        self.current_user = current_user
        self.current_max_episode_len = current_max_episode_len


class NextItemEnvironment:
    episodic_rng: EpisodicRandomGenerator

    discrete: bool
    n_users: int
    n_items: int
    n_ratings: int

    embeddings: Embeddings

    max_episode_len: tuple[int, int]
    global_timestep: int
    timestep: int
    timestamp: datetime.datetime
    state: User
    states: list[User]
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
        self.episodic_rng = EpisodicRandomGenerator(seed)
        self.global_config = global_config

        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = max_rating + 1
        self.max_episode_len = parse_max_episode_len(max_episode_len)

        self.embeddings = global_config.resolve_object(
            embeddings, object_type_or_factory=Embeddings, n_users=n_users, n_items=n_items
        )

        shared_user_state = self.global_config.resolve_object(
            user_state, embeddings=self.embeddings
        )
        self.states = [
            User(user_id=user_id, rng=sample_rng(self.rng), shared_state=shared_user_state)
            for user_id in range(self.n_users)
        ]

        self.global_timestep = self.timestep = 0
        self.timestamp = random_datetime(self.rng, end_year=2019)
        self.dummy = dummy
        self.extend_dummy()

    def set_rating_type(self, discrete: bool):
        self.discrete = discrete
        self.compiled_dummies = self.compile_dummy_values()

    def reset(
            self, user_id: int = None, mode: str = USER_RESET_MODE_CONTINUE,
            supply_info: set[str] = None
    ):
        self.episodic_rng.transit_to_next_episode()
        if user_id is None:
            user_id = self.rng.integers(self.n_users)

        self.state = self.states[user_id]
        self.state.reset(mode)

        self.timestep = 0
        self.timestamp += pause_random_duration(self.rng)
        self.current_max_episode_len = self.user_rng.integers(*self.max_episode_len)

        info = self.compiled_dummies | {
            TIMESTAMP_COL: self.timestamp,
            USER_ID_COL: self.state.id,
        }

        # if you need something in addition, add it here conditionally
        supply_info = isnone(supply_info, [])
        return self.state.id, info

    def step(self, item_id: int, supply_info: set[str] = None):
        relevance = self.state.relevance(item_id)

        cont_relevance, discrete_relevance = relevance
        self.state.consume_item(item_id=item_id, grade=discrete_relevance)

        self.timestep += 1
        self.global_timestep += 1
        self.timestamp += track_random_duration(self.user_rng)

        # stop by time-limit (user state independent effect)
        truncated = self.timestep >= self.current_max_episode_len
        # stop by preferences (user state dependent effect)
        terminate = self.sample_stop_listening()
        rating = relevance[self.discrete]

        info = {
            TIMESTAMP_COL: self.timestamp,
            USER_ID_COL: self.state.id,
            ITEM_ID_COL: item_id,
            RELEVANCE_CONT_COL: cont_relevance,
            RELEVANCE_INT_COL: discrete_relevance,
            RATING_COL: rating,
            TRUNCATED_COL: truncated,
            TERMINATE_COL: terminate,
        }

        # if you need something in addition, add it here conditionally
        supply_info = isnone(supply_info, [])
        return rating, terminate, truncated, info

    def sample_stop_listening(self):
        min_prob = self.state.shared.early_stop_min_prob
        k = self.state.shared.early_stop_delta
        satisfaction = self.state.volatile.satisfaction
        deterministic_early_stop = self.state.shared.deterministic_early_stop

        dissatisfaction = 5 - satisfaction

        # early stop increases with increasing speed based on dissatisfaction
        probability = min_prob + k * dissatisfaction * (dissatisfaction + 1) / 2

        if deterministic_early_stop:
            max_time_steps = (1 - probability) / probability
            return self.timestep >= max_time_steps

        # sample 3 times and take mean to smooth dependency by reducing variability
        return self.user_rng.random() < probability

    def make_checkpoint(self):
        # save checkpoint only after episode termination, but before reset!
        return EnvCheckpoint(
            global_timestep=self.global_timestep,
            timestamp=self.timestamp,
            users=[user.volatile.copy() for user in self.states],
            current_user=self.state,
            current_max_episode_len=self.current_max_episode_len
        )

    def restore_checkpoint(self, checkpoint: EnvCheckpoint):
        self.global_timestep = checkpoint.global_timestep
        self.timestamp = checkpoint.timestamp
        for i in range(self.n_users):
            # NB: it's crucial to pass a copy of the volatile state!
            self.states[i].volatile = checkpoint.users[i].copy()
        self.state = checkpoint.current_user
        self.current_max_episode_len = checkpoint.current_max_episode_len

    @property
    def rng(self):
        return self.episodic_rng.rng

    @property
    def user_rng(self):
        return self.state.rng

    def extend_dummy(self):
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
        hour=rng.integers(24),
        minute=rng.integers(60),
        second=rng.integers(60)
    )


def pause_random_duration(rng: Generator) -> datetime.timedelta:
    return datetime.timedelta(minutes=float(rng.integers(15, 600, endpoint=True)))


def track_random_duration(rng: Generator) -> datetime.timedelta:
    return datetime.timedelta(seconds=float(rng.integers(120, 260, endpoint=True)))


def parse_max_episode_len(max_episode_len) -> tuple[int, int]:
    if isinstance(max_episode_len, int):
        max_episode_len = (max_episode_len, 0)
    avg_len, delta = max_episode_len
    return avg_len - delta, avg_len + delta
