from __future__ import annotations

import numpy as np
from numpy.random import Generator

from recsys_mdp.mdp.utils import isnone
from recsys_mdp.simulator.embeddings import Embeddings
from recsys_mdp.simulator.utils import boosting, EpisodicRandomGenerator
from recsys_mdp.utils.base import sample_int, update_exp_trace, lin_sum

USER_RESET_MODE_CONTINUE = 'continue'
USER_RESET_MODE_MEMORYLESS = 'memoryless'


class SharedUserState:
    # [min, mean]
    satiation_distribution: tuple[float, float]
    base_satiation_speed: float | tuple[float, float]
    satiation_drift: float

    item_listening_trace_decay: float
    item_listening_trace_reset_power: float
    item_repeat_penalty_power: float

    base_satisfaction: float
    satisfaction_speed: float
    satisfaction_reset_power: float

    early_stop_probs: tuple[float, float, int]
    deterministic_early_stop: bool = False

    discrete_actions_distr: list[tuple[float, float]]
    deterministic_actions: bool = False

    relevance_boosting_k: tuple[float, float]
    boosting_softness: tuple[float, float]
    embeddings: Embeddings

    def __init__(
            self, *,
            embeddings: Embeddings,
            satiation_distribution: float | tuple[float, float],
            base_satiation_speed: float | tuple[float, float],
            satiation_drift: float,
            item_listening_trace_decay: float,
            item_listening_trace_reset_power: float,
            item_repeat_penalty_power: float,
            base_satisfaction: float,
            satisfaction_speed: float,
            satisfaction_reset_power: float,
            early_stop_probs: tuple[float, float, int],
            relevance_boosting: tuple[float, float],
            boosting_softness: tuple[float, float],
            discrete_actions: list[tuple[float, float]],
            deterministic_actions: bool = False,
            deterministic_early_stop: bool = False,
    ):
        self.satiation_distribution = parse_satiation_distribution(satiation_distribution)
        self.base_satiation_speed = base_satiation_speed
        self.satiation_drift = satiation_drift

        self.item_listening_trace_decay = item_listening_trace_decay
        self.item_listening_trace_reset_power = item_listening_trace_reset_power
        self.item_repeat_penalty_power = item_repeat_penalty_power

        self.base_satisfaction = base_satisfaction
        self.satisfaction_speed = satisfaction_speed
        self.satisfaction_reset_power = satisfaction_reset_power

        self.early_stop_probs = early_stop_probs
        self.deterministic_early_stop = deterministic_early_stop
        self.discrete_actions_distr = discrete_actions
        self.deterministic_actions = deterministic_actions
        self.relevance_boosting_k = tuple(relevance_boosting)
        self.boosting_softness = tuple(boosting_softness)
        self.embeddings = embeddings


class StaticUserState:
    user_id: int
    satiation_speed: np.ndarray

    def __init__(self, *, user_id: int, satiation_speed: np.ndarray):
        self.user_id = user_id
        self.satiation_speed = satiation_speed


class VolatileUserState:
    episodic_rng: EpisodicRandomGenerator
    satiation: np.ndarray
    item_listening_trace: np.ndarray
    satisfaction: float

    def __init__(
            self, *, seed: int,
            satiation: np.ndarray, item_listening_trace: np.ndarray, satisfaction: float
    ):
        self.episodic_rng = EpisodicRandomGenerator(seed)
        self.satiation = satiation
        self.item_listening_trace = item_listening_trace
        self.satisfaction = satisfaction

    def transit_to_next_episode(self):
        # This ensures that nevertheless the successfulness of the intra-episode interaction
        # with the user, his random state is deterministically progressed over episodes.
        self.episodic_rng.transit_to_next_episode()

    def copy(self):
        # Make a copy before transitioning to the next episode and always reset after restore.
        # Well, it is aligned with the natural env flow and doesn't require additional actions.
        return VolatileUserState(
            seed=self.episodic_rng.seed,
            satiation=self.satiation.copy(),
            item_listening_trace=self.item_listening_trace.copy(),
            satisfaction=self.satisfaction
        )


class User:
    shared: SharedUserState
    static: StaticUserState
    volatile: VolatileUserState

    def __init__(self, user_id: int, rng: Generator, shared_state: SharedUserState):
        self.shared = shared_state
        self.static = self.create_static_state(user_id=user_id, rng=rng, shared_state=self.shared)
        self.volatile = self.create_volatile_state(rng=rng, shared_state=shared_state)
        # NB: hard reset before the usage is mandatory!
        self.reset(mode=USER_RESET_MODE_MEMORYLESS)

    def reset(self, mode: str = USER_RESET_MODE_CONTINUE):
        # deterministically sets a unique seed and rng for this episode
        self.volatile.transit_to_next_episode()

        if mode == USER_RESET_MODE_CONTINUE:
            self.drift_satiation()
            self.drift_satisfaction(acceleration=self.shared.satisfaction_reset_power)
            self.update_listening_trace(acceleration=self.shared.item_listening_trace_reset_power)
        elif mode == USER_RESET_MODE_MEMORYLESS:
            # reset user "memory" and re-sample his state
            # NB: CONTINUE mode with extreme params emulates it, so use it instead. This one is for
            # simultaneous use of them both.
            self.drift_satiation(speed=1.0)
            self.drift_satisfaction(speed=1.0)
            self.update_listening_trace(decay=0.0)
        else:
            raise ValueError(f'User reset mode "{mode}" does not supported.')

    def consume_item(self, item_id: int, grade: int):
        self.update_satisfaction(grade=grade)
        self.update_listening_trace(item_id=item_id)
        self.increase_satiation(item_id)

    def relevance(self, item_id: int = None, with_satiation: bool = True):
        if item_id is None:
            # compute relevance for all items
            all_relevance = [
                self.relevance(item_id, with_satiation=with_satiation)
                for item_id in range(self.shared.embeddings.n_items)
            ]
            continuous_relevance = np.array([cont for cont, discr in all_relevance])
            discrete_relevance = np.array([discr for cont, discr in all_relevance])
            return continuous_relevance, discrete_relevance

        # get item similarity to user preferences
        relevance = self.static_relevance(item_id)

        if with_satiation:
            relevance *= self.satiation_boosting(item_id) * self.repeat_boosting(item_id)

        # discrete relevance as user feedback
        discrete_relevance = self.sample_discrete_response(relevance)
        return relevance, discrete_relevance

    def item_satiation(self, item_id):
        relevance = self.shared.embeddings.item_cluster_relevance[item_id]
        # item-cluster relevance based weighted cluster satiation
        return np.sum(self.volatile.satiation * relevance)

    def ranked_items(self, discrete: bool, with_satiation: bool) -> np.ndarray:
        continuous_relevance, discrete_relevance = self.relevance(with_satiation=with_satiation)
        relevance = discrete_relevance if discrete else continuous_relevance

        ranked_items = np.argsort(relevance)[::-1]
        return ranked_items

    def static_relevance(self, item_id: int) -> float:
        return self.shared.embeddings.similarity(self.tastes, self.shared.embeddings.items[item_id])

    def increase_satiation(self, item_id: int) -> np.ndarray:
        # increase cluster satiation via item-to-cluster similarity and speed
        speed = self.static.satiation_speed
        satiation = self.volatile.satiation
        cluster_attention = self.shared.embeddings.item_cluster_relevance[item_id]

        np.clip(satiation * (1.0 + cluster_attention * speed), 1e-4, 1e+4, out=satiation)
        return cluster_attention

    def drift_satiation(self, speed: float = None):
        speed = isnone(speed, self.shared.satiation_drift)

        # continue based on previous mood, but slightly drift to another mood;
        # in average, it directs to the expected mean satiation
        _, mean = self.shared.satiation_distribution
        mood = normalize_mood(self.volatile.satiation, mean=mean)
        new_mood = self.sample_satiation()

        # calculate new satiation vector
        new_satiation = lin_sum(x=mood, lr=speed, y=new_mood)

        # but update only those parts that are object to satiation
        # to prevent specific scenarios stop working expected way
        update_mask = self.static.satiation_speed > 0
        self.volatile.satiation[update_mask] = new_satiation[update_mask]

    def drift_satisfaction(self, speed: float = None, acceleration: float = None):
        new_satisfaction = self.sample_initial_satisfaction()
        self.update_satisfaction(grade=new_satisfaction, speed=speed, acceleration=acceleration)

    def update_satisfaction(self, grade: float, speed: float = None, acceleration: float = None):
        speed = isnone(speed, self.shared.satisfaction_speed)
        if acceleration is not None:
            speed = 1 - (1 - speed) ** acceleration
        self.volatile.satisfaction = lin_sum(x=self.volatile.satisfaction, lr=speed, y=grade)

    def update_listening_trace(
            self, item_id: int = None, decay: float = None, acceleration: float = None
    ):
        decay = isnone(decay, self.shared.item_listening_trace_decay)
        if acceleration is not None:
            decay = decay ** acceleration
        update_exp_trace(traces=self.volatile.item_listening_trace, decay=decay, tr=item_id)

    def sample_satiation(self) -> np.ndarray:
        n_clusters = self.shared.embeddings.n_item_clusters
        min_satiation, mean = self.shared.satiation_distribution

        # we want E[satiation] = 1 => E[U[0, 2]] = 1
        # or more general E[satiation] = mean ==> E[U[0, 2*mean]] = mean
        # => shifting both sides towards `mean` keeps mean unchanged
        mood = self.rng.uniform(min_satiation, 2 * mean - min_satiation, size=n_clusters)
        return normalize_mood(mood, mean=mean)

    def sample_discrete_response(self, relevance) -> int:
        mark_distributions = self.shared.discrete_actions_distr

        if self.shared.deterministic_actions:
            marks = np.array([center for center, _ in mark_distributions])
        else:
            marks = np.array([self.rng.normal(*distr) for distr in mark_distributions])

        return 2 + np.argmin(np.abs(marks - relevance))

    def satiation_boosting(self, item_id: int) -> float:
        """Compute boosting penalty based on item-cluster satiation."""
        item_satiation = self.item_satiation(item_id)
        is_penalized = item_satiation > 1.0

        boosting_k = self.shared.relevance_boosting_k[is_penalized]
        boosting_softness = self.shared.boosting_softness[is_penalized]
        return boosting(item_satiation, k=boosting_k, softness=boosting_softness)

    def repeat_boosting(self, item_id: int) -> float:
        """Compute boosting penalty based on the track repeats record."""
        repeat_trace = 1.0 + self.volatile.item_listening_trace[item_id]
        return repeat_trace ** (-self.shared.item_repeat_penalty_power)

    def sample_initial_satisfaction(self):
        return self.rng.normal(self.shared.base_satisfaction, 0.25)

    @staticmethod
    def create_static_state(*, user_id: int, rng: Generator, shared_state: SharedUserState):
        return StaticUserState(
            user_id=user_id,
            satiation_speed=sample_satiation_speed(rng, shared_state),
        )

    @staticmethod
    def create_volatile_state(*, rng: Generator, shared_state: SharedUserState):
        return VolatileUserState(
            seed=sample_int(rng),
            satiation=np.ones(shared_state.embeddings.n_item_clusters),
            item_listening_trace=np.zeros(shared_state.embeddings.n_items),
            satisfaction=shared_state.base_satisfaction
        )

    @property
    def id(self):
        return self.static.user_id

    @property
    def tastes(self) -> np.ndarray:
        return self.shared.embeddings.users[self.id]

    @property
    def rng(self):
        return self.volatile.episodic_rng.rng


def sample_satiation_speed(rng: Generator, shared_state: SharedUserState) -> np.ndarray:
    n_clusters = shared_state.embeddings.n_item_clusters
    base_satiation_speed = shared_state.base_satiation_speed

    if isinstance(base_satiation_speed, float):
        return np.full(n_clusters, base_satiation_speed)

    # tuple[float, float]
    base_satiation_speed, k = base_satiation_speed
    k = 1.0 + k
    min_speed, max_speed = 1/k * base_satiation_speed, k * base_satiation_speed
    return np.clip(
        rng.uniform(min_speed, max_speed, n_clusters),
        0., 1.0
    )


def normalize_mood(satiation: np.ndarray, mean: float):
    # makes satiation.sum() equal to the number of clusters (=shape[0])
    return satiation * (mean / satiation.mean())


def parse_satiation_distribution(distribution) -> tuple[float, float]:
    if isinstance(distribution, float):
        min_value, mean_value = distribution, 1.0
        return min_value, mean_value
    return distribution
