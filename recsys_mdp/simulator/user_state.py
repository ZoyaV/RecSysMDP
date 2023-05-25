from __future__ import annotations

import numpy as np
from numpy.random import Generator

from recsys_mdp.simulator.embeddings import Embeddings
from recsys_mdp.simulator.relevance import similarity
from recsys_mdp.simulator.utils import softmax, normalize, boosting
from recsys_mdp.utils.base import sample_rng, sample_int, update_exp_trace, lin_sum

USER_RESET_MODE_CONTINUE = 'continue'
USER_RESET_MODE_INIT = 'init'
USER_RESET_MODE_DISCONTINUE = 'discontinue'


class UserState:
    rng: Generator
    user_id: int

    base_satiation: float
    # all projected onto clusters
    # volatile, it also correlates to user's mood
    satiation: np.ndarray
    # changes with reset
    satiation_speed: np.ndarray
    satiation_drift: float
    init_mood_seed: int

    item_listening_trace_decay: float
    item_listening_trace: np.ndarray
    item_repeat_penalty_power: float

    satisfaction: float
    early_stop_delta: float
    initial_satisfaction: float = 4

    relevance_boosting_k: tuple[float, float]
    metric: str
    embeddings: Embeddings

    def __init__(
            self, user_id: int, embeddings: Embeddings,
            base_satiation: float,
            base_satiation_speed: float | tuple[float, float],
            satiation_drift: float,
            item_listening_trace_decay: float,
            item_repeat_penalty_power: float,
            early_stop_delta: float,
            similarity_metric: str,
            item_to_cluster_classification: str,
            relevance_boosting: tuple[float, float],
            boosting_softness: tuple[float, float],
            discrete_actions: list[tuple[float, float]],
            rng: Generator
    ):
        self.user_id = user_id
        self.rng = sample_rng(rng)

        self.base_satiation = base_satiation
        self.satiation_drift = satiation_drift
        self.item_listening_trace_decay = item_listening_trace_decay
        self.item_repeat_penalty_power = item_repeat_penalty_power
        self.early_stop_delta = early_stop_delta
        self.relevance_boosting_k = tuple(relevance_boosting)
        self.boosting_softness = tuple(boosting_softness)
        self.discrete_actions_distr = discrete_actions
        self.metric = similarity_metric
        self.embeddings = embeddings

        # how to normalize item-to-item_clusters similarity to attention/probability
        self.item_to_cluster_classification = dict(
            softmax=softmax,
            normalize=normalize
        )[item_to_cluster_classification]

        self.init_mood_seed = sample_int(self.rng)
        self.satiation = self.sample_satiation(self.init_mood_seed)
        self.satiation_speed = self.sample_satiation_speed(base_satiation_speed)
        self.item_listening_trace = np.zeros(self.embeddings.n_items)
        self.satisfaction = self.sample_initial_satisfaction()

    @property
    def tastes(self) -> np.ndarray:
        return self.embeddings.users[self.user_id]

    def reset(self, mode: str = USER_RESET_MODE_CONTINUE):
        self.satisfaction = self.sample_initial_satisfaction()
        self.item_listening_trace *= self.item_listening_trace_decay**7

        if mode == USER_RESET_MODE_CONTINUE:
            self.drift_satiation()
        elif mode == USER_RESET_MODE_INIT:
            # return to the initial mood
            self.satiation = self.sample_satiation(self.init_mood_seed)
            self.item_listening_trace[:] = 0
        elif mode == USER_RESET_MODE_DISCONTINUE:
            # re-sample new mood
            self.satiation = self.sample_satiation(sample_int(self.rng))
        else:
            raise ValueError(f'User reset mode "{mode}" does not supported.')

    def step(self, item_id: int):
        # 1) update item listening trace
        update_exp_trace(
            traces=self.item_listening_trace, tr=item_id, decay=self.item_listening_trace_decay
        )

        # 2) increase satiation via similarity and speed
        item_to_cluster_relevance = self.item_to_cluster_relevance(item_id)
        self.satiation *= 1.0 + item_to_cluster_relevance * self.satiation_speed
        np.clip(self.satiation, 1e-4, 1e+4, out=self.satiation)

        # 3) calculate relevance
        continuous_relevance, discrete_relevance = self.relevance(
            item_id, item_to_cluster_relevance=item_to_cluster_relevance
        )

        # 4) update current satisfaction
        self.satisfaction = lin_sum(self.satisfaction, .9, discrete_relevance)
        return continuous_relevance, discrete_relevance

    def sample_stop_listening(self):
        min_prob = 0.001
        k = self.early_stop_delta
        dissatisfaction = 5 - self.satisfaction

        # early stop increases with increasing speed based on dissatisfaction
        probability = min_prob + k * dissatisfaction * (dissatisfaction + 1) / 2
        return self.rng.random() < probability

    def sample_satiation(self, seed: int) -> np.ndarray:
        n_clusters = self.embeddings.n_item_clusters
        rng = np.random.default_rng(seed)
        return self.base_satiation + rng.uniform(size=n_clusters)

    def drift_satiation(self):
        # continue based on previous mood, but slightly drift to another mood;
        # in average, it directs to the expected mean satiation
        mood = self.satiation
        new_mood = self.sample_satiation(sample_int(self.rng))
        # calculate new satiation vector
        new_satiation = lin_sum(x=mood, lr=self.satiation_drift, y=new_mood)

        # but update only those parts that are object to satiation
        # to prevent specific scenarios stop working expected way
        update_mask = self.satiation_speed > 0
        self.satiation[update_mask] = new_satiation[update_mask]

    def relevance(
            self, item_id: int = None, with_satiation: bool = True,
            item_to_cluster_relevance: np.ndarray = None
    ):
        if item_id is None:
            # compute relevance for all items without consuming
            all_relevance = [
                self.relevance(item_id, with_satiation=with_satiation)
                for item_id in range(self.embeddings.n_items)
            ]
            continuous_relevance = np.array([cont for cont, discr in all_relevance])
            discrete_relevance = np.array([discr for cont, discr in all_relevance])
            return continuous_relevance, discrete_relevance

        # get item similarity to user preferences
        relevance = similarity(self.tastes, self.embeddings.items[item_id], metric=self.metric)

        if with_satiation:
            if item_to_cluster_relevance is None:
                item_to_cluster_relevance = self.item_to_cluster_relevance(item_id)
            # compute and apply relevance boosting based on item-to-clusters satiation
            relevance *= self.relevance_boosting(item_to_cluster_relevance)
            # compute and apply boosting penalty based on repeating tracks
            relevance *= self.repeat_boosting(item_id)

        # discrete relevance as user feedback
        discrete_relevance = self.sample_discrete_response(relevance)
        return relevance, discrete_relevance

    def ranked_items(self, discrete: bool, with_satiation: bool) -> np.ndarray:
        continuous_relevance, discrete_relevance = self.relevance(with_satiation=with_satiation)
        relevance = discrete_relevance if discrete else continuous_relevance

        ranked_items = np.argsort(relevance)[::-1]
        return ranked_items

    def sample_discrete_response(self, relevance) -> int:
        marks = np.array([self.rng.normal(*distr) for distr in self.discrete_actions_distr])
        return 2 + np.argmin(np.abs(marks - relevance))

    def relevance_boosting(self, item_to_cluster_relevance: np.ndarray = None) -> float:
        # compute boosting from the aggregate weighted cluster satiation
        aggregate_item_satiation = np.sum(self.satiation * item_to_cluster_relevance)

        boosting_k = self.relevance_boosting_k[aggregate_item_satiation > 1.0]
        boosting_softness = self.boosting_softness[aggregate_item_satiation > 1.0]
        return boosting(aggregate_item_satiation, k=boosting_k, softness=boosting_softness)

    def repeat_boosting(self, item_id: int) -> float:
        repeat = self.item_listening_trace[item_id]
        if repeat < 2:
            return 1.
        return (repeat - 1) ** (-self.item_repeat_penalty_power)

    def sample_satiation_speed(self, base_satiation_speed) -> np.ndarray:
        n_clusters = self.embeddings.n_item_clusters
        if isinstance(base_satiation_speed, float):
            return np.full(n_clusters, base_satiation_speed)

        # tuple[float, float]
        base_satiation_speed, k = base_satiation_speed
        k = 1.0 + k
        min_speed, max_speed = 1/k * base_satiation_speed, k * base_satiation_speed
        return np.clip(
            self.rng.uniform(min_speed, max_speed, n_clusters),
            0., 1.0
        )

    def item_to_cluster_relevance(self, item_id: int) -> np.ndarray:
        item_to_cluster_relevance = similarity(
            self.embeddings.items[item_id], self.embeddings.item_clusters, metric=self.metric
        )
        # normalize relevance to [0, 1]
        return self.item_to_cluster_classification(item_to_cluster_relevance)

    def sample_initial_satisfaction(self):
        return self.rng.normal(self.initial_satisfaction, 0.15)
