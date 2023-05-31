from __future__ import annotations

from recsys_mdp.mdp.utils import isnone


class GenerationPhaseParameters:
    epochs: int

    episodes_per_epoch: int | None
    samples_per_epoch: int | None
    use_cache: bool
    switch_to_eval_model: bool

    def __init__(
            self,
            epochs: int,
            episodes_per_epoch: int | None = None,
            samples_per_epoch: int | None = None,
            use_cache: bool = False,
            switch_to_eval_model: bool = True,
    ):
        self.epochs = epochs
        assert episodes_per_epoch is not None or samples_per_epoch is not None
        self.episodes_per_epoch = episodes_per_epoch
        self.samples_per_epoch = samples_per_epoch
        self.use_cache = use_cache
        self.switch_to_eval_model = switch_to_eval_model


class LearningPhaseParameters:
    epochs: int
    eval_schedule: int
    eval_episodes: int
    eval_episodes_all: int
    reinitialize: bool

    eval_users: list[int]

    def __init__(
            self, epochs: int, eval_schedule: int = 1,
            eval_episodes: int = 1, eval_episodes_all: int = None,
            eval_users: list[int] = None,
            reinitialize: bool = False,
    ):
        self.epochs = epochs
        self.eval_schedule = eval_schedule
        self.eval_episodes = eval_episodes
        self.eval_episodes_all = isnone(eval_episodes_all, self.eval_episodes)
        self.eval_users = isnone(eval_users, [])
        self.reinitialize = reinitialize
