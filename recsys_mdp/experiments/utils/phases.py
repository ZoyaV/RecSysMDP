from __future__ import annotations

from recsys_mdp.mdp.utils import isnone


class ExperimentPipeline:
    meta_epochs: int
    fix_generator: bool
    accumulate_data: bool
    retrain: bool

    def __init__(
            self, meta_epochs: int, fix_generator: bool,
            accumulate_data: bool, retrain: bool
    ):
        self.meta_epochs = meta_epochs
        self.fix_generator = fix_generator
        self.accumulate_data = accumulate_data
        self.retrain = retrain


class GenerationPhaseParameters:
    episodes: int | None
    samples: int | None
    use_cache: bool

    def __init__(
            self,
            episodes: int | None = None,
            samples: int | None = None,
            use_cache: bool = False,
    ):
        assert episodes is not None or samples is not None
        self.episodes = isnone(episodes, 1_000_000_000)
        self.samples = isnone(samples, 1_000_000_000)
        self.use_cache = use_cache


class LearningPhaseParameters:
    epochs: int

    def __init__(self, epochs: int):
        self.epochs = epochs


class EvaluationPhaseParameters:
    schedule: int
    episodes: int
    user_episodes: int
    track_users: list[int]

    def __init__(
            self, schedule: int = 1, episodes: int = 1, user_episodes: int = 1,
            track_users: list[int] = None, reinitialize: bool = False,
    ):
        self.schedule = schedule
        self.episodes = episodes
        self.user_episodes = user_episodes
        self.track_users = isnone(track_users, [])
        self.reinitialize = reinitialize
