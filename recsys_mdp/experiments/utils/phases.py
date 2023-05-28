from __future__ import annotations


class GenerationPhaseParameters:
    epochs: int

    episodes_per_epoch: int | None
    samples_per_epoch: int | None
    use_cache: bool

    def __init__(
            self,
            epochs: int,
            episodes_per_epoch: int | None = None,
            samples_per_epoch: int | None = None,
            use_cache: bool = False,
    ):
        self.epochs = epochs
        assert episodes_per_epoch is not None or samples_per_epoch is not None
        self.episodes_per_epoch = episodes_per_epoch
        self.samples_per_epoch = samples_per_epoch
        self.use_cache = use_cache


class LearningPhaseParameters:
    epochs: int
    eval_schedule: int
    eval_episodes: int

    def __init__(self, epochs: int, eval_schedule: int = 1, eval_episodes: int = 1):
        self.epochs = epochs
        self.eval_schedule = eval_schedule
        self.eval_episodes = eval_episodes
