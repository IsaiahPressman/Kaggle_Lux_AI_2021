from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Callable, Optional, Sequence

from .reward_spaces import Subtask
from ..lux.game import Game


class SubtaskSampler(ABC):
    def __init__(self, subtask_constructors: Sequence[Callable[..., Subtask]]):
        self.subtask_constructors = subtask_constructors

    @abstractmethod
    def sample(self, final_rewards: Optional[tuple[float, float]]) -> Subtask:
        pass


class RandomSampler(SubtaskSampler):
    def sample(self, final_rewards: Optional[tuple[float, float]]) -> Subtask:
        return self.subtask_constructors[random.randint(0, len(self.subtask_constructors))]()


class MultiSubtask(Subtask):
    def __init__(
            self,
            subtask_constructors: Sequence[Callable[..., Subtask]] = (),
            subtask_sampler_constructor: Callable[..., SubtaskSampler] = RandomSampler,
    ):
        self.subtask_constructors = subtask_constructors
        self.subtask_sampler = subtask_sampler_constructor(self.subtask_constructors)
        self.active_subtask = self.subtask_sampler.sample(None)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> tuple[tuple[float, float], bool]:
        reward, done = self.active_subtask.compute_rewards_and_done(game_state, done)
        if done:
            self.active_subtask = self.subtask_sampler.sample(reward)
        return reward, done

    def completed_task(self, game_state: Game) -> np.ndarray:
        raise NotImplementedError

    def update_info(self, info: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        updated_info = {}
        for key, val in info.items():
            if key.startswith("LOGGING_"):
                updated_info[f"LOGGING_{type(self.active_subtask).__name__}_{key[8:]}"] = val
            updated_info[key] = val
        return updated_info

    def get_subtask_encoding(self, subtask_encoding_dict: dict) -> int:
        return self.active_subtask.get_subtask_encoding(subtask_encoding_dict)
