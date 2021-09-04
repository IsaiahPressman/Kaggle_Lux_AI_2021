from abc import ABC, abstractmethod
import numpy as np
import random
from typing import Callable, Dict, Optional, Tuple, Sequence

from .reward_spaces import Subtask
from ..lux.game import Game


class SubtaskSampler(ABC):
    def __init__(self, subtask_constructors: Sequence[Callable[..., Subtask]]):
        self.subtask_constructors = subtask_constructors

    @abstractmethod
    def sample(self, final_rewards: Optional[Tuple[float, float]]) -> Subtask:
        pass

    # noinspection PyMethodMayBeStatic
    def get_info(self) -> Dict[str, np.ndarray]:
        return {}


class RandomSampler(SubtaskSampler):
    def sample(self, final_rewards: Optional[Tuple[float, float]]) -> Subtask:
        return self.subtask_constructors[random.randrange(len(self.subtask_constructors))]()


class DifficultySampler(SubtaskSampler):
    def __init__(self, subtask_constructors: Sequence[Callable[..., Subtask]]):
        super(DifficultySampler, self).__init__(subtask_constructors)
        self.active_subtask_idx = -1
        self.summed_rewards = np.zeros(len(self.subtask_constructors))
        self.n_trials = np.zeros(len(self.subtask_constructors))

    def sample(self, final_rewards: Optional[Tuple[float, float]]) -> Subtask:
        if final_rewards is not None:
            self.n_trials[self.active_subtask_idx] += 1
            self.summed_rewards[self.active_subtask_idx] += np.mean(final_rewards)

        self.active_subtask_idx = np.random.choice(len(self.subtask_constructors), p=self.weights)
        return self.subtask_constructors[self.active_subtask_idx]()

    @property
    def weights(self) -> np.ndarray:
        weights = Subtask.get_reward_spec().reward_max - self.summed_rewards / np.maximum(self.n_trials, 1)
        return weights / weights.sum()

    def get_info(self) -> Dict[str, np.ndarray]:
        return {
            f"LOGGING_{subtask.__name__}_subtask_difficulty": self.weights[i]
            for i, subtask in enumerate(self.subtask_constructors)
        }


class MultiSubtask(Subtask):
    def __init__(
            self,
            subtask_constructors: Sequence[Callable[..., Subtask]] = (),
            subtask_sampler_constructor: Callable[..., SubtaskSampler] = RandomSampler,
            **kwargs
    ):
        super(MultiSubtask, self).__init__(**kwargs)
        self.subtask_constructors = subtask_constructors
        self.subtask_sampler = subtask_sampler_constructor(self.subtask_constructors)
        self.active_subtask = self.subtask_sampler.sample(None)
        self.info = {
            f"LOGGING_{subtask.__name__}_subtask_reward": np.array([float("nan"), float("nan")])
            for subtask in self.subtask_constructors
        }

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        reward, done = self.active_subtask.compute_rewards_and_done(game_state, done)
        for subtask in self.subtask_constructors:
            reward_key = f"LOGGING_{subtask.__name__}_subtask_reward"
            if isinstance(self.active_subtask, subtask):
                self.info[reward_key] = np.array(reward)
            else:
                self.info[reward_key] = np.array([float("nan"), float("nan")])
        if done:
            self.active_subtask = self.subtask_sampler.sample(reward)
        return reward, done

    def completed_task(self, game_state: Game) -> np.ndarray:
        raise NotImplementedError

    def get_info(self) -> Dict[str, np.ndarray]:
        return dict(**self.info, **self.subtask_sampler.get_info())

    def get_subtask_encoding(self, subtask_encoding_dict: dict) -> int:
        return self.active_subtask.get_subtask_encoding(subtask_encoding_dict)
