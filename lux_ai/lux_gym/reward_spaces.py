from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import rankdata
from typing import *

from ..lux.game import Game
from ..lux.game_objects import Player


class RewardSpec(NamedTuple):
    reward_min: float
    reward_max: float
    zero_sum: bool


class BaseRewardSpace(ABC):
    @abstractmethod
    def get_reward_spec(self) -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> list[float, float]:
        pass


class GameResultReward(BaseRewardSpace):
    def get_reward_spec(self) -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True
        )

    def compute_rewards(self, game_state: Game, done: bool) -> list[float, float]:
        if not done:
            return [0., 0.]

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward._compute_player_reward(p)) for p in game_state.players]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return list(rewards)

    @staticmethod
    def _compute_player_reward(player: Player):
        ct_count = player.city_tile_count
        unit_count = len(player.units)
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        return ct_count * 10000 + unit_count


class CityTileReward(BaseRewardSpace):
    def get_reward_spec(self) -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True
        )

    def compute_rewards(self, game_state: Game, done: bool) -> list[float]:
        ct_count = np.array([player.city_tile_count for player in game_state.players])
        ct_count_zero_sum = ct_count - ct_count.mean()
        return list(ct_count_zero_sum / 1024.)
