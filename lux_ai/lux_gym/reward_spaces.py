from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import rankdata
from typing import *

from ..lux.game import Game
from ..lux.game_objects import Player


def count_cities(game_state: Game) -> np.ndarray:
    return np.array([player.city_tile_count for player in game_state.players])


def count_units(game_state: Game) -> np.ndarray:
    return np.array([len(player.units) for player in game_state.players])


def count_total_fuel(game_state: Game) -> np.ndarray:
    return np.array([
        sum([city.fuel for city in player.cities.values()])
        for player in game_state.players
    ])


def count_research_points(game_state: Game) -> np.ndarray:
    return np.array([player.research_points for player in game_state.players])


class RewardSpec(NamedTuple):
    reward_min: Optional[float]
    reward_max: Optional[float]
    zero_sum: bool


class BaseRewardSpace(ABC):
    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> list[float, float]:
        pass


class GameResultReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True
        )

    def compute_rewards(self, game_state: Game, done: bool) -> list[float, float]:
        if not done:
            return [0., 0.]

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return list(rewards)

    @staticmethod
    def compute_player_reward(player: Player):
        ct_count = player.city_tile_count
        unit_count = len(player.units)
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        return ct_count * 10000 + unit_count


class CityTileReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> list[float]:
        return list(count_cities(game_state) / 1024.)


class StatefulMultiReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False
        )

    def __init__(self):
        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 10.,
            "city": 1.,
            "unit": 1.,
            "research": 0.2,
            "fuel": 0.01
        }

    def compute_rewards(self, game_state: Game, done: bool) -> list[float, float]:
        new_city_count = count_cities(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)
        city_diff = new_city_count - self.city_count
        unit_diff = new_unit_count - self.unit_count
        research_diff = new_research_points - self.research_points
        fuel_diff = new_total_fuel - self.total_fuel

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
            self.research_points = new_research_points
            self.total_fuel = new_total_fuel

        reward = (
                game_result_reward * self.weights["game_result"] +
                city_diff * self.weights["city"] +
                unit_diff * self.weights["unit"] +
                research_diff * self.weights["research"] +
                fuel_diff * self.weights["fuel"]
        ) / 100.
        return list(reward)

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)


class ZeroSumStatefulMultiReward(StatefulMultiReward):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True
        )

    def compute_rewards(self, game_state: Game, done: bool) -> list[float, float]:
        reward = np.array(super(ZeroSumStatefulMultiReward, self).compute_rewards(game_state, done))
        return list(reward - reward.mean())
