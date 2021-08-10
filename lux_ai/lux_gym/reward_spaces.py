from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import rankdata
from typing import *

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
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
    reward_min: float
    reward_max: float
    zero_sum: bool
    only_once: bool


# All reward spaces defined below

class BaseRewardSpace(ABC):
    """
    A class used for defining a reward space and/or done state for either the full game or a sub-task
    """

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> tuple[tuple[float, float], bool]:
        pass

    def get_info(self) -> dict[str, np.ndarray]:
        return {}


# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> tuple[tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> tuple[float, float]:
        pass


class GameResultReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=True
        )

    def compute_rewards(self, game_state: Game, done: bool) -> tuple[float, float]:
        if not done:
            return 0., 0.

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return tuple(rewards)

    @staticmethod
    def compute_player_reward(player: Player):
        ct_count = player.city_tile_count
        unit_count = len(player.units)
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        return ct_count * 10000 + unit_count


class CityTileReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> tuple[float, float]:
        return tuple(count_cities(game_state) / 1024.)


class StatefulMultiReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(self, positive_weight: float = 1., negative_weight: float = 1.):
        assert positive_weight > 0.
        assert negative_weight > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 10.,
            "city": 1.,
            "unit": 0.5,
            "research": 0.1,
            "fuel": 0.005,
            # Penalize workers each step that their cargo remains full
            # "full_workers": -0.01,
            "full_workers": 0.,
        }
        self._reset()

    def compute_rewards(self, game_state: Game, done: bool) -> tuple[float, float]:
        new_city_count = count_cities(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        reward_items_dict = {
            "city": new_city_count - self.city_count,
            "unit": new_unit_count - self.unit_count,
            "research": new_research_points - self.research_points,
            "fuel": new_total_fuel - self.total_fuel,
            "full_workers": np.array([
                sum(unit.get_cargo_space_left() > 0 for unit in player.units if unit.is_worker())
                for player in game_state.players
            ]),
        }

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
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [self.weight_rewards(reward_items_dict[key] * w) for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        return tuple(reward / 100. / max(self.positive_weight, self.negative_weight))

    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(
            reward > 0.,
            self.positive_weight * reward,
            reward
        )
        reward = np.where(
            reward < 0.,
            self.negative_weight * reward,
            reward
        )
        return reward

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
            zero_sum=True,
            only_once=False
        )

    def compute_rewards(self, game_state: Game, done: bool) -> tuple[float, float]:
        reward = np.array(super(ZeroSumStatefulMultiReward, self).compute_rewards_and_done(game_state, done))
        return tuple(reward - reward.mean())


# Subtask reward spaces defined below
# NB: Subtasks that are "different enough" should be defined separately since each subtask gets its own embedding
# See obs_spaces.SUBTASK_ENCODING

# TODO: Somehow include target locations for subtasks?
class Subtask(BaseRewardSpace, ABC):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        """
        Don't override reward_spec or you risk breaking classes like multi_subtask.MultiSubtask
        """
        return RewardSpec(
            reward_min=0.,
            reward_max=1.,
            zero_sum=False,
            only_once=True
        )

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> tuple[tuple[float, float], bool]:
        goal_reached = self.completed_task(game_state)
        return tuple(goal_reached.astype(float)), goal_reached.any() or done

    @abstractmethod
    def completed_task(self, game_state: Game) -> np.ndarray:
        pass

    def get_subtask_encoding(self, subtask_encoding: dict) -> int:
        return subtask_encoding[type(self)]


class CollectNWood(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]):
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.wood for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNCoal(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 2):
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.coal for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNUranium(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 5):
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.uranium for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class MakeNCityTiles(Subtask):
    def __init__(self, n_city_tiles: int = 2):
        assert n_city_tiles > 1, "Players start with 1 city tile already"
        self.n_city_tiles = n_city_tiles

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_cities(game_state) >= self.n_city_tiles


class MakeNContiguousCityTiles(MakeNCityTiles):
    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            # Extra -1 is included to avoid taking max of empty sequence
            max([len(city.citytiles) for city in player.cities.values()] + [0])
            for player in game_state.players
        ]) >= self.n_city_tiles


class CollectNTotalFuel(Subtask):
    def __init__(self, n_total_fuel: int = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"] *
                                           GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]):
        self.n_total_fuel = n_total_fuel

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_total_fuel(game_state) >= self.n_total_fuel


class SurviveNNights(Subtask):
    def __init__(self, n_nights: int = 1):
        cycle_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        self.target_step = n_nights * cycle_len
        assert self.target_step <= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        self.city_count = np.empty((2,), dtype=int)
        self.unit_count = np.empty_like(self.city_count)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> tuple[tuple[float, float], bool]:
        failed_task = self.failed_task(game_state)
        completed_task = self.completed_task(game_state)
        if failed_task.any():
            rewards = np.where(
                failed_task,
                0.,
                0.5 + 0.5 * completed_task.astype(float)
            )
        else:
            rewards = completed_task.astype(float)
        done = failed_task.any() or completed_task.any() or done
        if done:
            self._reset()
        return tuple(rewards), done

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            game_state.turn >= self.target_step
        ]).repeat(2)

    def failed_task(self, game_state: Game) -> np.ndarray:
        new_city_count = count_cities(game_state)
        new_unit_count = count_units(game_state)

        failed = np.logical_or(
            new_city_count < self.city_count,
            new_unit_count < self.unit_count
        )
        self.city_count = new_city_count
        self.unit_count = new_unit_count
        return failed

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)


class GetNResearchPoints(Subtask):
    def __init__(self, n_research_points: int = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]):
        self.n_research_points = n_research_points

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([player.research_points for player in game_state.players]) >= self.n_research_points
