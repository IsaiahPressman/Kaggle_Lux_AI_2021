from abc import ABC, abstractmethod
import copy
import logging
import numpy as np
from scipy.stats import rankdata
from typing import Dict, NamedTuple, NoReturn, Tuple

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import Player


def count_city_tiles(game_state: Game) -> np.ndarray:
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


def should_early_stop(game_state: Game) -> bool:
    ct_count = count_city_tiles(game_state)
    unit_count = count_units(game_state)
    ct_pct = ct_count / max(ct_count.sum(), 1)
    unit_pct = unit_count / max(unit_count.sum(), 1)
    return ((ct_count == 0).any() or
            (unit_count == 0).any() or
            (ct_pct >= 0.75).any() or
            (unit_pct >= 0.75).any())


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
    def __init__(self, **kwargs):
        if kwargs:
            logging.warning(f"RewardSpace received unexpected kwargs: {kwargs}")

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}


# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
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

    def __init__(self, early_stop: bool = False, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)
        self.early_stop = early_stop

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
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

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        return tuple(count_city_tiles(game_state) / 1024.)


class StatefulMultiReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.,
            early_stop: bool = False,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.early_stop = early_stop

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
            # A reward given each step
            "step": 0.,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(StatefulMultiReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        reward_items_dict = {
            "city": new_city_count - self.city_count,
            "unit": new_unit_count - self.unit_count,
            "research": new_research_points - self.research_points,
            # Don't penalize losing fuel at night
            "fuel": np.maximum(new_total_fuel - self.total_fuel, 0),
            "full_workers": np.array([
                sum(unit.get_cargo_space_left() > 0 for unit in player.units if unit.is_worker())
                for player in game_state.players
            ]),
            "step": np.ones(2, dtype=float)
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

        return tuple(reward / 500. / max(self.positive_weight, self.negative_weight))

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

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        reward = np.array(super(ZeroSumStatefulMultiReward, self).compute_rewards(game_state, done))
        return tuple(reward - reward.mean())


class PunishingExponentialReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            reward_max=1. / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            **kwargs
    ):
        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)
        self.research_points = np.empty_like(self.city_count)
        self.total_fuel = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 0.,
            "city": 1.,
            "unit": 0.5,
            "research": 0.01,
            "fuel": 0.001,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(PunishingExponentialReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        new_research_points = count_research_points(game_state)
        new_total_fuel = count_total_fuel(game_state)

        city_diff = new_city_count - self.city_count
        unit_diff = new_unit_count - self.unit_count
        reward_items_dict = {
            "city": new_city_count,
            "unit": new_unit_count,
            "research": new_research_points,
            "fuel": new_total_fuel,
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
            [reward_items_dict[key] * w for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)

        lost_unit_or_city = (city_diff < 0) | (unit_diff < 0)
        reward = np.where(
            lost_unit_or_city,
            -0.1,
            reward / 1_000.
        )

        return tuple(reward), done or lost_unit_or_city.any()

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        raise NotImplementedError

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)
        self.research_points = np.zeros_like(self.research_points)
        self.total_fuel = np.zeros_like(self.total_fuel)


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

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        goal_reached = self.completed_task(game_state)
        return tuple(goal_reached.astype(float)), goal_reached.any() or done

    @abstractmethod
    def completed_task(self, game_state: Game) -> np.ndarray:
        pass

    def get_subtask_encoding(self, subtask_encoding: dict) -> int:
        return subtask_encoding[type(self)]


class CollectNWood(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"], **kwargs):
        super(CollectNWood, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.wood for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNCoal(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 2, **kwargs):
        super(CollectNCoal, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.coal for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class CollectNUranium(Subtask):
    def __init__(self, n: int = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // 5, **kwargs):
        super(CollectNUranium, self).__init__(**kwargs)
        self.n = n

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            sum([unit.cargo.uranium for unit in player.units])
            for player in game_state.players
        ]) >= self.n


class MakeNCityTiles(Subtask):
    def __init__(self, n_city_tiles: int = 2, **kwargs):
        super(MakeNCityTiles, self).__init__(**kwargs)
        assert n_city_tiles > 1, "Players start with 1 city tile already"
        self.n_city_tiles = n_city_tiles

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_city_tiles(game_state) >= self.n_city_tiles


class MakeNContiguousCityTiles(MakeNCityTiles):
    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([
            # Extra -1 is included to avoid taking max of empty sequence
            max([len(city.citytiles) for city in player.cities.values()] + [0])
            for player in game_state.players
        ]) >= self.n_city_tiles


class CollectNTotalFuel(Subtask):
    def __init__(self, n_total_fuel: int = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"] *
                                           GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"], **kwargs):
        super(CollectNTotalFuel, self).__init__(**kwargs)
        self.n_total_fuel = n_total_fuel

    def completed_task(self, game_state: Game) -> np.ndarray:
        return count_total_fuel(game_state) >= self.n_total_fuel


class SurviveNNights(Subtask):
    def __init__(self, n_nights: int = 1, **kwargs):
        super(SurviveNNights, self).__init__(**kwargs)
        cycle_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        self.target_step = n_nights * cycle_len
        assert self.target_step <= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        self.city_count = np.empty((2,), dtype=int)
        self.unit_count = np.empty_like(self.city_count)

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
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
        new_city_count = count_city_tiles(game_state)
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
    def __init__(
            self,
            n_research_points: int = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"],
            **kwargs
    ):
        super(GetNResearchPoints, self).__init__(**kwargs)
        self.n_research_points = n_research_points

    def completed_task(self, game_state: Game) -> np.ndarray:
        return np.array([player.research_points for player in game_state.players]) >= self.n_research_points
