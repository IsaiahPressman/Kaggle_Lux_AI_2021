import functools
import logging
from abc import ABC, abstractmethod
import gym
import itertools
import numpy as np
from typing import Dict, List, Tuple

from ..utility_constants import DN_CYCLE_LEN, MAX_RESOURCE, MAP_SIZES, MAX_BOARD_SIZE
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS

WOOD = Constants.RESOURCE_TYPES.WOOD
COAL = Constants.RESOURCE_TYPES.COAL
URANIUM = Constants.RESOURCE_TYPES.URANIUM
MAX_FUEL = 30 * 10 * 9
# Player count
P = 2


class BaseObsSpace(ABC):
    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def wrap_env(self, env) -> gym.Wrapper:
        pass


class FixedShapeObs(BaseObsSpace, ABC):
    pass


class MultiObs(BaseObsSpace):
    def __init__(self, named_obs_spaces: Dict[str, BaseObsSpace], *args, **kwargs):
        super(MultiObs, self).__init__(*args, **kwargs)
        self.named_obs_spaces = named_obs_spaces

    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            name + key: val
            for name, obs_space in self.named_obs_spaces.items()
            for key, val in obs_space.get_obs_spec(board_dims).spaces.items()
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _MultiObsWrapper(env, self.named_obs_spaces)


class _MultiObsWrapper(gym.Wrapper):
    def __init__(self, env, named_obs_spaces: Dict[str, BaseObsSpace]):
        super(_MultiObsWrapper, self).__init__(env)
        self.named_obs_space_wrappers = {key: val.wrap_env(env) for key, val in named_obs_spaces.items()}

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        return {
            name + key: val
            for name, obs_space in self.named_obs_space_wrappers.items()
            for key, val in obs_space.observation(observation).items()
        }


class FixedShapeContinuousObs(FixedShapeObs):
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            # Player specific observations
            # none, worker
            "worker": gym.spaces.MultiBinary((1, P, x, y)),
            # none, cart
            "cart": gym.spaces.MultiBinary((1, P, x, y)),
            # Number of units in the square (only relevant on city tiles)
            "worker_COUNT": gym.spaces.Box(0., float("inf"), shape=(1, P, x, y)),
            "cart_COUNT": gym.spaces.Box(0., float("inf"), shape=(1, P, x, y)),
            # NB: cooldowns and cargo are always zero when on city tiles, so one layer will do for
            # the entire map
            # Normalized from 0-3
            "worker_cooldown": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-5
            "cart_cooldown": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-100
            f"worker_cargo_{WOOD}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"worker_cargo_{COAL}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"worker_cargo_{URANIUM}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-2000
            f"cart_cargo_{WOOD}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"cart_cargo_{COAL}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"cart_cargo_{URANIUM}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Whether the worker/cart is full
            "worker_cargo_full": gym.spaces.MultiBinary((1, P, x, y)),
            "cart_cargo_full": gym.spaces.MultiBinary((1, P, x, y)),
            # none, city_tile
            "city_tile": gym.spaces.MultiBinary((1, P, x, y)),
            # Normalized from 0-MAX_FUEL
            "city_tile_fuel": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-CITY_LIGHT_UPKEEP
            "city_tile_cost": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-9
            "city_tile_cooldown": gym.spaces.Box(0., 1., shape=(1, P, x, y)),

            # Player-agnostic observations
            # Normalized from 0-6
            "road_level": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            # Resources
            f"{WOOD}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            f"{COAL}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            f"{URANIUM}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),

            # Non-spatial observations
            # Normalized from 0-200
            "research_points": gym.spaces.Box(0., 1., shape=(1, P)),
            # coal is researched
            "researched_coal": gym.spaces.MultiBinary((1, P)),
            # uranium is researched
            "researched_uranium": gym.spaces.MultiBinary((1, P)),
            # True when it is night
            "night": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + 2),
            # The turn number % 40
            "day_night_cycle": gym.spaces.Box(0., 1., shape=(1, 1)),
            # The turn number // 40
            "phase": gym.spaces.MultiDiscrete(
                np.zeros((1, 1)) + GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN
            ),
            # The turn number, normalized from 0-360
            "turn": gym.spaces.Box(0., 1., shape=(1, 1)),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _FixedShapeContinuousObsWrapper(env)


class _FixedShapeContinuousObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(_FixedShapeContinuousObsWrapper, self).__init__(env)
        self._empty_obs = {}
        for key, spec in FixedShapeContinuousObs().get_obs_spec().spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        w_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        ca_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
        w_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.
        ca_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.
        ci_light = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
        ci_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
        max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

        obs = {
            key: val.copy() if val.ndim == 2 else val[:, :, :observation.map_width, :observation.map_height].copy()
            for key, val in self._empty_obs.items()
        }

        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                if unit.is_worker():
                    obs["worker"][0, p_id, x, y] = 1
                    obs["worker_COUNT"][0, p_id, x, y] += 1
                    obs["worker_cooldown"][0, p_id, x, y] = unit.cooldown / w_cooldown
                    obs["worker_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    obs[f"worker_cargo_{WOOD}"][0, p_id, x, y] = unit.cargo.wood / w_capacity
                    obs[f"worker_cargo_{COAL}"][0, p_id, x, y] = unit.cargo.coal / w_capacity
                    obs[f"worker_cargo_{URANIUM}"][0, p_id, x, y] = unit.cargo.uranium / w_capacity
                elif unit.is_cart():
                    obs["cart"][0, p_id, x, y] = 1
                    obs["cart_COUNT"][0, p_id, x, y] += 1
                    obs["cart_cooldown"][0, p_id, x, y] = unit.cooldown / ca_cooldown
                    obs["cart_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    obs[f"cart_cargo_{WOOD}"][0, p_id, x, y] = unit.cargo.wood / ca_capacity
                    obs[f"cart_cargo_{COAL}"][0, p_id, x, y] = unit.cargo.coal / ca_capacity
                    obs[f"cart_cargo_{URANIUM}"][0, p_id, x, y] = unit.cargo.uranium / ca_capacity
                else:
                    raise NotImplementedError(f'New unit type: {unit}')

            for city in player.cities.values():
                city_fuel_normalized = city.fuel / MAX_FUEL / len(city.citytiles)
                city_light_normalized = city.light_upkeep / ci_light / len(city.citytiles)
                for city_tile in city.citytiles:
                    x, y = city_tile.pos.x, city_tile.pos.y
                    obs["city_tile"][0, p_id, x, y] = 1
                    obs["city_tile_fuel"][0, p_id, x, y] = city_fuel_normalized
                    # NB: This doesn't technically register the light upkeep of a given city tile, but instead
                    # the average light cost of every tile in the given city
                    obs["city_tile_cost"][0, p_id, x, y] = city_light_normalized
                    obs["city_tile_cooldown"][0, p_id, x, y] = city_tile.cooldown / ci_cooldown

            for cell in itertools.chain(*observation.map.map):
                x, y = cell.pos.x, cell.pos.y
                obs["road_level"][0, 0, x, y] = cell.road / max_road
                if cell.has_resource():
                    obs[f"{cell.resource.type}"][0, 0, x, y] = cell.resource.amount / MAX_RESOURCE[cell.resource.type]

            obs["research_points"][0, p_id] = min(player.research_points / max_research, 1.)
            obs["researched_coal"][0, p_id] = player.researched_coal()
            obs["researched_uranium"][0, p_id] = player.researched_uranium()
        obs["night"][0, 0] = observation.is_night
        obs["day_night_cycle"][0, 0] = (observation.turn % DN_CYCLE_LEN) / DN_CYCLE_LEN
        obs["phase"][0, 0] = min(
            observation.turn // DN_CYCLE_LEN,
            GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN - 1
        )
        obs["turn"][0, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        return obs


class FixedShapeContinuousObsV2(FixedShapeObs):
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            # Player specific observations
            # none, worker
            "worker": gym.spaces.MultiBinary((1, P, x, y)),
            # none, cart
            "cart": gym.spaces.MultiBinary((1, P, x, y)),
            # Number of units in the square (only relevant on city tiles)
            "worker_COUNT": gym.spaces.Box(0., float("inf"), shape=(1, P, x, y)),
            "cart_COUNT": gym.spaces.Box(0., float("inf"), shape=(1, P, x, y)),
            # NB: cooldowns and cargo are always zero when on city tiles, so one layer will do for
            # the entire map
            # Normalized from 0-3
            "worker_cooldown": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-5
            "cart_cooldown": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-100
            f"worker_cargo_{WOOD}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"worker_cargo_{COAL}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"worker_cargo_{URANIUM}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-2000
            f"cart_cargo_{WOOD}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"cart_cargo_{COAL}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            f"cart_cargo_{URANIUM}": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Whether the worker/cart is full
            "worker_cargo_full": gym.spaces.MultiBinary((1, P, x, y)),
            "cart_cargo_full": gym.spaces.MultiBinary((1, P, x, y)),
            # none, city_tile
            "city_tile": gym.spaces.MultiBinary((1, P, x, y)),
            # Normalized from 0-MAX_FUEL
            "city_tile_fuel": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-CITY_LIGHT_UPKEEP
            "city_tile_cost": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # Normalized from 0-9
            "city_tile_cooldown": gym.spaces.Box(0., 1., shape=(1, P, x, y)),

            # Player-agnostic observations
            # Normalized from 0-6
            "road_level": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            # Resources
            f"{WOOD}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            f"{COAL}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            f"{URANIUM}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            "dist_from_center_x": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            "dist_from_center_y": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),

            # Non-spatial observations
            # Normalized from 0-200
            "research_points": gym.spaces.Box(0., 1., shape=(1, P)),
            # coal is researched
            "researched_coal": gym.spaces.MultiBinary((1, P)),
            # uranium is researched
            "researched_uranium": gym.spaces.MultiBinary((1, P)),
            # True when it is night
            "night": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + 2),
            # The turn number % 40
            "day_night_cycle": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + DN_CYCLE_LEN),
            # The turn number // 40
            "phase": gym.spaces.MultiDiscrete(
                np.zeros((1, 1)) + GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN
            ),
            # The turn number, normalized from 0-360
            "turn": gym.spaces.Box(0., 1., shape=(1, 1)),
            # 12, 16, 24, or 32
            "board_size": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + len(MAP_SIZES)),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _FixedShapeContinuousObsWrapperV2(env)


class _FixedShapeContinuousObsWrapperV2(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(_FixedShapeContinuousObsWrapperV2, self).__init__(env)
        self._empty_obs = {}
        for key, spec in FixedShapeContinuousObsV2().get_obs_spec().spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        w_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        ca_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
        w_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.
        ca_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.
        ci_light = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
        ci_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
        max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

        obs = {
            key: val.copy() if val.ndim == 2 else val[:, :, :observation.map_width, :observation.map_height].copy()
            for key, val in self._empty_obs.items()
        }

        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                if unit.is_worker():
                    obs["worker"][0, p_id, x, y] = 1
                    obs["worker_COUNT"][0, p_id, x, y] += 1
                    obs["worker_cooldown"][0, p_id, x, y] = unit.cooldown / w_cooldown
                    obs["worker_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    obs[f"worker_cargo_{WOOD}"][0, p_id, x, y] = unit.cargo.wood / w_capacity
                    obs[f"worker_cargo_{COAL}"][0, p_id, x, y] = unit.cargo.coal / w_capacity
                    obs[f"worker_cargo_{URANIUM}"][0, p_id, x, y] = unit.cargo.uranium / w_capacity

                elif unit.is_cart():
                    obs["cart"][0, p_id, x, y] = 1
                    obs["cart_COUNT"][0, p_id, x, y] += 1
                    obs["cart_cooldown"][0, p_id, x, y] = unit.cooldown / ca_cooldown
                    obs["cart_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    obs[f"cart_cargo_{WOOD}"][0, p_id, x, y] = unit.cargo.wood / ca_capacity
                    obs[f"cart_cargo_{COAL}"][0, p_id, x, y] = unit.cargo.coal / ca_capacity
                    obs[f"cart_cargo_{URANIUM}"][0, p_id, x, y] = unit.cargo.uranium / ca_capacity
                else:
                    raise NotImplementedError(f'New unit type: {unit}')

            for city in player.cities.values():
                city_fuel_normalized = city.fuel / MAX_FUEL / len(city.citytiles)
                city_light_normalized = city.light_upkeep / ci_light / len(city.citytiles)
                for city_tile in city.citytiles:
                    x, y = city_tile.pos.x, city_tile.pos.y
                    obs["city_tile"][0, p_id, x, y] = 1
                    obs["city_tile_fuel"][0, p_id, x, y] = city_fuel_normalized
                    # NB: This doesn't technically register the light upkeep of a given city tile, but instead
                    # the average light cost of every tile in the given city
                    obs["city_tile_cost"][0, p_id, x, y] = city_light_normalized
                    obs["city_tile_cooldown"][0, p_id, x, y] = city_tile.cooldown / ci_cooldown

            for cell in itertools.chain(*observation.map.map):
                x, y = cell.pos.x, cell.pos.y
                obs["road_level"][0, 0, x, y] = cell.road / max_road
                if cell.has_resource():
                    obs[f"{cell.resource.type}"][0, 0, x, y] = cell.resource.amount / MAX_RESOURCE[cell.resource.type]

            obs["research_points"][0, p_id] = min(player.research_points / max_research, 1.)
            obs["researched_coal"][0, p_id] = player.researched_coal()
            obs["researched_uranium"][0, p_id] = player.researched_uranium()
        obs["dist_from_center_x"][:] = self.get_dist_from_center_x(observation.map_width, observation.map_height)
        obs["dist_from_center_y"][:] = self.get_dist_from_center_y(observation.map_width, observation.map_height)
        obs["night"][0, 0] = observation.is_night
        obs["day_night_cycle"][0, 0] = observation.turn % DN_CYCLE_LEN
        obs["phase"][0, 0] = min(
            observation.turn // DN_CYCLE_LEN,
            GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN - 1
        )
        obs["turn"][0, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        obs["board_size"][0, 0] = MAP_SIZES.index((observation.map_width, observation.map_height))

        return obs

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_dist_from_center_x(map_height: int, map_width: int) -> np.ndarray:
        pos = np.linspace(0, 2, map_width, dtype=np.float32)[None, :].repeat(map_height, axis=0)
        return np.abs(1 - pos)[None, None, :, :]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_dist_from_center_y(map_height: int, map_width: int) -> np.ndarray:
        pos = np.linspace(0, 2, map_height)[:, None].repeat(map_width, axis=1)
        return np.abs(1 - pos)[None, None, :, :]


class FixedShapeEmbeddingObs(FixedShapeObs):
    """
    An observation consisting almost entirely of embeddings.
    The non-embedded observations are the three resources and 'turn'
    """
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        raise NotImplementedError("This class needs upkeep after the August patch")
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            # Player specific observations
            # none, 1 worker, 2 workers, 3 workers, 4+ workers
            "worker": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 5),
            # none, 1 cart, 2 carts, 3 carts, 4+ carts
            "cart": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 5),
            # NB: cooldowns and cargo are always zero when on city tiles, so one layer will do for
            # the entire map
            # All possible values from 1-3 in 0.5 second increments, + a value for <1
            "worker_cooldown": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 6),
            # All possible values from 1-5 in 0.5 second increments, + a value for <1
            "cart_cooldown": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 10),
            # 1 channel for each resource for cart and worker cargos
            # 5 buckets: [0, 100], increments of 20, 0-19, 20-39, ..., 80-100
            f"worker_cargo_{WOOD}": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 5),
            # 20 buckets: [0, 100], increments of 5, 0-4, 5-9, ..., 95-100
            f"worker_cargo_{COAL}": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20),
            # 50 buckets: [0, 100], increments of 2, 0-1, 2-3, ..., 98-100
            f"worker_cargo_{URANIUM}": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 50),
            # 20 buckets for all resources: 0-99, 100-199, ..., 1800-1899, 1900-2000
            f"cart_cargo_{WOOD}": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20),
            f"cart_cargo_{COAL}": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20),
            f"cart_cargo_{URANIUM}": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20),
            # Whether the worker/cart is full
            "worker_cargo_full": gym.spaces.MultiBinary((1, P, x, y)),
            "cart_cargo_full": gym.spaces.MultiBinary((1, P, x, y)),
            # none, city_tile
            "city_tile": gym.spaces.MultiBinary((1, P, x, y)),
            # How many nights this city tile would survive without more fuel [0-20+], increments of 1
            "city_tile_nights_of_fuel": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 21),

            # 0-30, 31-
            # "city_tile_fuel": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # 10, 15, 20, 25, 30
            # "city_tile_cost": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 5),

            # [0, 9], increments of 1
            "city_tile_cooldown": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 10),

            # Player-agnostic observations
            # [0, 6], increments of 0.5
            "road_level": gym.spaces.MultiDiscrete(np.zeros((1, 1, x, y)) + 13),
            # Resources normalized to [0-1]
            f"{WOOD}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            f"{COAL}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            f"{URANIUM}": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),

            # Non-spatial observations
            # [0, 200], increments of 1
            # "research_points": gym.spaces.MultiDiscrete(
            #     np.zeros((1, P)) + GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"] + 1
            # ),
            "research_points": gym.spaces.Box(0., 1., shape=(1, P)),
            # coal is researched
            "researched_coal": gym.spaces.MultiBinary((1, P)),
            # uranium is researched
            "researched_uranium": gym.spaces.MultiBinary((1, P)),
            # True when it is night
            "night": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + 2),
            # The turn number % 40
            "day_night_cycle": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + DN_CYCLE_LEN),
            # The turn number // 40
            "phase": gym.spaces.MultiDiscrete(
                np.zeros((1, 1)) + GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN
            ),
            # The number of turns
            # "turn": gym.spaces.MultiDiscrete(np.zeros((1, 1)) + GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]),
            "turn": gym.spaces.Box(0., 1., shape=(1, 1)),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _FixedShapeEmbeddingObsWrapper(env)


class _FixedShapeEmbeddingObsWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env
    ):
        super(_FixedShapeEmbeddingObsWrapper, self).__init__(env)
        raise NotImplementedError("This class needs upkeep after the August patch")
        self._empty_obs = {}
        for key, spec in self.observation_space.spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Dict[str, np.ndarray]:
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

        obs = {
            key: val.copy() for key, val in self._empty_obs.items()
        }

        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                if unit.is_worker():
                    obs["worker"][0, p_id, x, y] = min(obs["worker"][0, p_id, x, y] + 1, 4)
                    obs["worker_cooldown"][0, p_id, x, y] = int(min(
                        max(unit.cooldown - 0.5, 0.) * 2,
                        # TODO: This outer min statement should not be necessary
                        5
                    ))
                    # Equivalent to:
                    """
                    obs["worker_cooldown"][0, p_id, x, y] = np.digitize(
                        unit.cooldown,
                        np.linspace(1., 3., 5)
                    )
                    """
                    obs["worker_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    for r, cargo in (WOOD, unit.cargo.wood), (COAL, unit.cargo.coal), (URANIUM, unit.cargo.uranium):
                        collection_rate = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"][r.upper()]
                        obs[f"worker_cargo_{r}"][0, p_id, x, y] = min(
                            cargo // collection_rate,
                            (GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] // collection_rate) - 1
                        )
                        # Equivalent to:
                        """
                        obs[f"worker_cargo_{r}"][0, p_id, x, y] = np.digitize(
                            cargo,
                            np.arange(
                                0,
                                GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"],
                                collection_rate
                            )
                        ) - 1
                        """
                elif unit.is_cart():
                    obs["cart"][0, p_id, x, y] = min(obs["cart"][0, p_id, x, y] + 1, 4)
                    obs["cart_cooldown"][0, p_id, x, y] = int(min(
                        max(unit.cooldown - 0.5, 0.) * 2,
                        # TODO: This outer min statement should not be necessary
                        9
                    ))
                    # Equivalent to:
                    """
                    obs["cart_cooldown"][0, p_id, x, y] = np.digitize(
                        unit.cooldown,
                        np.linspace(1., 5., 9)
                    )
                    """
                    obs["cart_cargo_full"][0, p_id, x, y] = unit.get_cargo_space_left() == 0

                    for r, cargo in (WOOD, unit.cargo.wood), (COAL, unit.cargo.coal), (URANIUM, unit.cargo.uranium):
                        bucket_size = GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]
                        obs[f"cart_cargo_{r}"][0, p_id, x, y] = min(
                            cargo // bucket_size,
                            (GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] // bucket_size) - 1
                        )
                        # Equivalent to:
                        """
                        obs[f"cart_cargo_{r}"][0, p_id, x, y] = np.digitize(
                            cargo,
                            np.arange(
                                0,
                                GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"],
                                bucket_size
                            )
                        ) - 1
                        """
                else:
                    raise NotImplementedError(f'New unit type: {unit}')

            for city in player.cities.values():
                # NB: This doesn't technically register the light upkeep of a given city tile, but instead
                # the average light cost of every tile in the given city
                for city_tile in city.citytiles:
                    x, y = city_tile.pos.x, city_tile.pos.y
                    obs["city_tile"][0, p_id, x, y] = 1
                    obs["city_tile_nights_of_fuel"][0, p_id, x, y] = min(
                        city.fuel // city.light_upkeep,
                        20
                    )
                    # Equivalent to:
                    """
                    obs["city_tile_nights_of_fuel"][0, p_id, x, y] = np.digitize(
                        city_fuel,
                        np.arange(1, 21) * city_light_cost
                    )
                    """
                    # TODO: This min statement should be unnecessary
                    obs["city_tile_cooldown"][0, p_id, x, y] = min(city_tile.cooldown, 9)

            for cell in itertools.chain(*observation.map.map):
                x, y = cell.pos.x, cell.pos.y
                obs["road_level"][0, 0, x, y] = min(int(cell.road * 2), 12)
                # Equivalent to:
                """
                obs["road_level"][0, 0, x, y] = np.digitize(
                    cell.road,
                    np.linspace(0.5, 6., 12)
                )
                """
                if cell.has_resource():
                    obs[f"{cell.resource.type}"][0, 0, x, y] = cell.resource.amount / MAX_RESOURCE[cell.resource.type]

            """
            obs["research_points"][0, p_id] = player.research_points
            """
            obs["research_points"][0, p_id] = min(player.research_points / max_research, 1.)
            obs["researched_coal"][0, p_id] = player.researched_coal()
            obs["researched_uranium"][0, p_id] = player.researched_uranium()
        obs["night"][0, 0] = observation.is_night
        obs["day_night_cycle"][0, 0] = observation.turn % DN_CYCLE_LEN
        obs["phase"][0, 0] = min(
            observation.turn // DN_CYCLE_LEN,
            GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN - 1
        )
        """
        obs["turn"][0, 0] = observation.turn
        """
        obs["turn"][0, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        return obs


class SequenceObs(BaseObsSpace, ABC):
    _entities = [
        "",
        "my_city_tile",
        "opp_city_tile",
        "my_worker",
        "opp_worker",
        "my_cart",
        "opp_cart",
        "non_city_road",
        WOOD,
        COAL,
        URANIUM
    ]
    _entity_encodings = {val: i for i, val in enumerate(_entities)}

    @classmethod
    def get_entities(cls) -> List[str]:
        return cls._entities

    @classmethod
    def get_entity_encodings(cls) -> Dict[str, int]:
        return cls._entity_encodings


class SequenceContinuousObs(SequenceObs):
    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        seq_len = board_dims[0] * board_dims[1]
        return gym.spaces.Dict({
            # One of: "", city_tile (me or opp), worker (me or opp), cart (me or opp), road, coal, wood, uranium
            "entity": gym.spaces.MultiDiscrete(np.zeros((P, 1, seq_len))),
            # The entity's location, rescaled to 0-1
            "pos_x": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),
            "pos_y": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),

            # City tile and unit observations
            # Cooldown normalized from 0-1 (when relevant)
            "normalized_cooldown": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),
            # Whether the unit is full
            "cargo_full": gym.spaces.MultiBinary((P, 1, seq_len)),
            # Cargo normalized from 0-1
            f"cargo_{WOOD}": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),
            f"cargo_{COAL}": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),
            f"cargo_{URANIUM}": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),
            # How much fuel this unit/city_tile has
            "fuel": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),
            # How much fuel this unit costs per turn at night
            "cost": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),

            # Road observations
            "road_level": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),

            # Resource observations
            f"resource_amount": gym.spaces.Box(0., 1., shape=(P, 1, seq_len)),

            # Non-spatial observations
            "research_points": gym.spaces.Box(0., 1., shape=(P, P)),
            # coal is researched
            "researched_coal": gym.spaces.MultiBinary((P, P)),
            # uranium is researched
            "researched_uranium": gym.spaces.MultiBinary((P, P)),
            # True when it is night
            "night": gym.spaces.MultiDiscrete(np.zeros((P, 1)) + 2),
            # The turn number % 40
            "day_night_cycle": gym.spaces.MultiDiscrete(np.zeros((P, 1)) + DN_CYCLE_LEN),
            # The turn number // 40
            "phase": gym.spaces.MultiDiscrete(
                np.zeros((1, 1)) + GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN
            ),
            # The number of turns
            "turn": gym.spaces.Box(0., 1., shape=(P, 1)),
        })

    def wrap_env(self, env, seq_len: int = MAX_BOARD_SIZE[0] * MAX_BOARD_SIZE[1]) -> gym.Wrapper:
        return _SequenceEmbeddingObsWrapper(env, seq_len)


class _SequenceEmbeddingObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seq_len: int):
        super(_SequenceEmbeddingObsWrapper, self).__init__(env)
        self.seq_len = seq_len
        self._empty_obs = {}
        for key, spec in self.observation_space.spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros((*spec.shape[:-1], seq_len), dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros((*spec.shape[:-1], seq_len), dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")
        self._empty_info = {
            "pos_x": np.zeros((*self.observation_space.spaces["pos_x"].shape[:-1], seq_len), dtype=np.int64),
            "pos_y": np.zeros((*self.observation_space.spaces["pos_y"].shape[:-1], seq_len), dtype=np.int64),
            "sequence_mask": np.zeros((*self.observation_space.spaces["entity"].shape[:-1], seq_len), dtype=bool),
        }

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        max_capacity = max(GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"],
                           GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"])
        max_cooldown = max(GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.,
                           GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.,
                           GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"] - 1.)
        wood_fuel_val = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"]["WOOD"]
        coal_fuel_val = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"]["COAL"]
        uranium_fuel_val = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"]["URANIUM"]
        max_light_cost = max(GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"].values())
        worker_light_cost = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER"]
        cart_light_cost = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CART"]
        city_light_cost = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
        max_resources_on_tile = max(MAX_RESOURCE.values())
        max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

        obs = {
            key: val.copy() for key, val in self._empty_obs.items()
        }
        extra_info = {
            key: val.copy() for key, val in self._empty_info.items()
        }

        seq_idx = 0
        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                # Break with warning if the max seq_len is reached
                if seq_idx is None or seq_idx >= self.seq_len:
                    seq_idx = None
                    break
                if unit.is_worker():
                    unit_type = "worker"
                    light_cost = worker_light_cost
                elif unit.is_cart():
                    unit_type = "cart"
                    light_cost = cart_light_cost
                else:
                    raise NotImplementedError(f'New unit type: {unit}')
                obs["entity"][p_id, 0, seq_idx] = SequenceContinuousObs.get_entity_encodings()[f"my_{unit_type}"]
                obs["entity"][p_id - 1, 0, seq_idx] = SequenceContinuousObs.get_entity_encodings()[f"opp_{unit_type}"]
                obs["normalized_cooldown"][:, 0, seq_idx] = unit.cooldown / max_cooldown
                obs["cargo_full"][:, 0, seq_idx] = unit.get_cargo_space_left() == 0
                obs[f"cargo_{WOOD}"][:, 0, seq_idx] = unit.cargo.wood / max_capacity
                obs[f"cargo_{COAL}"][:, 0, seq_idx] = unit.cargo.coal / max_capacity
                obs[f"cargo_{URANIUM}"][:, 0, seq_idx] = unit.cargo.uranium / max_capacity
                obs["fuel"][:, 0, seq_idx] = (unit.cargo.wood * wood_fuel_val +
                                              unit.cargo.coal * coal_fuel_val +
                                              unit.cargo.uranium * uranium_fuel_val) / MAX_FUEL
                obs["cost"][:, 0, seq_idx] = light_cost / max_light_cost
                seq_idx += 1

            city_tile_positions = set()
            for city in player.cities.values():
                # Break with warning if the max seq_len is reached
                if seq_idx is None or seq_idx >= self.seq_len:
                    seq_idx = None
                    break
                # NB: This doesn't technically register the light upkeep of a given city tile, but instead
                # the average light cost of every tile in the given city
                city_fuel_normalized = city.fuel / MAX_FUEL / len(city.citytiles)
                city_light_normalized = city.light_upkeep / city_light_cost / len(city.citytiles)
                for city_tile in city.citytiles:
                    # Break with warning if the max seq_len is reached
                    if seq_idx is None or seq_idx >= self.seq_len:
                        seq_idx = None
                        break
                    city_tile_positions.add(city_tile.pos.astuple())
                    obs["entity"][p_id, 0, seq_idx] = SequenceContinuousObs.get_entity_encodings()["my_city_tile"]
                    obs["entity"][p_id - 1, 0, seq_idx] = SequenceContinuousObs.get_entity_encodings()["opp_city_tile"]
                    obs["normalized_cooldown"] = city_tile.cooldown / max_cooldown
                    obs["fuel"][:, 0, seq_idx] = city_fuel_normalized
                    obs["cost"] = city_light_normalized
                    seq_idx += 1

            # First iterate over cells and place resources, then roads afterwards in case seq_len is reached
            road_cells = []
            for cell in itertools.chain(*observation.map.map):
                # Break with warning if the max seq_len is reached
                if seq_idx is None or seq_idx >= self.seq_len:
                    seq_idx = None
                    break
                if cell.road > 0. and not cell.pos.astuple() in city_tile_positions:
                    road_cells.append(cell)
                if cell.has_resource():
                    obs["entity"][:, 0, seq_idx] = SequenceContinuousObs.get_entity_encodings()[cell.resource.type]
                    obs["resource_amount"] = cell.resource.amount / max_resources_on_tile
            for cell in road_cells:
                # Break with warning if the max seq_len is reached
                if seq_idx is None or seq_idx >= self.seq_len:
                    seq_idx = None
                    break
                obs["entity"][:, 0, seq_idx] = SequenceContinuousObs.get_entity_encodings()["non_city_road"]
                obs["road_level"][:, 0, seq_idx] = cell.road / max_road

            if seq_idx is None:
                logging.warning(f"{self.__class__.__name__}: maximum sequence length of {self.seq_len} exceeded.")

        for p_id in (0, 1):
            player = observation.players[p_id]
            if p_id == 0:
                obs["research_points"][:, :] = min(player.research_points / max_research, 1.)
                obs["researched_coal"][:, :] = player.researched_coal()
                obs["researched_uranium"][:, :] = player.researched_uranium()
            else:
                obs["research_points"][0, 1] = min(player.research_points / max_research, 1.)
                obs["research_points"][1, 0] = min(player.research_points / max_research, 1.)
                obs["researched_coal"][0, 1] = player.researched_coal()
                obs["researched_coal"][1, 0] = player.researched_coal()
                obs["researched_uranium"][0, 1] = player.researched_uranium()
                obs["researched_uranium"][1, 0] = player.researched_uranium()

        obs["night"][:, 0] = observation.is_night
        obs["day_night_cycle"][:, 0] = (observation.turn % DN_CYCLE_LEN) / DN_CYCLE_LEN
        obs["phase"][0, 0] = min(
            observation.turn // DN_CYCLE_LEN,
            GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN - 1
        )
        obs["turn"][:, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        return obs, extra_info
