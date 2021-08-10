from abc import ABC, abstractmethod
import gym
import itertools
import numpy as np
from typing import Optional

from . import reward_spaces
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS

# TODO: Verify max resources on launch
MAX_RESOURCE = {
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L227
    Constants.RESOURCE_TYPES.WOOD: 400.,
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L248
    Constants.RESOURCE_TYPES.COAL: 650.,
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L269
    Constants.RESOURCE_TYPES.URANIUM: 600.
}
# TODO: Fix max fuel amount
MAX_FUEL = 5000.
UNIT_ENCODING = {
    Constants.UNIT_TYPES.WORKER: 0,
    Constants.UNIT_TYPES.CART: 1,
    None: 2
}
RESOURCE_ENCODING = {
    Constants.RESOURCE_TYPES.WOOD: 0,
    Constants.RESOURCE_TYPES.COAL: 1,
    Constants.RESOURCE_TYPES.URANIUM: 2,
    None: 3
}
MAX_BOARD_SIZE = (32, 32)
ALL_SUBTASKS = []
for rspace in reward_spaces.__dict__.values():
    if isinstance(rspace, type) and issubclass(rspace, reward_spaces.Subtask) and rspace is not reward_spaces.Subtask:
        ALL_SUBTASKS.append(rspace)
ALL_SUBTASKS.append(None)
SUBTASK_ENCODING = {
    task: i for i, task in enumerate(ALL_SUBTASKS)
}


class BaseObsSpace(ABC):
    def __init__(self, include_subtask_encoding: bool = False, override_n_subtasks: Optional[int] = None):
        self.include_subtask_encoding = include_subtask_encoding
        if override_n_subtasks:
            self.n_subtasks = override_n_subtasks
        else:
            self.n_subtasks = len(SUBTASK_ENCODING)

    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(
            self,
            board_dims: tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        if self.include_subtask_encoding:
            return self.get_subtask_encoding(board_dims)
        else:
            return gym.spaces.Dict({})

    @abstractmethod
    def get_subtask_encoding(self, board_dims: tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def wrap_env(self, env, reward_space: Optional[reward_spaces.BaseRewardSpace]) -> gym.Wrapper:
        pass


class FixedShapeContinuousObs(BaseObsSpace):
    def get_obs_spec(
            self,
            board_dims: tuple[int, int] = MAX_BOARD_SIZE
    ) -> gym.spaces.Dict:
        subtask_encoding = super(FixedShapeContinuousObs, self).get_obs_spec(board_dims)
        x = board_dims[0]
        y = board_dims[1]
        # Player count
        p = 2
        return gym.spaces.Dict({
            # Player specific observations
            # none, worker
            "worker": gym.spaces.MultiBinary((1, p, x, y)),
            # none, cart
            "cart": gym.spaces.MultiBinary((1, p, x, y)),
            # Number of units in the square (only relevant on city tiles)
            "worker_COUNT": gym.spaces.Box(0., float("inf"), shape=(1, p, x, y)),
            "cart_COUNT": gym.spaces.Box(0., float("inf"), shape=(1, p, x, y)),
            # NB: cooldowns and cargo are always zero when on city tiles, so one layer will do for
            # the entire map
            # Normalized from 0-3
            "worker_cooldown": gym.spaces.Box(0., 1., shape=(1, p, x, y)),
            # Normalized from 0-5
            "cart_cooldown": gym.spaces.Box(0., 1., shape=(1, p, x, y)),
            # Normalized from 0-100
            "worker_cargo": gym.spaces.Box(0., 1., shape=(3, p, x, y)),
            # Normalized from 0-2000
            "cart_cargo": gym.spaces.Box(0., 1., shape=(3, p, x, y)),
            # none, city_tile
            "city_tile": gym.spaces.MultiBinary((1, p, x, y)),
            # Normalized from 0-MAX_FUEL
            "city_tile_fuel": gym.spaces.Box(0., 1., shape=(1, p, x, y)),
            # Normalized from 0-30
            "city_tile_cost": gym.spaces.Box(0., 1., shape=(1, p, x, y)),
            # Normalized from 0-9
            "city_tile_cooldown": gym.spaces.Box(0., 1., shape=(1, p, x, y)),

            # Player-agnostic observations
            # Normalized from 0-6
            "road_level": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            # Wood, coal, uranium
            "resources": gym.spaces.Box(0., 1., shape=(3, 1, x, y)),

            # Non-spatial observations
            # Normalized from 0-200
            "research_points": gym.spaces.Box(0., 1., shape=(1, p)),
            # coal is researched
            "researched_coal": gym.spaces.MultiBinary((1, p)),
            # uranium is researched
            "researched_uranium": gym.spaces.MultiBinary((1, p)),
            # True when it is night
            "night": gym.spaces.MultiBinary((1, 1)),
            # The turn number, normalized from 0-360
            "turn": gym.spaces.Box(0., 1., shape=(1, 1)),
            **subtask_encoding.spaces
        })

    def get_subtask_encoding(self, board_dims: tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            "subtask": gym.spaces.MultiDiscrete(np.zeros((1, 1, x, y), dtype=int) + self.n_subtasks)
        })

    def wrap_env(self, env, subtask_reward_space: Optional[reward_spaces.Subtask] = None) -> gym.Wrapper:
        return _FixedShapeContinuousObsWrapper(env, self.include_subtask_encoding, subtask_reward_space)


class _FixedShapeContinuousObsWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            include_subtask_encoding: bool,
            subtask_reward_space: Optional[reward_spaces.Subtask]
    ):
        super(_FixedShapeContinuousObsWrapper, self).__init__(env)
        if include_subtask_encoding:
            if subtask_reward_space is None:
                raise ValueError("Cannot use subtask_encoding without providing subtask_reward_space.")
            elif not isinstance(subtask_reward_space, reward_spaces.Subtask):
                raise ValueError("Reward_space must be an instance of Subtask")
        self.include_subtask_encoding = include_subtask_encoding
        self.subtask_reward_space = subtask_reward_space
        self._empty_obs = {}
        for key, spec in self.observation_space.spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32)
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Game) -> dict[str, np.ndarray]:
        w_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        ca_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
        w_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.
        ca_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.
        ci_light = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
        ci_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
        max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

        wood_idx = RESOURCE_ENCODING[Constants.RESOURCE_TYPES.WOOD]
        coal_idx = RESOURCE_ENCODING[Constants.RESOURCE_TYPES.COAL]
        uranium_idx = RESOURCE_ENCODING[Constants.RESOURCE_TYPES.URANIUM]
        obs = {
            key: val.copy() for key, val in self._empty_obs.items()
        }

        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                if unit.is_worker():
                    obs["worker"][0, p_id, x, y] = 1
                    obs["worker_COUNT"][0, p_id, x, y] += 1
                    obs["worker_cooldown"][0, p_id, x, y] = unit.cooldown / w_cooldown

                    obs["worker_cargo"][wood_idx, p_id, x, y] = unit.cargo.wood / w_capacity
                    obs["worker_cargo"][coal_idx, p_id, x, y] = unit.cargo.coal / w_capacity
                    obs["worker_cargo"][uranium_idx, p_id, x, y] = unit.cargo.uranium / w_capacity
                elif unit.is_cart():
                    obs["cart"][0, p_id, x, y] = 1
                    obs["cart_COUNT"][0, p_id, x, y] += 1
                    obs["cart_cooldown"][0, p_id, x, y] = unit.cooldown / ca_cooldown

                    obs["cart_cargo"][wood_idx, p_id, x, y] = unit.cargo.wood / ca_capacity
                    obs["cart_cargo"][coal_idx, p_id, x, y] = unit.cargo.coal / ca_capacity
                    obs["cart_cargo"][uranium_idx, p_id, x, y] = unit.cargo.uranium / ca_capacity
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
                    obs["resources"][RESOURCE_ENCODING[cell.resource.type], 0, x, y] = \
                        cell.resource.amount / MAX_RESOURCE[cell.resource.type]

            obs["research_points"][0, p_id] = player.research_points / max_research
            obs["researched_coal"][0, p_id] = player.researched_coal()
            obs["researched_uranium"][0, p_id] = player.researched_uranium()
        dn_cycle_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        obs["night"][0, 0] = observation.turn % dn_cycle_len >= GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        obs["turn"][0, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        if self.include_subtask_encoding:
            obs["subtask"][:] = self.subtask_reward_space.get_subtask_encoding(SUBTASK_ENCODING)

        return obs
