import itertools
from enum import Enum, auto
import gym
import numpy as np

from .lux.game import Game
from .lux.constants import Constants
from .lux.game_constants import GAME_CONSTANTS

MAX_BOARD_SIZE = (32, 32)
MAX_RESOURCE = {
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L227
    Constants.RESOURCE_TYPES.WOOD: 1300.,
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L248
    Constants.RESOURCE_TYPES.COAL: 450.,
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L269
    Constants.RESOURCE_TYPES.URANIUM: 350.
}

MAX_FUEL = None
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


class ObsType(Enum):
    """
    An enum of all available obs_types
    WARNING: enum order is subject to change
    """
    FIXED_SHAPE_CONTINUOUS_OBS = auto()
    VARIABLE_SHAPE_EMBEDDING_OBS = auto()

    # NB: Avoid using Discrete() space, as it returns a shape of ()
    def get_obs_spec(self, board_size: tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        x = board_size[0]
        y = board_size[1]
        # Player count
        p = 2
        if self == ObsType.FIXED_SHAPE_CONTINUOUS_OBS:
            return gym.spaces.Dict({
                # Player specific observations
                # none, worker
                "worker": gym.spaces.MultiBinary((1, p, x, y)),
                # none, cart
                "cart": gym.spaces.MultiBinary((1, p, x, y)),
                # Number of units in the square (only relevant on city tiles)
                "worker_count": gym.spaces.Box(0., float("inf"), shape=(1, p, x, y)),
                "cart_count": gym.spaces.Box(0., float("inf"), shape=(1, p, x, y)),
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
                "resource_count": gym.spaces.Box(0., 1., shape=(3, 1, x, y)),

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
                "turn": gym.spaces.Box(0., 1., shape=(1, 1))
            })
        else:
            raise NotImplementedError(f'ObsType not yet implemented: {self.name}')

    def wrap_env(self, env) -> gym.ObservationWrapper:
        if self == ObsType.FIXED_SHAPE_CONTINUOUS_OBS:
            return _FixedShapeContinuousObs(env)
        else:
            raise NotImplementedError(f'ObsType not yet implemented: {self.name}')


class _FixedShapeContinuousObs(gym.ObservationWrapper):
    def observation(self, observation: Game) -> dict[str, np.ndarray]:
        w_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
        ca_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
        w_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.
        ca_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.
        ci_light = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
        ci_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
        max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
        max_research = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())
        obs = {
            key: np.zeros(spec.shape, dtype=np.float32) for spec, key in self.observation_space
        }

        for player in observation.players:
            p_id = player.team
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                if unit.is_worker():
                    obs["worker"][0, p_id, x, y] = 1.
                    obs["worker_count"][0, p_id, x, y] += 1.
                    obs["worker_cooldown"][0, p_id, x, y] = unit.cooldown / w_cooldown

                    obs["worker_cargo"][Constants.RESOURCE_TYPES.WOOD, p_id, x, y] = unit.cargo.wood / w_capacity
                    obs["worker_cargo"][Constants.RESOURCE_TYPES.COAL, p_id, x, y] = unit.cargo.coal / w_capacity
                    obs["worker_cargo"][Constants.RESOURCE_TYPES.URANIUM, p_id, x, y] = unit.cargo.uranium / w_capacity
                elif unit.is_cart():
                    obs["cart"][0, p_id, x, y] = 1.
                    obs["cart_count"][0, p_id, x, y] += 1.
                    obs["cart_cooldown"][0, p_id, x, y] = unit.cooldown / ca_cooldown

                    obs["cart_cargo"][Constants.RESOURCE_TYPES.WOOD, p_id, x, y] = unit.cargo.wood / ca_capacity
                    obs["cart_cargo"][Constants.RESOURCE_TYPES.COAL, p_id, x, y] = unit.cargo.coal / ca_capacity
                    obs["cart_cargo"][Constants.RESOURCE_TYPES.URANIUM, p_id, x, y] = unit.cargo.uranium / ca_capacity
                else:
                    raise NotImplementedError(f'New unit type: {unit}')

            for city in player.cities.values():
                city_fuel_normalized = city.fuel / MAX_FUEL / len(city.citytiles)
                city_light_normalized = city.light_upkeep / ci_light / len(city.citytiles)
                for city_tile in city.citytiles:
                    x, y = city_tile.pos.x, city_tile.pos.y
                    obs["city_tile"][0, p_id, x, y] = 1.
                    obs["city_tile_fuel"][0, p_id, x, y] = city_fuel_normalized
                    # NB: This doesn't technically register the light upkeep of a given city tile, but instead
                    # the average light cost of every tile in the given city
                    obs["city_tile_cost"][0, p_id, x, y] = city_light_normalized
                    obs["city_tile_cooldown"][0, p_id, x, y] = city_tile.cooldown / ci_cooldown

            for cell in itertools.chain(*observation.map.map):
                x, y = cell.pos.x, cell.pos.y
                obs["road_level"][0, 0, x, y] = cell.road / max_road
                if cell.has_resource():
                    obs["resource_count"][RESOURCE_ENCODING[cell.resource.type], 0, x, y] = \
                        cell.resource.amount / MAX_RESOURCE[cell.resource.type]

            obs["research_points"][0, p_id] = player.research_points / max_research
            obs["researched_coal"][0, p_id] = player.researched_coal()
            obs["researched_uranium"][1, p_id] = player.researched_uranium()
        dn_cycle_len = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        obs["night"][0, 0] = (observation.turn - 1) % dn_cycle_len >= GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        obs["turn"][0, 0] = observation.turn / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]

        return obs

    @property
    def observation_space(self):
        return self.env.observation_space
