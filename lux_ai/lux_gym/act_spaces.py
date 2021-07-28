from abc import ABC, abstractmethod
from functools import cache, cached_property
import gym
import numpy as np
import torch
from typing import *


from .obs_spaces import MAX_BOARD_SIZE
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import CityTile, Unit

DIRECTIONS = (
    Constants.DIRECTIONS.NORTH,
    Constants.DIRECTIONS.EAST,
    Constants.DIRECTIONS.SOUTH,
    Constants.DIRECTIONS.WEST
)
RESOURCES = (
    Constants.RESOURCE_TYPES.WOOD,
    Constants.RESOURCE_TYPES.COAL,
    Constants.RESOURCE_TYPES.URANIUM
)
_MAX_CAPACITY = max(GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"].values())

ACTION_MEANINGS = {
    "worker": [
        "NO-OP",
    ],
    "cart": [
        "NO-OP",
    ],
    "city_tile": [
        "NO-OP",
        "BUILD_WORKER",
        "BUILD_CART",
        "RESEARCH",
    ]
}
for u in ["worker", "cart"]:
    for d in DIRECTIONS:
        ACTION_MEANINGS[u].append(f"MOVE_{d}")
    for r in RESOURCES:
        for d in DIRECTIONS:
            ACTION_MEANINGS[u].append(f"TRANSFER_{r}_{d}")
ACTION_MEANINGS["worker"].extend(["PILLAGE", "BUILD_CITY"])
ACTION_MEANINGS_TO_IDX = {
    actor: {
        action: idx for action, idx in enumerate(actions)
    } for actor, actions in ACTION_MEANINGS.items()
}


def _no_op(game_object: Union[Unit, CityTile]) -> Optional[str]:
    return None


def _pillage(worker: Unit) -> str:
    return worker.pillage()


def _build_city(worker: Unit) -> str:
    return worker.build_city()


def _build_worker(city_tile: CityTile) -> str:
    return city_tile.build_worker()


def _build_cart(city_tile: CityTile) -> str:
    return city_tile.build_cart()


def _research(city_tile: CityTile) -> str:
    return city_tile.research()


def _move_factory(action_meaning: str) -> Callable[[Unit], str]:
    direction = action_meaning.split("_")[1]
    if direction not in DIRECTIONS:
        raise ValueError(f"Unrecognized direction '{direction}' in action_meaning '{action_meaning}'")

    def _move_func(unit: Unit) -> str:
        return unit.move(direction)

    return _move_func


def _transfer_factory(action_meaning: str) -> Callable[..., str]:
    resource, direction = action_meaning.split("_")[1:]
    if resource not in RESOURCES:
        raise ValueError(f"Unrecognized resource '{resource}' in action_meaning '{action_meaning}'")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unrecognized direction '{direction}' in action_meaning '{action_meaning}'")

    def _transfer_func(unit: Unit, pos_to_unit_dict: dict[tuple, Optional[Unit]]) -> str:
        dest_pos = unit.pos.translate(direction, 1)
        dest_unit = pos_to_unit_dict.get((dest_pos.x, dest_pos.y), None)
        # If the square is not on the map or there is not an allied unit in that square
        if dest_unit is None:
            return ""
        # NB: Technically, this does limit the agent's action space, particularly in that they cannot transfer anything
        # except the maximum amount. I don't want to deal with continuous action spaces, but perhaps the transfer
        # action could be bucketed if partial transfers become important.
        # The game engine automatically determines the actual maximum legal transfer
        # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/index.ts#L704
        return unit.transfer(dest_id=dest_unit, resourceType=resource, amount=_MAX_CAPACITY)

    return _transfer_func


ACTION_MEANING_TO_FUNC = {
    "worker": {
        "NO-OP": _no_op,
        "PILLAGE": _pillage,
        "BUILD_CITY": _build_city,
    },
    "cart": {
        "NO-OP": _no_op,
    },
    "city_tile": {
        "NO-OP": _no_op,
        "BUILD_WORKER": _build_worker,
        "BUILD_CART": _build_cart,
        "RESEARCH": _research,
    }
}
for u in ["worker", "cart"]:
    for d in DIRECTIONS:
        a = f"MOVE_{d}"
        ACTION_MEANING_TO_FUNC[a] = _move_factory(a)


class ActSpace(ABC):
    @abstractmethod
    def get_action_space(self, board_dims: tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def process_actions(
            self,
            action_tensors_dict: dict[str, np.ndarray],
            game_state: Game,
            board_dims: tuple[int, int],
            pos_to_unit_dict: dict[tuple, Optional[Unit]]
    ) -> tuple[list[list[str]], dict[str, np.ndarray]]:
        pass

    @cached_property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.get_action_space().spaces.keys())

    @cached_property
    def _DEPRECATING_act_space_channel_slices(self) -> dict[str, slice]:
        channel_slices = {}
        last_val = 0
        for key, val in self.get_action_space().spaces.items():
            channel_slices[key] = slice(last_val, val)
            last_val = val
        return channel_slices

    def from_dict(
            self,
            actions_like: dict[str, Union[torch.Tensor, np.ndarray]],
            expanded: bool = False,
            concatenation_func: Callable = torch.cat
    ) -> Union[torch.Tensor, np.ndarray]:
        if expanded:
            # In the case that the actions dimension has been expanded at the end
            return concatenation_func([actions_like[key] for key in self.keys], -5)
        else:
            return concatenation_func([actions_like[key] for key in self.keys], -4)

    def DEPRECATING_to_dict(self, actions_array_like: Any, expanded: bool = False) -> dict[str, Any]:
        if expanded:
            # In the case that the actions dimension has been expanded at the end
            return {
                key: actions_array_like[..., s, :, :, :, :] for key, s in self._DEPRECATING_act_space_channel_slices
            }
        else:
            return {
                key: actions_array_like[..., s, :, :, :] for key, s in self._DEPRECATING_act_space_channel_slices
            }


class BasicActionSpace(ActSpace):
    def __init__(self, default_board_dims: Optional[tuple[int, int]] = None):
        self.default_board_dims = MAX_BOARD_SIZE if default_board_dims is None else default_board_dims

    @cache
    def get_action_space(self, board_dims: Optional[tuple[int, int]] = None) -> gym.spaces.Dict:
        if board_dims is None:
            board_dims = self.default_board_dims
        x = board_dims[0]
        y = board_dims[1]
        # Player count
        p = 2
        # There are up to 4 action planes for workers/carts when they are stacked on city tiles.
        # All remaining actions are no-ops
        return gym.spaces.Dict({
            "worker": gym.spaces.MultiDiscrete(np.zeros((4, p, x, y), dtype=int) + len(ACTION_MEANINGS["worker"])),
            "cart": gym.spaces.MultiDiscrete(np.zeros((4, p, x, y), dtype=int) + len(ACTION_MEANINGS["cart"])),
            "city_tile": gym.spaces.MultiDiscrete(
                np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["city_tile"])
            ),
        })

    @cache
    def get_action_space_expanded_shape(self, *args, **kwargs) -> dict[str, tuple[int, ...]]:
        action_space = self.get_action_space(*args, **kwargs)
        action_space_expanded = {}
        for key, val in action_space.spaces.items():
            action_space_expanded[key] = val.shape + (len(ACTION_MEANINGS[key]),)
        return action_space_expanded

    def process_actions(
            self,
            action_tensors_dict: dict[str, np.ndarray],
            game_state: Game,
            board_dims: tuple[int, int],
            pos_to_unit_dict: dict[tuple, Optional[Unit]]
    ) -> tuple[list[list[str]], dict[str, np.ndarray]]:
        action_strs = [[], []]
        actions_taken = {
            key: np.zeros(space.shape, dtype=bool) for key, space in self.get_action_space(board_dims).spaces.items()
        }
        for player in game_state.players:
            p_id = player.team
            worker_actions_taken_count = np.zeros(board_dims, dtype=int)
            cart_actions_taken_count = np.zeros_like(worker_actions_taken_count)
            for unit in player.units:
                if unit.can_act():
                    x, y = unit.pos.x, unit.pos.y
                    if unit.is_worker():
                        unit_type = "worker"
                        actions_taken_count = worker_actions_taken_count
                    elif unit.is_cart():
                        unit_type = "cart"
                        actions_taken_count = cart_actions_taken_count
                    else:
                        raise NotImplementedError(f'New unit type: {unit}')
                    # Action plane is selected for stacked units
                    action_plane = actions_taken_count[x, y]
                    action_idx = action_tensors_dict[unit_type][action_plane, p_id, x, y]
                    action = get_unit_action(unit, action_idx, pos_to_unit_dict)
                    actions_taken[unit_type][action_plane, p_id, x, y] = True
                    # None means no-op
                    # "" means invalid transfer action - fed to game as no-op
                    if action is not None and action != "":
                        # noinspection PyTypeChecker
                        action_strs[p_id].append(action)
                    actions_taken_count[x, y] += 1
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        x, y = city_tile.pos.x, city_tile.pos.y
                        action_idx = action_tensors_dict["city_tile"][0, p_id, x, y]
                        action = get_city_tile_action(city_tile, action_idx)
                        actions_taken["city_tile"][0, p_id, x, y] = True
                        # None means no-op
                        if action is not None:
                            # noinspection PyTypeChecker
                            action_strs[p_id].append(action)
        return action_strs, actions_taken


def get_unit_action(unit: Unit, action_idx: int, pos_to_unit_dict: dict[tuple, Optional[Unit]]) -> Optional[str]:
    if unit.is_worker():
        unit_type = "worker"
    elif unit.is_cart():
        unit_type = "cart"
    else:
        raise NotImplementedError(f'New unit type: {unit}')
    action = ACTION_MEANINGS[unit_type][action_idx]
    if action.startswith("TRANSFER"):
        # noinspection PyArgumentList
        return ACTION_MEANING_TO_FUNC[unit_type][action](unit, pos_to_unit_dict)
    else:
        return ACTION_MEANING_TO_FUNC[unit_type][action](unit)


def get_city_tile_action(city_tile: CityTile, action_idx: int) -> Optional[str]:
    action = ACTION_MEANINGS["city_tile"][action_idx]
    return ACTION_MEANING_TO_FUNC["city_tile"][action](city_tile)
