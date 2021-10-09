from abc import ABC, abstractmethod
from functools import lru_cache
import gym
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


from ..utility_constants import MAX_CAPACITY, MAX_RESEARCH, MAX_BOARD_SIZE
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_objects import CityTile, Unit

# The maximum number of actions that can be taken by units sharing a square
# All remaining units take the no-op action
MAX_OVERLAPPING_ACTIONS = 4
DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=False)
RESOURCES = Constants.RESOURCE_TYPES.astuple()

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
        action: idx for idx, action in enumerate(actions)
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

    def _transfer_func(unit: Unit, pos_to_unit_dict: Dict[Tuple, Optional[Unit]]) -> str:
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
        return unit.transfer(dest_id=dest_unit.id, resourceType=resource, amount=MAX_CAPACITY)

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
        ACTION_MEANING_TO_FUNC[u][a] = _move_factory(a)
    for r in RESOURCES:
        for d in DIRECTIONS:
            actions_str = f"TRANSFER_{r}_{d}"
            ACTION_MEANING_TO_FUNC[u][actions_str] = _transfer_factory(actions_str)


class BaseActSpace(ABC):
    @abstractmethod
    def get_action_space(self, board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def process_actions(
            self,
            action_tensors_dict: Dict[str, np.ndarray],
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        pass

    @abstractmethod
    def get_available_actions_mask(
            self,
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
            pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
    ) -> Dict[str, np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        pass


class BasicActionSpace(BaseActSpace):
    def __init__(self, default_board_dims: Optional[Tuple[int, int]] = None):
        self.default_board_dims = MAX_BOARD_SIZE if default_board_dims is None else default_board_dims

    @lru_cache(maxsize=None)
    def get_action_space(self, board_dims: Optional[Tuple[int, int]] = None) -> gym.spaces.Dict:
        if board_dims is None:
            board_dims = self.default_board_dims
        x = board_dims[0]
        y = board_dims[1]
        # Player count
        p = 2
        return gym.spaces.Dict({
            "worker": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["worker"])),
            "cart": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["cart"])),
            "city_tile": gym.spaces.MultiDiscrete(
                np.zeros((1, p, x, y), dtype=int) + len(ACTION_MEANINGS["city_tile"])
            ),
        })

    @lru_cache(maxsize=None)
    def get_action_space_expanded_shape(self, *args, **kwargs) -> Dict[str, Tuple[int, ...]]:
        action_space = self.get_action_space(*args, **kwargs)
        action_space_expanded = {}
        for key, val in action_space.spaces.items():
            action_space_expanded[key] = val.shape + (len(ACTION_MEANINGS[key]),)
        return action_space_expanded

    def process_actions(
            self,
            action_tensors_dict: Dict[str, np.ndarray],
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        action_strs = [[], []]
        actions_taken = {
            key: np.zeros(space, dtype=bool) for key, space in self.get_action_space_expanded_shape(board_dims).items()
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
                    actor_count = actions_taken_count[x, y]
                    if actor_count >= MAX_OVERLAPPING_ACTIONS:
                        action = None
                    else:
                        action_idx = action_tensors_dict[unit_type][0, p_id, x, y, actor_count]
                        action_meaning = ACTION_MEANINGS[unit_type][action_idx]
                        action = get_unit_action(unit, action_idx, pos_to_unit_dict)
                        action_was_taken = action_meaning == "NO-OP" or (action is not None and action != "")
                        actions_taken[unit_type][0, p_id, x, y, action_idx] = action_was_taken
                        # If action is NO-OP, skip remaining actions for units at same location
                        if action_meaning == "NO-OP":
                            actions_taken_count[x, y] += MAX_OVERLAPPING_ACTIONS
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
                        action_idx = action_tensors_dict["city_tile"][0, p_id, x, y, 0]
                        action_meaning = ACTION_MEANINGS["city_tile"][action_idx]
                        action = get_city_tile_action(city_tile, action_idx)
                        action_was_taken = action_meaning == "NO-OP" or (action is not None and action != "")
                        actions_taken["city_tile"][0, p_id, x, y, action_idx] = action_was_taken
                        # None means no-op
                        if action is not None:
                            # noinspection PyTypeChecker
                            action_strs[p_id].append(action)
        return action_strs, actions_taken

    def get_available_actions_mask(
            self,
            game_state: Game,
            board_dims: Tuple[int, int],
            pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
            pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
    ) -> Dict[str, np.ndarray]:
        available_actions_mask = {
            key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
            for key, space in self.get_action_space(board_dims).spaces.items()
        }
        for player in game_state.players:
            p_id = player.team
            for unit in player.units:
                if unit.can_act():
                    x, y = unit.pos.x, unit.pos.y
                    if unit.is_worker():
                        unit_type = "worker"
                    elif unit.is_cart():
                        unit_type = "cart"
                    else:
                        raise NotImplementedError(f"New unit type: {unit}")
                    # No-op is always a legal action
                    # Moving is usually a legal action, except when:
                    #   The unit is at the edge of the board and would try to move off of it
                    #   The unit would move onto an opposing city tile
                    #   The unit would move onto another unit with cooldown > 0
                    # Transferring is only a legal action when:
                    #   There is an allied unit in the target square
                    #   The transferring unit has > 0 cargo of the designated resource
                    #   The receiving unit has cargo space remaining
                    # Workers: Pillaging is only a legal action when on a road tile and is not on an allied city
                    # Workers: Building a city is only a legal action when the worker has the required resources and
                    #       is not on a resource tile
                    for direction in DIRECTIONS:
                        new_pos_tuple = unit.pos.translate(direction, 1)
                        new_pos_tuple = new_pos_tuple.x, new_pos_tuple.y
                        # Moving and transferring - check that the target position exists on the board
                        if new_pos_tuple not in pos_to_unit_dict.keys():
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False
                            for resource in RESOURCES:
                                available_actions_mask[unit_type][
                                    :,
                                    p_id,
                                    x,
                                    y,
                                    ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
                                ] = False
                            continue
                        # Moving - check that the target position does not contain an opposing city tile
                        new_pos_city_tile = pos_to_city_tile_dict[new_pos_tuple]
                        if new_pos_city_tile and new_pos_city_tile.team != p_id:
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False
                        # Moving - check that the target position does not contain a unit with cooldown > 0
                        new_pos_unit = pos_to_unit_dict[new_pos_tuple]
                        if new_pos_unit and new_pos_unit.cooldown > 0:
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
                            ] = False
                        for resource in RESOURCES:
                            if (
                                    # Transferring - check that there is an allied unit in the target square
                                    (new_pos_unit is None or new_pos_unit.team != p_id) or
                                    # Transferring - check that the transferring unit has the designated resource
                                    (unit.cargo.get(resource) <= 0) or
                                    # Transferring - check that the receiving unit has cargo space
                                    (new_pos_unit.get_cargo_space_left() <= 0)
                            ):
                                available_actions_mask[unit_type][
                                    :,
                                    p_id,
                                    x,
                                    y,
                                    ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
                                ] = False
                    if unit.is_worker():
                        # Pillaging - check that worker is on a road tile and not on an allied city tile
                        if game_state.map.get_cell_by_pos(unit.pos).road <= 0 or \
                                pos_to_city_tile_dict[(unit.pos.x, unit.pos.y)] is not None:
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type]["PILLAGE"]
                            ] = False
                        # Building a city - check that worker has >= the required resources and is not on a resource
                        if not unit.can_build(game_state.map):
                            available_actions_mask[unit_type][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX[unit_type]["BUILD_CITY"]
                            ] = False
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        # No-op is always a legal action
                        # Research is a legal action whenever research_points < max_research
                        # Building a new unit is only a legal action when n_units < n_city_tiles
                        x, y = city_tile.pos.x, city_tile.pos.y
                        if player.research_points >= MAX_RESEARCH:
                            available_actions_mask["city_tile"][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX["city_tile"]["RESEARCH"]
                            ] = False
                        if len(player.units) >= player.city_tile_count:
                            available_actions_mask["city_tile"][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX["city_tile"]["BUILD_WORKER"]
                            ] = False
                            available_actions_mask["city_tile"][
                                :,
                                p_id,
                                x,
                                y,
                                ACTION_MEANINGS_TO_IDX["city_tile"]["BUILD_CART"]
                            ] = False
        return available_actions_mask

    @staticmethod
    def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
        out = {}
        for space, actions in actions_taken.items():
            out[space] = {
                ACTION_MEANINGS[space][i]: actions[..., i].sum()
                for i in range(actions.shape[-1])
            }
        return out


def get_unit_action(unit: Unit, action_idx: int, pos_to_unit_dict: Dict[Tuple, Optional[Unit]]) -> Optional[str]:
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
