import itertools
import numpy as np
import math
from typing import *

from . import map_processing, utils
from .actions import Action
from .utility_constants import LOCAL_EVAL, DAY_LEN, NIGHT_LEN, DN_CYCLE_LEN, MAX_RESEARCH
from ..lux_ai.lux.constants import Constants
from ..lux_ai.lux.game import Game
from ..lux_ai.lux.game_constants import GAME_CONSTANTS
from ..lux_ai.lux.game_objects import CityTile, Unit, Player
from ..lux_ai.lux.game_map import Cell
from ..lux_ai.lux import annotate


AGENT = None


class Agent:
    # noinspection PyProtectedMember
    def __init__(self, obs, conf):
        # Do not edit
        self.game_state = self.game_state = Game()
        self.game_state._initialize(obs["updates"])
        self.game_state._update(obs["updates"][2:])

        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]
        self.w, self.h = self.game_state.map.width, self.game_state.map.height

        self.all_cities_mat = np.zeros((2, self.w, self.h), dtype=bool)
        self.all_mobile_units_mat = np.zeros_like(self.all_cities_mat)
        self.all_immobile_units_mat = np.zeros_like(self.all_cities_mat)

        self.road_mat = np.zeros((self.w, self.h), dtype=float)

        self.wood_mat = np.zeros((self.w, self.h), dtype=float)
        self.coal_mat = np.zeros_like(self.wood_mat)
        self.uranium_mat = np.zeros_like(self.wood_mat)

        self.city_tile_actions: Dict[CityTile, List[Action]] = {}
        self.worker_actions: Dict[Unit, List[Action]] = {}
        self.cart_actions: Dict[Unit, List[Action]] = {}
        self.debug_actions: List[Action] = []

        self.resource_per_second_mat = np.zeros((self.w, self.h), dtype=float)
        self.fuel_per_second_mat = np.zeros_like(self.resource_per_second_mat)

    def __call__(self, obs, conf) -> List[str]:
        self.reset(obs, conf)
        self.preprocess()

        self.update_strategy()
        self.assign_city_tile_actions()
        self.assign_unit_duties()
        self.assign_unit_actions()

        self.debug_actions.append(Action(actor=self.me, action_str=annotate.sidetext(f"Turn: {self.game_state.turn}")))
        return self.get_actions()

    # noinspection PyProtectedMember
    def reset(self, obs, conf) -> NoReturn:
        # Do not edit
        self.game_state._update(obs["updates"])

        # Code starts here
        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]

        # Reset actions
        self.city_tile_actions = {}
        self.worker_actions = {}
        self.cart_actions = {}
        self.debug_actions = []

        # Fill the city and unit matrices
        self.all_cities_mat[:] = False
        self.all_mobile_units_mat[:] = False
        self.all_immobile_units_mat[:] = False
        for player in self.game_state.players:
            p_id = player.team
            for city_tile in utils.get_city_tiles(player):
                self.all_cities_mat[p_id, city_tile.pos.x, city_tile.pos.y] = True

            for unit in player.units:
                if unit.can_act():
                    mat = self.all_mobile_units_mat
                else:
                    mat = self.all_immobile_units_mat
                mat[p_id, unit.pos.x, unit.pos.y] = True

        # Fill the road and resource matrices
        for cell in itertools.chain(*self.game_state.map.map):
            x, y = cell.pos.x, cell.pos.y
            self.road_mat[x, y] = cell.road
            if cell.has_resource():
                if cell.resource.type == Constants.RESOURCE_TYPES.WOOD:
                    self.wood_mat[x, y] = cell.resource.amount
                elif cell.resource.type == Constants.RESOURCE_TYPES.COAL:
                    self.coal_mat[x, y] = cell.resource.amount
                elif cell.resource.type == Constants.RESOURCE_TYPES.URANIUM:
                    self.uranium_mat[x, y] = cell.resource.amount
                else:
                    raise ValueError(f"Unrecognized resource type: {cell.resource.type}")

    def preprocess(self) -> NoReturn:
        pass

    def update_strategy(self) -> NoReturn:
        pass

    def assign_city_tile_actions(self) -> NoReturn:
        units_to_build = max(self.me.city_tile_count - len(self.me.units), 0)
        research_remaining = max(MAX_RESEARCH - self.me.research_points, 0)

        for city_tile in utils.get_city_tiles(self.me):
            if not city_tile.can_act():
                continue

            # TODO: medium priority - refine which city tiles build units instead of just the first one in the list
            if units_to_build > 0:
                # TODO: medium priority - refine when to build carts
                self.add_action(Action(actor=city_tile, action_str=city_tile.build_worker()))
                units_to_build -= 1
            elif research_remaining > 0:
                # TODO: low priority - refine when to No-op instead of research
                # TODO: very low priority - is it ever correct to research instead of building a unit?
                self.add_action(Action(actor=city_tile, action_str=city_tile.research()))
                research_remaining -= 1
            elif city_tile not in self.city_tile_actions:
                self.city_tile_actions[city_tile] = []

    def assign_unit_duties(self) -> NoReturn:


    def assign_unit_actions(self) -> NoReturn:
        for unit in self.me.units:
            raise NotImplementedError

    # Utility functions and properties here:
    def add_action(self, action: Action) -> NoReturn:
        if isinstance(action.actor, CityTile):
            action_list = self.city_tile_actions.setdefault(action.actor, [])
        elif isinstance(action.actor, Unit):
            if action.actor.is_worker():
                action_list = self.worker_actions.setdefault(action.actor, [])
            elif action.actor.is_cart():
                action_list = self.cart_actions.setdefault(action.actor, [])
            else:
                raise NotImplementedError(f"Unrecognized unit: {action.actor}")
        else:
            raise NotImplementedError(f"Unrecognized actor: {action.actor}")
        action_list.append(action)

    def get_actions(self) -> List[str]:
        actions = []
        for actor, act_list in dict(
                **self.city_tile_actions,
                **self.worker_actions,
                **self.cart_actions
        ).items():
            if act_list:
                actions.append(act_list[0].action_str)
            else:
                self.debug_actions.append(Action(
                    actor=actor,
                    action_str=annotate.text(actor.pos.x, actor.pos.y, f"{actor}: NO-OP")
                ))

        if LOCAL_EVAL:
            actions.extend([a.action_str for a in self.debug_actions])

        return actions

    @property
    def my_cities_mat(self) -> np.ndarray:
        return self.all_cities_mat[self.me.team, ...]

    @property
    def opp_cities_mat(self) -> np.ndarray:
        return self.all_cities_mat[self.opp.team, ...]

    @property
    def my_mobile_units_mat(self) -> np.ndarray:
        return self.all_mobile_units_mat[self.me.team, ...]

    @property
    def opp_mobile_units_mat(self) -> np.ndarray:
        return self.all_mobile_units_mat[self.opp.team, ...]

    @property
    def my_immobile_units_mat(self) -> np.ndarray:
        return self.all_immobile_units_mat[self.me.team, ...]

    @property
    def opp_immobile_units_mat(self) -> np.ndarray:
        return self.all_immobile_units_mat[self.opp.team, ...]

    @property
    def my_available_mat(self) -> np.ndarray:
        return ~self.all_immobile_units_mat.any(axis=0) | self.my_cities_mat

    @property
    def opp_available_mat(self) -> np.ndarray:
        return ~self.all_immobile_units_mat.any(axis=0) | self.opp_cities_mat

    @property
    def turns_until_night(self) -> int:
        return max(DAY_LEN - (self.game_state.turn % DN_CYCLE_LEN), 0)

    @property
    def turns_until_day(self) -> int:
        if not self.is_night:
            return 0
        return max(NIGHT_LEN - (self.game_state.turn % DN_CYCLE_LEN - 30), 0)

    @property
    def is_night(self) -> bool:
        return self.game_state.turn % DN_CYCLE_LEN >= DAY_LEN


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = Agent(obs, conf)
    return AGENT(obs, conf)
