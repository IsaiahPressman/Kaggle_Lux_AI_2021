import copy
import itertools
import numpy as np
from typing import *

from . import map_processing, duties
from .actions import Action
from .. import utils
from ..utility_constants import LOCAL_EVAL, DAY_LEN, NIGHT_LEN, DN_CYCLE_LEN, MAX_RESEARCH
from ..lux.constants import Constants
from ..lux.game import Game
from ..lux.game_objects import CityTile, Unit
from ..lux import annotate


AGENT = None


class Agent:
    # noinspection PyProtectedMember
    def __init__(self, obs, conf):
        # Do not edit
        self.game_state = self.game_state = Game()
        self.game_state._initialize(obs["updates"])

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

        self.duties: List[duties.Duty] = []
        self.city_tile_actions: Dict[CityTile, List[Action]] = {}
        self.worker_actions: Dict[Unit, List[Action]] = {}
        self.cart_actions: Dict[Unit, List[Action]] = {}
        self.debug_actions: List[str] = []

        self.resource_per_second_mat = np.zeros((self.w, self.h), dtype=float)
        self.fuel_per_second_mat = np.zeros_like(self.resource_per_second_mat)
        self.smoothed_rps_mats: List[np.ndarray] = []
        self.smoothed_fps_mats: List[np.ndarray] = []

    def __call__(self, obs, conf) -> List[str]:
        self.reset(obs, conf)
        self.preprocess()

        self.update_strategy()
        self.assign_unit_duties()
        # TODO: low priority - assign_city_tile_duties?
        self.assign_city_tile_actions()
        self.assign_unit_actions()
        self.resolve_actions()

        self.debug_actions.append(annotate.sidetext(f"Turn: {self.game_state.turn}"))
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
            for city_tile in player.city_tiles:
                self.all_cities_mat[p_id, city_tile.pos.x, city_tile.pos.y] = True

            for unit in player.units:
                if unit.can_act():
                    mat = self.all_mobile_units_mat
                else:
                    mat = self.all_immobile_units_mat
                mat[p_id, unit.pos.x, unit.pos.y] = True

        # Fill the road and resource matrices
        self.road_mat[:] = 0
        self.wood_mat[:] = 0
        self.coal_mat[:] = 0
        self.uranium_mat[:] = 0
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

        # Clear other matrices
        self.resource_per_second_mat[:] = 0
        self.fuel_per_second_mat[:] = 0
        self.smoothed_rps_mats = []
        self.smoothed_fps_mats = []

    def preprocess(self) -> NoReturn:
        # TODO: low priority - tune and hyperpameterize time_horizon
        time_horizon = 2
        # TODO: low priority - tune and hyperpameterize n_iter_smoothing
        n_iter_smoothing = 5

        self.resource_per_second_mat += map_processing.get_resource_per_second_mat(
            self.wood_mat,
            Constants.RESOURCE_TYPES.WOOD,
            time_horizon=time_horizon
        )
        # TODO: low priority - add coal/uranium to rps/fps mats when research is near completion?
        if self.me.researched_coal():
            self.resource_per_second_mat += map_processing.get_resource_per_second_mat(
                self.coal_mat,
                Constants.RESOURCE_TYPES.COAL,
                time_horizon=time_horizon
            )
        if self.me.researched_uranium():
            self.resource_per_second_mat += map_processing.get_resource_per_second_mat(
                self.uranium_mat,
                Constants.RESOURCE_TYPES.URANIUM,
                time_horizon=time_horizon
            )
        self.smoothed_rps_mats = [self.resource_per_second_mat]
        for _ in range(n_iter_smoothing):
            self.smoothed_rps_mats.append(map_processing.smooth_mining_heatmap(
                self.smoothed_rps_mats[-1]
            ))

        self.fuel_per_second_mat += map_processing.get_fuel_per_second_mat(
            self.wood_mat,
            Constants.RESOURCE_TYPES.WOOD,
            time_horizon=time_horizon
        )
        if self.me.researched_coal():
            self.fuel_per_second_mat += map_processing.get_fuel_per_second_mat(
                self.coal_mat,
                Constants.RESOURCE_TYPES.COAL,
                time_horizon=time_horizon
            )
        if self.me.researched_uranium():
            self.fuel_per_second_mat += map_processing.get_fuel_per_second_mat(
                self.uranium_mat,
                Constants.RESOURCE_TYPES.URANIUM,
                time_horizon=time_horizon
            )
        self.smoothed_fps_mats = [self.fuel_per_second_mat]
        for _ in range(n_iter_smoothing):
            self.smoothed_fps_mats.append(map_processing.smooth_mining_heatmap(
                self.smoothed_fps_mats[-1]
            ))

    def update_strategy(self) -> NoReturn:
        pass

    def assign_city_tile_actions(self) -> NoReturn:
        units_to_build = max(self.me.city_tile_count - len(self.me.units), 0)
        research_remaining = max(MAX_RESEARCH - self.me.research_points, 0)

        for city_tile in self.me.city_tiles:
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
        # First clean up all completed duties
        for duty in copy.copy(self.duties):
            if duty.is_complete():
                self.duties.remove(duty)
        # Then check for unassigned units
        unassigned_units = set(self.me.units)
        for unit in self.me.units:
            for duty in self.duties:
                # Replace the unit instance with the new unit instance with the updated unit position
                if unit in duty.units:
                    duty.units.remove(unit)
                    duty.units.add(unit)
                    unassigned_units.remove(unit)
        # Create new duties for unassigned units or assign them to existing ones
        if len(unassigned_units) > 0:
            self.duties.append(duties.MineFuelLocal(
                    units=self.me.units,
                    priority=0.,
                    target_city=list(self.me.cities.values())[0],  # TODO
                    target_fuel=100_000,  # TODO
                ))

        # Finally, perform checks that all units are assigned, and that all duties have the requisite units
        all_units = set(self.me.units)
        assigned_units = set(u for d in self.duties for u in d.units)
        utils.RUNTIME_ASSERT(
            all_units == assigned_units,
            f"Not all units were assigned a Duty: {all_units.symmetric_difference(assigned_units)}"
        )
        for duty in self.duties:
            duty.validate()

    def assign_unit_actions(self) -> NoReturn:
        for duty in self.duties:
            for act_prefs in duty.get_action_preferences(
                    available_mat=self.my_available_mat,
                    smoothed_rps_mats=self.smoothed_rps_mats,
                    smoothed_fps_mats=self.smoothed_fps_mats,
            ).values():
                for act in act_prefs:
                    self.add_action(act)

    def resolve_actions(self) -> NoReturn:
        # TODO: high priority
        pass

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
        for actor, act_list in {
                **self.city_tile_actions,
                **self.worker_actions,
                **self.cart_actions
        }.items():
            if act_list:
                actions.append(act_list[0].action_str)
                self.debug_actions.extend(act_list[0].get_debug_strs())
            else:
                self.debug_actions.append(annotate.text(actor.pos.x, actor.pos.y, f"{actor}: NO-OP"))

        if LOCAL_EVAL:
            actions.extend(self.debug_actions)

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
        return self.my_cities_mat | (~(self.all_immobile_units_mat.any(axis=0)) & ~self.opp_cities_mat)

    @property
    def opp_available_mat(self) -> np.ndarray:
        return self.opp_cities_mat | (~(self.all_immobile_units_mat.any(axis=0)) & ~self.my_cities_mat)

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

    # Helper functions for debugging
    def set_to_turn(self, obs, conf, turn: int) -> NoReturn:
        self.game_state.turn = turn - 1
        return self(obs, conf)


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = Agent(obs, conf)
    return AGENT(obs, conf)
