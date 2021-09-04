# functions executing the actions

import os, random, collections
import builtins as __builtin__
from typing import Tuple, Dict, Set, DefaultDict

from lux import game

from lux.game import Game, Player, Mission, Missions
from lux.game_map import Cell, RESOURCE_TYPES
from lux.game_objects import City, CityTile, Unit
from lux.game_position import Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS

from heuristics import *

DIRECTIONS = Constants.DIRECTIONS


def make_city_actions(game_state: Game, DEBUG=False) -> List[str]:
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    player = game_state.player

    units_cap = sum([len(x.citytiles) for x in player.cities.values()])
    units_cnt = len(player.units)  # current number of units

    actions: List[str] = []

    def do_research(city_tile: CityTile):
        action = city_tile.research()
        game_state.player.research_points += 1
        actions.append(action)

    def build_workers(city_tile: CityTile):
        nonlocal units_cnt
        action = city_tile.build_worker()
        actions.append(action)
        units_cnt += 1

    city_tiles: List[CityTile] = []
    for city in player.cities.values():
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)
    if not city_tiles:
        return []

    for city_tile in city_tiles[::-1]:
        if not city_tile.can_act():
            continue

        unit_limit_exceeded = (units_cnt >= units_cap)  # recompute every time

        if player.researched_uranium() and unit_limit_exceeded:
            print("skip city", city_tile.cityid, city_tile.pos.x, city_tile.pos.y)
            continue

        if not player.researched_uranium() and game_state.turns_to_night < 3:
            print("research and dont build units at night", city_tile.pos.x, city_tile.pos.y)
            do_research(city_tile)
            continue

        nearest_resource_distance = game_state.distance_from_resource[city_tile.pos.y, city_tile.pos.x]
        travel_range = game_state.turns_to_night // GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
        resource_in_travel_range = nearest_resource_distance < travel_range

        if resource_in_travel_range and not unit_limit_exceeded:
            print("build_worker", city_tile.cityid, city_tile.pos.x, city_tile.pos.y, nearest_resource_distance, travel_range)
            build_workers(city_tile)
            continue

        if not player.researched_uranium():
            print("research", city_tile.pos.x, city_tile.pos.y)
            do_research(city_tile)
            continue

        # otherwise don't do anything

    return actions


def make_unit_missions(game_state: Game, missions: Missions, DEBUG=False) -> Missions:
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    player = game_state.player
    missions.cleanup(player, game_state.player_city_tile_xy_set, game_state.opponent_city_tile_xy_set)  # remove dead units

    unit_ids_with_missions_assigned_this_turn = set()

    for distance_threshold in [0,1,2,3,4,10,22,30,100,1000,10**9+7]:
      for unit in player.units:
        # mission is planned regardless whether the unit can act

        if unit.id in unit_ids_with_missions_assigned_this_turn:
            continue

        # avoid sharing the same target
        game_state.repopulate_targets(missions)

        # if the unit is waiting for dawn at the side of resource
        stay_up_till_dawn = (unit.get_cargo_space_left() <= 4 and (not game_state.is_day_time or game_state.turn%40 == 0))
        # if the unit is full and it is going to be day the next few days
        # go to an empty tile and build a citytile
        # print(unit.id, unit.get_cargo_space_left())
        if unit.get_cargo_space_left() == 0 or stay_up_till_dawn:
            nearest_position, nearest_distance = game_state.get_nearest_empty_tile_and_distance(unit.pos)
            if stay_up_till_dawn or nearest_distance * 2 <= game_state.turns_to_night - 2:
                if unit.pos - nearest_position > distance_threshold:
                    continue
                print("plan mission to build citytile", unit.id, unit.pos, "->", nearest_position)
                mission = Mission(unit.id, nearest_position, unit.build_city())
                missions.add(mission)
                unit_ids_with_missions_assigned_this_turn.add(unit.id)
                continue

        if unit.id in missions:
            mission: Mission = missions[unit.id]
            if mission.target_position == unit.pos:
                # take action and not make missions if already at position
                continue

        if unit.id in missions:
            # the mission will be recaluated if the unit fails to make a move
            continue

        best_position, best_cell_value = find_best_cluster(game_state, unit, DEBUG=DEBUG)
        # [TODO] what if best_cell_value is zero
        if unit.pos - best_position > distance_threshold:
            continue
        print("plan mission adaptative", unit.id, unit.pos, "->", best_position)
        mission = Mission(unit.id, best_position, None)
        missions.add(mission)
        unit_ids_with_missions_assigned_this_turn.add(unit.id)

        # [TODO] when you can secure a city all the way to the end of time, do it

        # [TODO] just let units die perhaps

    return missions


def make_unit_actions(game_state: Game, missions: Missions, DEBUG=False) -> Tuple[Missions, List[str]]:
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    player, opponent = game_state.player, game_state.opponent
    actions = []

    units_with_mission_but_no_action = set(missions.keys())
    prev_actions_len = -1
    while prev_actions_len < len(actions):
      prev_actions_len = len(actions)

      for unit in player.units:
        if not unit.can_act():
            units_with_mission_but_no_action.discard(unit.id)
            continue

        # if there is no mission, continue
        if unit.id not in missions:
            units_with_mission_but_no_action.discard(unit.id)
            continue

        mission: Mission = missions[unit.id]

        print("attempting action for", unit.id, unit.pos)

        # if the location is reached, take action
        if unit.pos == mission.target_position:
            units_with_mission_but_no_action.discard(unit.id)
            print("location reached and make action", unit.id, unit.pos)
            action = mission.target_action

            # do not build city at last light
            if action and action[:5] == "bcity" and game_state.turn%40 == 30:
                del missions[unit.id]
                continue

            if action:
                actions.append(action)
            del missions[unit.id]
            continue

        # the unit will need to move
        direction = attempt_direction_to(game_state, unit, mission.target_position)
        if direction != "c":
            units_with_mission_but_no_action.discard(unit.id)
            action = unit.move(direction)
            print("make move", unit.id, unit.pos, direction)
            actions.append(action)
            continue

        # [TODO] make it possible for units to swap positions

    for unit_id in units_with_mission_but_no_action:
        mission: Mission = missions[unit_id]
        mission.delays += 1
        if mission.delays >= 1:
            del missions[unit_id]

    return missions, actions


def attempt_direction_to(game_state: Game, unit: Unit, target_pos: Position) -> DIRECTIONS:
    check_dirs = [
        DIRECTIONS.NORTH,
        DIRECTIONS.EAST,
        DIRECTIONS.SOUTH,
        DIRECTIONS.WEST,
    ]
    random.shuffle(check_dirs)
    closest_dist = 10**9+7
    closest_dir = DIRECTIONS.CENTER
    closest_pos = unit.pos

    for direction in check_dirs:
        newpos = unit.pos.translate(direction, 1)

        if tuple(newpos) in game_state.occupied_xy_set:
            continue

        # do not go into a city tile if you are carrying substantial wood
        if tuple(newpos) in game_state.player_city_tile_xy_set and unit.cargo.wood >= 60:
            continue

        dist = game_state.retrieve_distance(newpos.x, newpos.y, target_pos.x, target_pos.y)

        if dist < closest_dist:
            closest_dir = direction
            closest_dist = dist
            closest_pos = newpos

    if closest_dir != DIRECTIONS.CENTER:
        game_state.occupied_xy_set.discard(tuple(unit.pos))
        if tuple(closest_pos) not in game_state.player_city_tile_xy_set:
            game_state.occupied_xy_set.add(tuple(closest_pos))
        unit.cooldown += 2

    return closest_dir
