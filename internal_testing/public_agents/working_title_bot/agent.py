import os
import pickle

import numpy as np
import builtins as __builtin__

from lux.game import Game, Mission, Missions
import lux.annotate as annotate

from actions import *
from heuristics import *
from typing import DefaultDict

game_state = Game()
missions = Missions()


def game_logic(game_state: Game, missions: Missions, DEBUG=False):
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    actions_by_cities = make_city_actions(game_state, DEBUG=DEBUG)
    missions = make_unit_missions(game_state, missions, DEBUG=DEBUG)
    mission_annotations = print_and_annotate_missions(game_state, missions)
    missions, actions_by_units = make_unit_actions(game_state, missions, DEBUG=DEBUG)
    movement_annotations = annotate_movements(game_state, actions_by_units)

    print("actions_by_cities", actions_by_cities)
    print("actions_by_units", actions_by_units)
    print("mission_annotations", mission_annotations)
    print("movement_annotations", movement_annotations)
    actions = actions_by_cities + actions_by_units + mission_annotations + movement_annotations
    return actions, game_state, missions


def print_game_state(game_state: Game, DEBUG=False):
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    print("Turn number: ", game_state.turn)
    print("Citytile count: ", game_state.player.city_tile_count)
    print("Unit count: ", len(game_state.player.units))

    # you can also read the pickled game_state and print its attributes
    return


def print_and_annotate_missions(game_state: Game, missions: Missions, DEBUG=False):
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    print("Missions")
    print(missions)
    # you can also read the pickled missions and print its attributes

    annotations: List[str] = []
    player: Player = game_state.player

    for unit_id, mission in missions.items():
        mission: Mission = mission
        unit: Unit = player.units_by_id[unit_id]

        annotation = annotate.line(unit.pos.x, unit.pos.y, mission.target_position.x, mission.target_position.y)
        annotations.append(annotation)

        if mission.target_action and mission.target_action.split(" ")[0] == "bcity":
            annotation = annotate.circle(mission.target_position.x, mission.target_position.y)
            annotations.append(annotation)
        else:
            annotation = annotate.x(mission.target_position.x, mission.target_position.y)
            annotations.append(annotation)

    annotation = annotate.sidetext("U:{} C:{} L:{}/{}".format(len(game_state.player.units),
                                                              len(game_state.player_city_tile_xy_set),
                                                              len(game_state.targeted_leaders),
                                                              game_state.xy_to_resource_group_id.get_group_count()))
    annotations.append(annotation)

    return annotations


def annotate_movements(game_state: Game, actions_by_units: List[str]):
    annotations = []
    dirs = [
        DIRECTIONS.NORTH,
        DIRECTIONS.EAST,
        DIRECTIONS.SOUTH,
        DIRECTIONS.WEST,
        DIRECTIONS.CENTER
    ]
    d5 = [(0,-1), (1,0), (0,1), (-1,0), (0,0)]

    for action_by_units in actions_by_units:
        if action_by_units[:2] != "m ":
            continue
        unit_id, dir = action_by_units.split(" ")[1:]
        unit = game_state.player.units_by_id[unit_id]
        x, y = unit.pos.x, unit.pos.y
        dx, dy = d5[dirs.index(dir)]
        annotation = annotate.line(x, y, x+dx, y+dy)
        annotations.append(annotation)

    return annotations


def agent(observation, configuration, DEBUG=False):
    if DEBUG: print = __builtin__.print
    else: print = lambda *args: None

    del configuration  # unused
    global game_state, missions

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state.player_id = observation.player
        game_state._update(observation["updates"][2:])
    else:
        # actually rebuilt and recomputed from scratch
        game_state._update(observation["updates"])

    if not os.environ.get('GFOOTBALL_DATA_DIR', ''):  # on Kaggle compete, do not save items
        str_step = str(observation["step"]).zfill(3)
        with open('snapshots/observation-{}.pkl'.format(str_step), 'wb') as handle:
            pickle.dump(observation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('snapshots/game_state-{}.pkl'.format(str_step), 'wb') as handle:
            pickle.dump(game_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('snapshots/missions-{}.pkl'.format(str_step), 'wb') as handle:
            pickle.dump(missions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    actions, game_state, missions = game_logic(game_state, missions)
    return actions
