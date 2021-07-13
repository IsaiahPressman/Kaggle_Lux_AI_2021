import numpy as np
import math
from typing import *
import sys

if __package__ == "":
    # not sure how to fix this atm
    from lux.game import Game
    from lux.game_map import Cell
    from lux.constants import Constants
    from lux.game_constants import GAME_CONSTANTS
    from lux import annotate
else:
    from .lux.game import Game
    from .lux.game_map import Cell
    from .lux.constants import Constants
    from .lux.game_constants import GAME_CONSTANTS
    from .lux import annotate

DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES
AGENT = None


class RuleBasedAgent:
    def __init__(self, obs, conf):
        # Do not edit
        self.game_state = self.game_state = Game()
        self.game_state._initialize(obs["updates"])
        self.game_state._update(obs["updates"][2:])

        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]
        self.w, self.h = self.game_state.map.width, self.game_state.map.height

        self.my_cities_mat = np.zeros((self.w, self.h), dtype=bool)
        self.available_mat = np.ones((self.w, self.h), dtype=bool)

    def __call__(self, obs, conf) -> List[str]:
        self.preprocess(obs, conf)
        actions = []

        resource_tiles: List[Cell] = []
        for y in range(self.h):
            for x in range(self.w):
                cell = self.game_state.map.get_cell(x, y)
                if cell.has_resource():
                    resource_tiles.append(cell)
        
        cities_to_build = 0
        for k, city in self.me.cities.items():
            if city.fuel > city.get_light_upkeep() * GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"] + 1000:
                # if our city has enough fuel to survive the whole night and 1000 extra fuel,
                # lets increment citiesToBuild and let our workers know we have room for more city tiles
                cities_to_build += 1
            for citytile in city.citytiles:
                if citytile.can_act():
                    if self.me.city_tile_count > len(self.me.units):
                        actions.append(citytile.build_worker())
                    else:
                        actions.append(citytile.research())

        # we iterate over all our units and do something with them
        for unit in self.me.units:
            actions.append(annotate.sidetext(f"Turn: {self.game_state.turn}"))
            if self.game_state.turn % 5 == 0:
                actions.append(unit.move(DIRECTIONS.WEST))
            elif obs.player == 0:
                actions.append(unit.move(DIRECTIONS.CENTER))
            else:
                pass
            actions.append(annotate.text(unit.pos.x, unit.pos.y, f"Can act: {unit.can_act()}"))
            continue
            if unit.is_worker() and unit.can_act():
                closest_dist = math.inf
                closest_resource_tile = None
                if unit.get_cargo_space_left() > 0:
                    # if the unit is a worker and we have space in cargo,
                    # lets find the nearest resource tile and try to mine it
                    for resource_tile in resource_tiles:
                        if resource_tile.resource.type == RESOURCE_TYPES.COAL and not self.me.researched_coal():
                            continue
                        if resource_tile.resource.type == RESOURCE_TYPES.URANIUM and not self.me.researched_uranium():
                            continue
                        dist = resource_tile.pos.distance_to(unit.pos)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_resource_tile = resource_tile
                    if closest_resource_tile is not None:
                        actions.append(unit.move(unit.pos.direction_to(closest_resource_tile.pos)))
                else:
                    # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                    if len(self.me.cities) > 0:
                        closest_dist = math.inf
                        closest_city_tile = None
                        for k, city in self.me.cities.items():
                            for city_tile in city.citytiles:
                                dist = city_tile.pos.distance_to(unit.pos)
                                if dist < closest_dist:
                                    closest_dist = dist
                                    closest_city_tile = city_tile
                        if closest_city_tile is not None:
                            move_dir = unit.pos.direction_to(closest_city_tile.pos)
                            # here we consider building city tiles provided we are adjacent to a city tile and we can build
                            if (cities_to_build > 0 and
                                    unit.pos.is_adjacent(closest_city_tile.pos) and
                                    unit.can_build(self.game_state.map)):
                                actions.append(unit.build_city())
                            else:
                                actions.append(unit.move(move_dir))

        """
        # you can add debug annotations using the functions in the annotate object
        i = game_state.turn + observation.player
        actions.append(annotate.circle(i % width, i % height))
        i += 1
        actions.append(annotate.x(i % width, i % height))
        i += 1
        actions.append(annotate.line(i % width, i % height, (i + 3) % width, (i + 3) % height))
        actions.append(annotate.text(0, 1, f"{game_state.turn}_Text!"))
        actions.append(annotate.sidetext(f"Research points: {player.research_points}"))
        """

        return actions

    def preprocess(self, obs, conf) -> NoReturn:
        # Do not edit
        self.game_state._update(obs["updates"])

        # Code starts here
        # These lines are probably unnecessary
        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]

        self.my_cities_mat[:] = False
        for city_tile in [c for city in self.me.cities.values() for c in city.citytiles]:
            self.my_cities_mat[city_tile.pos.x, city_tile.pos.y] = True

        self.available_mat[:] = True
        for unit in self.me.units:
            if not unit.can_act():
                self.available_mat[unit.pos.x, unit.pos.y] = False


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = RuleBasedAgent(obs, conf)
    return AGENT(obs, conf)
