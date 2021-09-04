from typing import DefaultDict, Dict, List, Tuple, Set
from collections import defaultdict, deque
import heapq

import numpy as np

from .constants import Constants
from .game_map import GameMap, RESOURCE_TYPES
from .game_objects import Player, Unit, City, CityTile
from .game_position import Position
from .game_constants import GAME_CONSTANTS

INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
DISTANCE_TRANSITION_VALUE = 12


class Mission:
    def __init__(self, unit_id: str, target_position: Position, target_action: str = ""):
        self.target_position: Position = target_position
        self.target_action: str = target_action
        self.unit_id: str = unit_id
        self.delays: int = 0
        # [TODO] some expiry date for each mission

    def __str__(self):
        return " ".join([str(self.target_position), self.target_action])


class Missions(defaultdict):
    def __init__(self):
        self: DefaultDict[str, Mission] = defaultdict(Mission)

    def add(self, mission: Mission):
        self[mission.unit_id] = mission

    def cleanup(self, player: Player, player_city_tile_xy_set: Set[Tuple], opponent_city_tile_xy_set: Set[Tuple]):
        for unit_id in list(self.keys()):
            mission: Mission = self[unit_id]

            # if dead, delete from list
            if unit_id not in player.units_by_id:
                del self[unit_id]
                continue

            unit: Unit = player.units_by_id[unit_id]
            # if you want to build city without resource, delete from list
            if mission.target_action and mission.target_action[:5] == "bcity":
                if unit.cargo == 0:
                    del self[unit_id]
                    continue

            # if opponent has already built a base, reconsider your mission
            if tuple(mission.target_position) in opponent_city_tile_xy_set:
                del self[unit_id]
                continue

            # if you are in a base, reconsider your mission
            if tuple(unit.pos) in player_city_tile_xy_set:
                del self[unit_id]
                continue


    def __str__(self):
        return " ".join([unit_id + " " + str(x) for unit_id,x in self.items()])

    def get_targets(self):
        return [mission.target_position for unit_id, mission in self.items()]


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item, index=False):
        if item not in self.parent:
            if not index:
                return None
        return self._index(item)

    def _index(self, item):
        if item not in self.parent:
            self.parent[item] = item
            return item
        elif self.parent[item] == item:
            return item
        else:
            res = self.find(self.parent[item])
            self.parent[item] = res
            return res

    def union(self, set1, set2):
        root1 = self._index(set1)
        root2 = self._index(set2)
        self.parent[root1] = root2

    def get_groups(self):
        groups = defaultdict(list)
        for element in self.parent:
            leader = self.find(element)
            if leader:
                groups[leader].append(element)
        return groups.values()

    def get_group_count(self):
        return sum(len(group) > 1 for group in self.get_groups())


class Game:

    def _initialize(self, messages):
        """
        initialize state
        """
        self.player_id: int = int(messages[0])
        self.turn: int = -1
        # get some other necessary initial input
        mapInfo = messages[1].split(" ")
        self.map_width: int = int(mapInfo[0])
        self.map_height: int = int(mapInfo[1])
        self.map: GameMap = GameMap(self.map_width, self.map_height)
        self.players: List[Player] = [Player(0), Player(1)]

        self.targeted_xy_set: Set = set()
        self.targeted_leaders: Set = set()

        self.distance_transition_value = 8

    def _end_turn(self):
        print("D_FINISH")


    def _reset_player_states(self):
        self.players[0].units = []
        self.players[0].cities = {}
        self.players[0].city_tile_count = 0
        self.players[1].units = []
        self.players[1].cities = {}
        self.players[1].city_tile_count = 0

        self.player: Player = self.players[self.player_id]
        self.opponent: Player = self.players[1 - self.player_id]


    def _update(self, messages):
        """
        update state
        """
        self.map = GameMap(self.map_width, self.map_height)
        self.turn += 1
        self._reset_player_states()

        # [TODO] Use constants here
        self.night_turns_left = (360 - self.turn)//40 * 10 + min(10, (360 - self.turn)%40)

        self.turns_to_night = (30 - self.turn)%40
        self.turns_to_night = 0 if self.turns_to_night > 30 else self.turns_to_night

        self.turns_to_dawn = (40 - self.turn%40)
        self.turns_to_dawn = 0 if self.turns_to_dawn > 10 else self.turns_to_dawn

        self.is_day_time = self.turns_to_dawn == 0


        for update in messages:
            if update == "D_DONE":
                break
            strs = update.split(" ")
            input_identifier = strs[0]

            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strs[1])   # probably player_id
                self.players[team].research_points = int(strs[2])

            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strs[1]
                x = int(strs[2])
                y = int(strs[3])
                amt = int(float(strs[4]))
                self.map._setResource(r_type, x, y, amt)

            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unittype = int(strs[1])
                team = int(strs[2])
                unitid = strs[3]
                x = int(strs[4])
                y = int(strs[5])
                cooldown = float(strs[6])
                wood = int(strs[7])
                coal = int(strs[8])
                uranium = int(strs[9])
                unit = Unit(team, unittype, unitid, x, y, cooldown, wood, coal, uranium)
                self.players[team].units.append(unit)
                self.map.get_cell(x, y).unit = unit

            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strs[1])
                cityid = strs[2]
                fuel = float(strs[3])
                lightupkeep = float(strs[4])
                self.players[team].cities[cityid] = City(team, cityid, fuel, lightupkeep)

            elif input_identifier == INPUT_CONSTANTS.CITY_TILES:
                team = int(strs[1])
                cityid = strs[2]
                x = int(strs[3])
                y = int(strs[4])
                cooldown = float(strs[5])
                city = self.players[team].cities[cityid]
                citytile = city._add_city_tile(x, y, cooldown)
                self.map.get_cell(x, y).citytile = citytile
                self.players[team].city_tile_count += 1

            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strs[1])
                y = int(strs[2])
                road = float(strs[3])
                self.map.get_cell(x, y).road = road

        # update matrices
        self.calculate_matrix()
        self.calculate_resource_matrix()
        self.calculate_resource_groups()
        self.calculate_distance_matrix()

        # make indexes
        self.player.make_index_units_by_id()
        self.opponent.make_index_units_by_id()


    def init_zero_matrix(self, default_value=0):
        # [TODO] check if order of map_height and map_width is correct
        return np.full((self.map_height,self.map_width), default_value)


    def calculate_matrix(self):

        self.empty_tile_matrix = self.init_zero_matrix()

        self.wood_amount_matrix = self.init_zero_matrix()
        self.coal_amount_matrix = self.init_zero_matrix()
        self.uranium_amount_matrix = self.init_zero_matrix()
        self.all_resource_amount_matrix = self.init_zero_matrix()

        self.player_city_tile_matrix = self.init_zero_matrix()
        self.opponent_city_tile_matrix = self.init_zero_matrix()

        self.player_units_matrix = self.init_zero_matrix()
        self.opponent_units_matrix = self.init_zero_matrix()

        self.empty_tile_matrix = self.init_zero_matrix()

        for y in range(self.map_height):
            for x in range(self.map_width):
                cell = self.map.get_cell(x, y)

                is_empty = True

                if cell.unit:
                    is_empty = False
                    if cell.unit.team == self.player_id:
                        self.player_units_matrix[y,x] += 1
                    else:   # unit belongs to opponent
                        self.opponent_units_matrix[y,x] += 1

                if cell.has_resource():
                    is_empty = False
                    if cell.resource.type == RESOURCE_TYPES.WOOD:
                        self.wood_amount_matrix[y,x] += cell.resource.amount
                    if cell.resource.type == RESOURCE_TYPES.COAL:
                        self.coal_amount_matrix[y,x] += cell.resource.amount
                    if cell.resource.type == RESOURCE_TYPES.URANIUM:
                        self.uranium_amount_matrix[y,x] += cell.resource.amount
                    self.all_resource_amount_matrix[y,x] += cell.resource.amount

                elif cell.citytile:
                    is_empty = False
                    if cell.citytile.team == self.player_id:
                        self.player_city_tile_matrix[y,x] += 1
                    else:   # city tile belongs to opponent
                        self.opponent_city_tile_matrix[y,x] += 1

                if is_empty:
                    self.empty_tile_matrix[y,x] += 1

        self.convert_into_sets()


    def convert_into_sets(self):
        # or should we use dict?
        self.wood_amount_xy_set = set()
        self.coal_amount_xy_set = set()
        self.uranium_amount_xy_set = set()
        self.player_city_tile_xy_set = set()
        self.opponent_city_tile_xy_set = set()
        self.player_units_xy_set = set()
        self.opponent_units_xy_set = set()
        self.empty_tile_xy_set = set()

        for set_object, matrix in [
            [self.wood_amount_xy_set,           self.wood_amount_matrix],
            [self.coal_amount_xy_set,           self.coal_amount_matrix],
            [self.uranium_amount_xy_set,        self.uranium_amount_matrix],
            [self.player_city_tile_xy_set,      self.player_city_tile_matrix],
            [self.opponent_city_tile_xy_set,    self.opponent_city_tile_matrix],
            [self.player_units_xy_set,          self.player_units_matrix],
            [self.opponent_units_xy_set,        self.opponent_units_matrix],
            [self.empty_tile_xy_set,            self.empty_tile_matrix]]:

            for y in range(self.map.height):
                for x in range(self.map.width):
                    if matrix[y,x] > 0:
                        set_object.add((x,y))

        out_of_map = set()
        for y in [-1, self.map_height]:
            for x in range(self.map_width):
                out_of_map.add((x,y))
        for y in range(self.map_height):
            for x in [-1, self.map_width]:
                out_of_map.add((x,y))

        self.occupied_xy_set = (self.player_units_xy_set | self.opponent_units_xy_set | self.opponent_city_tile_xy_set | out_of_map) \
                                - self.player_city_tile_xy_set


    def calculate_distance_matrix(self, distance_transition_value=DISTANCE_TRANSITION_VALUE, blockade_multiplier_value=5):
        # calculate distance from resource (with fulfilled research requirements)
        visited = set()
        self.distance_from_resource = self.init_zero_matrix(self.map_height + self.map_width)
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.resource_rate_matrix[y,x] > 0:
                    visited.add((x,y))
                    self.distance_from_resource[y,x] = 0

        queue = deque(list(visited))
        while queue:
            x,y = queue.popleft()
            for dx,dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                xx, yy = x+dx, y+dy
                if (xx,yy) in visited:
                    continue
                if 0 <= xx < self.map_width and 0 <= yy < self.map_height:
                    self.distance_from_resource[yy,xx] = self.distance_from_resource[y,x] + 1
                    queue.append((xx,yy))
                    visited.add((xx,yy))

        # calculating the full matrix takes too much time
        self.positions_to_calculate_distances_from = set()
        for x,y in self.player_units_xy_set:
            self.positions_to_calculate_distances_from.add((x,y),)
            self.positions_to_calculate_distances_from.add((x+1,y),)
            self.positions_to_calculate_distances_from.add((x-1,y),)
            self.positions_to_calculate_distances_from.add((x,y+1),)
            self.positions_to_calculate_distances_from.add((x,y-1),)

        self.distance_matrix = np.full((self.map_height,self.map_width,self.map_height,self.map_width), 1001)

        for sy in range(self.map_height):
            for sx in range(self.map_width):
                if (sx,sy) not in self.positions_to_calculate_distances_from:
                    continue

                start_pos = (sx,sy)
                xy_processed = set()

                d4 = [(1,0),(0,1),(-1,0),(0,-1)]
                heap = [(0, start_pos),]
                while heap:
                    curdist, (x,y) = heapq.heappop(heap)
                    if (x,y) in xy_processed:
                        continue
                    xy_processed.add((x,y),)
                    self.distance_matrix[sy,sx,y,x] = curdist

                    for dx,dy in d4:
                        xx,yy = x+dx,y+dy
                        if not (0 <= xx < self.map_width and 0 <= yy < self.map_height):
                            continue
                        if (xx,yy) in xy_processed:
                            continue

                        # lazy_processing
                        if abs(sx-xx) + abs(sy-yy) > distance_transition_value:
                            continue

                        edge_length = 1
                        if (xx,yy) in self.occupied_xy_set:
                            edge_length = blockade_multiplier_value
                        heapq.heappush(heap, (curdist + edge_length, (xx,yy)))


    def retrieve_distance(self, sx, sy, ex, ey, distance_transition_value=DISTANCE_TRANSITION_VALUE, long_range_multiplier_value=5):

        if abs(sx-ex) + abs(sy-ey) > distance_transition_value:
            return (abs(sx-ex) + abs(sy-ey)) * long_range_multiplier_value

        if (sx, sy) not in self.positions_to_calculate_distances_from:
            return (abs(sx-ex) + abs(sy-ey)) * long_range_multiplier_value

        return self.distance_matrix[sy,sx,ey,ex]


    def convolve(self, matrix):
        new_matrix = matrix.copy()
        new_matrix[:-1,:] += matrix[1:,:]
        new_matrix[:,:-1] += matrix[:,1:]
        new_matrix[1:,:] += matrix[:-1,:]
        new_matrix[:,1:] += matrix[:,:-1]
        return new_matrix


    def calculate_resource_matrix(self):

        wood_fuel_rate = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"][RESOURCE_TYPES.WOOD.upper()]
        wood_count_rate = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"][RESOURCE_TYPES.COAL.upper()]
        # fuel - fuel amount if converted
        # count - how many remaining
        # rate - rate of extraction
        self.resource_fuel_matrix = self.wood_amount_matrix * wood_fuel_rate
        self.resource_count_matrix = (self.resource_fuel_matrix > 0) * wood_count_rate
        self.resource_rate_matrix = (self.resource_fuel_matrix > 0) * wood_fuel_rate * wood_count_rate

        if self.player.researched_coal():
            coal_fuel_rate = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"][RESOURCE_TYPES.COAL.upper()]
            coal_count_rate = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"][RESOURCE_TYPES.COAL.upper()]
            coal_fuel_matrix = self.coal_amount_matrix
            self.resource_fuel_matrix += coal_fuel_matrix * coal_fuel_rate
            self.resource_count_matrix += (coal_fuel_matrix > 0) * coal_count_rate
            self.resource_rate_matrix += (coal_fuel_matrix > 0) * coal_fuel_rate * coal_count_rate

        if self.player.researched_uranium():
            uranium_fuel_rate = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"][RESOURCE_TYPES.URANIUM.upper()]
            uranium_count_rate = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"][RESOURCE_TYPES.URANIUM.upper()]
            uranium_fuel_matrix = self.uranium_amount_matrix
            self.resource_fuel_matrix += uranium_fuel_matrix * uranium_fuel_rate
            self.resource_count_matrix += (uranium_fuel_matrix > 0) * uranium_count_rate
            self.resource_rate_matrix += (uranium_fuel_matrix > 0) * uranium_fuel_rate * uranium_count_rate

        # from the position
        self.convolved_fuel_matrix = self.convolve(self.resource_fuel_matrix)
        self.convolved_count_matrix = self.convolve(self.resource_count_matrix)
        self.convolved_rate_matrix = self.convolve(self.resource_rate_matrix)


    def calculate_resource_groups(self):
        self.xy_to_resource_group_id: DisjointSet = DisjointSet()
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self.resource_rate_matrix[y,x] > 0:
                    for dy,dx in [(1,0),(0,1),(-1,0),(0,-1)]:
                        xx, yy = x+dx, y+dy
                        if 0 <= yy < self.map_height and 0 <= xx < self.map_width:
                            self.xy_to_resource_group_id.union((x,y), (xx,yy))


    def repopulate_targets(self, missions: Missions):
        pos_list = missions.get_targets()
        self.targeted_leaders: Set = set(self.xy_to_resource_group_id.find(tuple(pos)) for pos in pos_list)
        self.targeted_xy_set: Set = set(tuple(pos) for pos in pos_list) - self.player_city_tile_xy_set

        self.resource_leader_to_locating_units: DefaultDict[Tuple, Set[str]] = defaultdict(set)
        self.resource_leader_to_targeting_units: DefaultDict[Tuple, Set[str]] = defaultdict(set)

        for unit_id in missions:

            unit: Unit = self.player.units_by_id[unit_id]
            current_position = tuple(unit.pos)
            leader = self.xy_to_resource_group_id.find(current_position)
            if leader:
                self.resource_leader_to_locating_units[leader].add(unit_id)

            mission: Mission = missions[unit_id]
            target_position = tuple(mission.target_position)
            leader = self.xy_to_resource_group_id.find(target_position)
            if leader:
                self.resource_leader_to_targeting_units[leader].add(unit_id)


    def calculate_dominance_matrix(self, feature_matrix, masking_factor = 0.5, exempted=(-1,-1)):
        # [TODO] marked for deletion
        mask = (1 - masking_factor * self.player_units_matrix)
        feature_matrix = self.convolve(feature_matrix)
        masked_matrix = mask * feature_matrix
        if exempted != (-1,-1):
            # the exempted cell is the position of the unit
            masked_matrix[exempted[0],exempted[1]] = feature_matrix[exempted[0],exempted[1]]
        return masked_matrix


    def get_nearest_empty_tile_and_distance(self, current_position: Position) -> Tuple[Position, int]:
        if self.all_resource_amount_matrix[current_position.y, current_position.x] == 0:
            if tuple(current_position) not in self.player_city_tile_xy_set:
                return current_position, 0

        width, height = self.map_width, self.map_height

        nearest_distance = width + height
        nearest_position: Position = None

        for y in range(height):
            for x in range(width):
                if self.empty_tile_matrix[y,x] == 0:  # not empty
                    continue

                position = Position(x, y)
                distance = position - current_position

                if self.distance_from_resource[y,x] != 1:
                    distance += 10

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_position = position

        return nearest_position, nearest_distance
