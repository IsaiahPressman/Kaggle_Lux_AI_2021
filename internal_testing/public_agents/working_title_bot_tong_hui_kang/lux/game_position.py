from lux import game
import random
from typing import List, Set, Tuple

from .constants import Constants

DIRECTIONS = Constants.DIRECTIONS


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, pos: 'Position') -> int:
        return abs(pos.x - self.x) + abs(pos.y - self.y)

    def distance_to(self, pos: 'Position'):
        """
        Returns Manhattan (L1/grid) distance to pos
        """
        return self - pos

    def is_adjacent(self, pos: 'Position'):
        return (self - pos) <= 1

    def __eq__(self, pos: 'Position') -> bool:
        return self.x == pos.x and self.y == pos.y

    def equals(self, pos: 'Position'):
        return self == pos

    def translate(self, direction, units) -> 'Position':
        if direction == DIRECTIONS.NORTH:
            return Position(self.x, self.y - units)
        elif direction == DIRECTIONS.EAST:
            return Position(self.x + units, self.y)
        elif direction == DIRECTIONS.SOUTH:
            return Position(self.x, self.y + units)
        elif direction == DIRECTIONS.WEST:
            return Position(self.x - units, self.y)
        elif direction == DIRECTIONS.CENTER:
            return Position(self.x, self.y)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __iter__(self):
        for i in (self.x, self.y):
            yield i
