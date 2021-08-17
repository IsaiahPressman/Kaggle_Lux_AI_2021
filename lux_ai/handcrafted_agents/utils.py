import sys
from typing import List, NoReturn, Tuple

from .utility_constants import LOCAL_EVAL
from ..lux.game import Player
from ..lux.game_map import Position
from ..lux.game_objects import CityTile


def get_city_tiles(player: Player) -> List[CityTile]:
    return [ct for city in player.cities.values() for ct in city.citytiles]


def in_bounds(pos: Position, board_dims: Tuple[int, int]) -> bool:
    return 0 <= pos.x < board_dims[0] and 0 <= pos.y < board_dims[1]


def DEBUG_MESSAGE(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)


def RUNTIME_ASSERT(statement: bool, msg: str = "") -> NoReturn:
    if statement:
        return

    if LOCAL_EVAL:
        raise RuntimeError(msg)
    else:
        DEBUG_MESSAGE(msg)
