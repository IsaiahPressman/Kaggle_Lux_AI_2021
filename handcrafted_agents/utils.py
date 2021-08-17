from typing import *

from ..lux_ai.lux.game import Player
from ..lux_ai.lux.game_objects import CityTile


def get_city_tiles(player: Player) -> List[CityTile]:
    return [ct for city in player.cities.values() for ct in city.citytiles]
