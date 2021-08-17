from dataclasses import dataclass
from typing import Optional, Union

from ..lux_ai.lux.game_objects import CityTile, Unit, Player
from ..lux_ai.lux.game_map import Position


@dataclass
class Action:
    actor: Optional[Union[CityTile, Unit, Player]]
    action_str: str

    @property
    def action_type(self):
        raise NotImplementedError


"""
class Action:
    def __init__(self, actor: Union[CityTile, Unit], action_str: str):
        self.actor = actor
        self.action_str = action_str
        self.affected_pos: Optional[Position] = None
    
    @property
    def type(self) -> str:
        return self.__class__.__name__


class CityTileAction(Action):
    pass


class BuildWorker(CityTileAction):
    def __init__(self, actor: Union[CityTile, Unit], action_str: str):
        super(BuildWorker, self).__init__(actor, action_str)
"""