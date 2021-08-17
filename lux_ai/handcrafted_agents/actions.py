from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from ..lux.game_objects import CityTile, Unit, Player


@dataclass
class Action:
    actor: Optional[Union[CityTile, Unit, Player]]
    action_str: str
    debug_strs: Optional[Union[str, Sequence[str]]] = None

    @property
    def action_type(self):
        raise NotImplementedError

    def get_debug_strs(self) -> List[str]:
        if self.debug_strs:
            if not isinstance(self.debug_strs, str) and isinstance(self.debug_strs, Sequence):
                return [s for s in self.debug_strs]
            return [self.debug_strs]
        return []


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