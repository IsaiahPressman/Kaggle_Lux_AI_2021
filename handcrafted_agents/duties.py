from abc import ABC, abstractmethod
import numpy as np
from typing import List, NoReturn, Set

from .actions import Action
from .utility_constants import LOCAL_EVAL
from ..lux_ai.lux.constants import Constants
from ..lux_ai.lux.game_objects import City, Unit


class Duty(ABC):
    def __init__(
            self,
            units: Set[Unit],
            priority: float
    ):
        self.units = units
        self.priority = priority

    @abstractmethod
    def get_action_preferences(self, *args, **kwargs) -> dict[Unit, List[Action]]:
        pass

    def validate(self) -> NoReturn:
        validation_msg = self.get_validation_msg()
        if validation_msg:
            validation_msg = f"Duty {self.__class__.__name__} failed validation: {validation_msg}"
            if LOCAL_EVAL:
                raise RuntimeError(validation_msg)
            else:
                print(validation_msg)

    @abstractmethod
    def get_validation_msg(self) -> str:
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        pass


class MineFuelLocal(Duty):
    def __init__(self, target_city: City, target_fuel: int, *args, **kwargs):
        super(MineFuelLocal, self).__init__(*args, **kwargs)
        self.target_city = target_city
        self.target_fuel = target_fuel

    def get_action_preferences(
            self,
            resource_per_second_mat: np.ndarray,
            fuel_per_second_mat: np.ndarray
    ) -> dict[Unit, List[Action]]:
        action_preferences = {}
        for unit in self.units:
            if not unit.can_act():
                continue

            if unit.get_cargo_space_left() == 0:
                act_to_dist = {}
                for d in Constants.DIRECTIONS.astuple(move_only=False):
                    new_pos = unit.pos.translate(d, 1)
                    act_to_dist[d] = min(new_pos.distance_to(ct.pos) for ct in self.target_city.citytiles)
                action_preferences = [
                    Action(actor=unit, action_str=unit.move(d))
                    for d, _ in sorted(act_to_dist.items(), key=lambda item: item[1])
                ]
            else:
                raise NotImplementedError

        return action_preferences

    def get_validation_msg(self) -> str:
        if any((not u.is_worker()) for u in self.units):
            return f"Cart was found in self.units: {self.units}"
        return ""

    def is_complete(self) -> bool:
        return self.target_city.fuel >= self.target_fuel


"""
class MineFuelExpedition(Duty):
    pass
"""
"""
class ExpandCity(Duty):
    pass
"""
"""
class BuildNewCity(Duty):
    pass
"""
"""
class Deforest(Duty):
    pass
"""
