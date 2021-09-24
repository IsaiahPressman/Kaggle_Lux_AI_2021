from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, NoReturn, Set

from .actions import Action
from ..utils import in_bounds, DEBUG_MESSAGE, RUNTIME_ASSERT
from ..utility_constants import COLLECTION_RATES
from ..lux import annotate
from ..lux.constants import Constants
from ..lux.game_objects import City, Unit


class Duty(ABC):
    def __init__(
            self,
            units: Set[Unit],
            priority: float
    ):
        self.units = units
        self.priority = priority

    @abstractmethod
    def get_action_preferences(self, **kwargs) -> Dict[Unit, List[Action]]:
        pass

    def validate(self) -> NoReturn:
        validation_msg = self.get_validation_msg()
        RUNTIME_ASSERT(validation_msg is None, f"Duty {self.__class__.__name__} failed validation: {validation_msg}")

    @abstractmethod
    def get_validation_msg(self) -> Optional[str]:
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
            *,
            available_mat: np.ndarray,
            smoothed_rps_mats: List[np.ndarray],
            smoothed_fps_mats: List[np.ndarray],
    ) -> Dict[Unit, List[Action]]:
        action_preferences = {}
        for unit in self.units:
            if not unit.can_act():
                continue

            act_to_dist = {}
            for d in Constants.DIRECTIONS.astuple(include_center=True):
                new_pos = unit.pos.translate(d, 1)
                if in_bounds(new_pos, available_mat.shape) and available_mat[new_pos.x, new_pos.y]:
                    act_to_dist[d] = min(new_pos.distance_to(ct.pos) for ct in self.target_city.citytiles)
            act_to_fps = {}
            for d in Constants.DIRECTIONS.astuple(include_center=True):
                new_pos = unit.pos.translate(d, 1)
                if (
                        in_bounds(new_pos, available_mat.shape) and
                        available_mat[new_pos.x, new_pos.y] and
                        # Workers are not allowed to stray too far from the city for local mining
                        act_to_dist[d] <= 8
                ):
                    act_to_fps[d] = smoothed_fps_mats[0][new_pos.x, new_pos.y]

            # TODO: high priority - change to account for fuel when next taking action + dist_to_city
            cargo_space_if_noop = unit.get_cargo_space_left() - 2 * smoothed_rps_mats[0][unit.pos.x, unit.pos.y]
            # If cargo space <= minimum collection rate, we return to the city
            if unit.get_cargo_space_left() <= min(COLLECTION_RATES.values()):
                action_preferences[unit] = [
                    Action(
                        actor=unit,
                        action_str=unit.move(d),
                        debug_strs=annotate.text(
                            unit.pos.translate(d, 1).x,
                            unit.pos.translate(d, 1).y,
                            f"{self.__class__.__name__} {i/len(act_to_dist)}: {self.target_city}/{self.target_fuel:.0f}"
                        )
                    )
                    # Smaller distances are better, so reverse=False (default)
                    for i, (d, _) in enumerate(sorted(
                        act_to_dist.items(),
                        # Tie-break towards no-op
                        key=lambda item: (item[1], (item[0] == Constants.DIRECTIONS.CENTER) * -1.),
                    ))
                ]
            # If cargo space - 2 * a square equal or closer to the city
            else:
                # Iteratively work through the smoothed fps matrices until all directions have a non-zero fuel value
                for fps_mat in smoothed_fps_mats[1:]:
                    for d in Constants.DIRECTIONS.astuple(include_center=True):
                        # Stop if value is already non-zero
                        if act_to_fps.get(d, 0) > 0:
                            continue

                        new_pos = unit.pos.translate(d, 1)
                        if (
                                in_bounds(new_pos, available_mat.shape) and
                                available_mat[new_pos.x, new_pos.y] and
                                # Workers are not allowed to stray too far from the city for local mining
                                act_to_dist[d] <= 8
                        ):
                            act_to_fps[d] = fps_mat[new_pos.x, new_pos.y]
                    # Stop if the available actions all have a non-zero value
                    if all(v > 0 for v in act_to_fps.values()):
                        break
                action_preferences[unit] = [
                    Action(
                        actor=unit,
                        action_str=unit.move(d),
                        debug_strs=annotate.text(
                            unit.pos.translate(d, 1).x,
                            unit.pos.translate(d, 1).y,
                            f"{self.__class__.__name__} {i}/{len(act_to_fps)}: {fuel:.0f}"
                        )
                    )
                    # More fuel is better, so reverse=True
                    for i, (d, fuel) in enumerate(sorted(
                        act_to_fps.items(),
                        # Tie-break towards no-op
                        key=lambda item: (item[1], (item[0] == Constants.DIRECTIONS.CENTER) * 1.),
                        reverse=True
                    ))
                ]

        return action_preferences

    def get_validation_msg(self) -> Optional[str]:
        if any((not u.is_worker()) for u in self.units):
            return f"Cart was found in self.units: {self.units}"
        return None

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
