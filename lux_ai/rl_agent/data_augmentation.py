from abc import ABC, abstractmethod
import torch
from typing import Dict

from ..lux.constants import Constants
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX

DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=False)


class DataAugmenter(ABC):
    def __init__(self):
        directions_mapped_forward = self.get_directions_mapped()
        direction_mapped_inverse = {val: key for key, val in directions_mapped_forward.items()}
        assert len(directions_mapped_forward) == len(direction_mapped_inverse)

        self.transformed_action_idxs_forward = {}
        self.transformed_action_idxs_inverse = {}
        for space, meanings_to_idx in ACTION_MEANINGS_TO_IDX.items():
            transformed_space_idxs_forward = []
            for action, idx in meanings_to_idx.items():
                for d, d_mapped in directions_mapped_forward:
                    if action.endswith(f"_{d}"):
                        transformed_space_idxs_forward.append(meanings_to_idx[action[:-1] + d_mapped])
                        break
                else:
                    transformed_space_idxs_forward.append(idx)
            self.transformed_action_idxs_forward[space] = transformed_space_idxs_forward

            transformed_space_idxs_inverse = []
            for action, idx in meanings_to_idx.items():
                for d, d_mapped in direction_mapped_inverse:
                    if action.endswith(f"_{d}"):
                        transformed_space_idxs_inverse.append(meanings_to_idx[action[:-1] + d_mapped])
                        break
                else:
                    transformed_space_idxs_inverse.append(idx)
            self.transformed_action_idxs_inverse[space] = transformed_space_idxs_inverse

    def apply(self, x: Dict[str, torch.Tensor], inverse: bool, is_policy: bool) -> Dict[str, torch.Tensor]:
        x_transformed = {
            key: self._op(val, inverse=inverse, is_policy=is_policy)
            for key, val in x.items()
        }
        if is_policy:
            return self._transform_policy(x_transformed, inverse=inverse)
        return x_transformed

    @abstractmethod
    def get_directions_mapped(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def _op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        pass

    def _transform_policy(self, policy: Dict[str, torch.Tensor], inverse: bool) -> Dict[str, torch.Tensor]:
        if inverse:
            return {
                space: p[..., self.transformed_action_idxs_inverse[space]]
                for space, p in policy.items()
            }
        else:
            return {
                space: p[..., self.transformed_action_idxs_forward[space]]
                for space, p in policy.items()
            }


class VerticalFlip(DataAugmenter):
    def get_directions_mapped(self) -> Dict[str, str]:
        # Switch all N/S actions
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.EAST,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.WEST
        }

    def _op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-3,) if is_policy else (-2,)
        return torch.flip(t, dims=dims)


class HorizontalFlip(DataAugmenter):
    def get_directions_mapped(self) -> Dict[str, str]:
        # Switch all E/W actions
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.EAST
        }

    def _op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-2,) if is_policy else (-1,)
        return torch.flip(t, dims=dims)


class Rot90(DataAugmenter):
    def get_directions_mapped(self) -> Dict[str, str]:
        # Rotate all actions 90 degrees
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.EAST,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.NORTH
        }

    def _op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        k = -1 if inverse else 1
        dims = (-2, -3) if is_policy else (-1, -2)
        return torch.rot90(t, k=k, dims=dims)


class Rot180(DataAugmenter):
    def get_directions_mapped(self) -> Dict[str, str]:
        # Rotate all actions 180 degrees
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.SOUTH,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.EAST
        }

    def _op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-2, -3) if is_policy else (-1, -2)
        return torch.rot90(t, k=2, dims=dims)


class Rot270(DataAugmenter):
    def get_directions_mapped(self) -> Dict[str, str]:
        # Rotate all actions 270 degrees
        return {
            Constants.DIRECTIONS.NORTH: Constants.DIRECTIONS.WEST,
            Constants.DIRECTIONS.EAST: Constants.DIRECTIONS.NORTH,
            Constants.DIRECTIONS.SOUTH: Constants.DIRECTIONS.EAST,
            Constants.DIRECTIONS.WEST: Constants.DIRECTIONS.SOUTH
        }

    def _op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        k = 1 if inverse else -1
        dims = (-2, -3) if is_policy else (-1, -2)
        return torch.rot90(t, k=k, dims=dims)
