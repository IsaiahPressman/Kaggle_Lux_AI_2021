from abc import ABC, abstractmethod
import torch
from typing import Dict

from ..lux.constants import Constants
from ..lux.game import Game
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from ..utils import DEBUG_MESSAGE


class DataAugmenter(ABC):
    def __init__(self, *args, **kwargs):
        directions_mapped_forward = self.get_directions_mapped()
        direction_mapped_inverse = {val: key for key, val in directions_mapped_forward.items()}
        assert len(directions_mapped_forward) == len(direction_mapped_inverse)

        self.transformed_action_idxs_forward = {}
        self.transformed_action_idxs_inverse = {}
        for space, meanings_to_idx in ACTION_MEANINGS_TO_IDX.items():
            transformed_space_idxs_forward = []
            for action, idx in meanings_to_idx.items():
                for d, d_mapped in directions_mapped_forward.items():
                    if action.endswith(f"_{d_mapped}"):
                        transformed_space_idxs_forward.append(meanings_to_idx[action[:-1] + d])
                        break
                else:
                    transformed_space_idxs_forward.append(idx)
            self.transformed_action_idxs_forward[space] = transformed_space_idxs_forward

            transformed_space_idxs_inverse = []
            for action, idx in meanings_to_idx.items():
                for d, d_mapped in direction_mapped_inverse.items():
                    if action.endswith(f"_{d_mapped}"):
                        transformed_space_idxs_inverse.append(meanings_to_idx[action[:-1] + d])
                        break
                else:
                    transformed_space_idxs_inverse.append(idx)
            self.transformed_action_idxs_inverse[space] = transformed_space_idxs_inverse

    def apply(self, x: Dict[str, torch.Tensor], inverse: bool, is_policy: bool) -> Dict[str, torch.Tensor]:
        n_dims = 6 if is_policy else 5
        for tensor in x.values():
            if tensor.dim() != n_dims:
                continue
            if is_policy:
                assert tensor.shape[-3] == tensor.shape[-2]
            else:
                assert tensor.shape[-2] == tensor.shape[-1]
        x_transformed = {
            key: self.op(val, inverse=inverse, is_policy=is_policy) if val.dim() == n_dims else val
            for key, val in x.items()
        }
        if is_policy:
            return self._transform_policy(x_transformed, inverse=inverse)
        return x_transformed

    def _apply_and_apply_inverse(self, x: Dict[str, torch.Tensor], is_policy: bool) -> Dict[str, torch.Tensor]:
        """
        This method is for debugging only.
        If everything is working correctly, it should leave the input unchanged.
        """
        x_transformed = self.apply(x, inverse=False, is_policy=is_policy)
        return self.apply(x_transformed, inverse=True, is_policy=is_policy)

    @abstractmethod
    def get_directions_mapped(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
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

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-2,) if is_policy else (-1,)
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

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-3,) if is_policy else (-2,)
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

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        k = -1 if inverse else 1
        dims = (-3, -2) if is_policy else (-2, -1)
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

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        dims = (-3, -2) if is_policy else (-2, -1)
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

    def op(self, t: torch.Tensor, inverse: bool, is_policy: bool) -> torch.Tensor:
        k = 1 if inverse else -1
        dims = (-3, -2) if is_policy else (-2, -1)
        return torch.rot90(t, k=k, dims=dims)


def player_relative_reflection(game_state: Game) -> DataAugmenter:
    p1_city_pos, p2_city_pos = [p.city_tiles[0].pos for p in game_state.players]
    if p1_city_pos.x == p2_city_pos.x:
        DEBUG_MESSAGE("Reflection mode: vertical")
        return VerticalFlip()
    else:
        DEBUG_MESSAGE("Reflection mode: horizontal")
        return HorizontalFlip()
