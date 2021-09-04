from copy import copy
import gym
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Tuple, Union

from ...lux_gym.act_spaces import MAX_OVERLAPPING_ACTIONS

Buffers = List[Dict[str, Union[Dict, torch.Tensor]]]


def fill_buffers_inplace(buffers: Union[Dict, torch.Tensor], fill_vals: Union[Dict, torch.Tensor], step: int):
    if isinstance(fill_vals, dict):
        for key, val in copy(fill_vals).items():
            fill_buffers_inplace(buffers[key], val, step)
    else:
        buffers[step, ...] = fill_vals[:]


def stack_buffers(buffers: Buffers, dim: int) -> Dict[str, Union[Dict, torch.Tensor]]:
    stacked_buffers = {}
    for key, val in copy(buffers[0]).items():
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim)
        else:
            stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=dim)
    return stacked_buffers


def split_buffers(
        buffers: Dict[str, Union[Dict, torch.Tensor]],
        split_size_or_sections: Union[int, List[int]],
        dim: int,
        contiguous: bool,
) -> List[Union[Dict, torch.Tensor]]:
    buffers_split = None
    for key, val in copy(buffers).items():
        if isinstance(val, dict):
            bufs = split_buffers(val, split_size_or_sections, dim, contiguous)
        else:
            bufs = torch.split(val, split_size_or_sections, dim=dim)
            if contiguous:
                bufs = [b.contiguous() for b in bufs]

        if buffers_split is None:
            buffers_split = [{} for _ in range(len(bufs))]
        assert len(bufs) == len(buffers_split)
        buffers_split = [dict(**{key: buf}, **d) for buf, d in zip(bufs, buffers_split)]
    return buffers_split


def buffers_apply(buffers: Union[Dict, torch.Tensor], func: Callable[[torch.Tensor], Any]) -> Union[Dict, torch.Tensor]:
    if isinstance(buffers, dict):
        return {
            key: buffers_apply(val, func) for key, val in copy(buffers).items()
        }
    else:
        return func(buffers)


def _create_buffers_from_specs(specs: Dict[str, Union[Dict, Tuple, torch.dtype]]) -> Union[Dict, torch.Tensor]:
    if isinstance(specs, dict) and "dtype" not in specs.keys():
        new_buffers = {}
        for key, val in specs.items():
            new_buffers[key] = _create_buffers_from_specs(val)
        return new_buffers
    else:
        return torch.empty(**specs).share_memory_()


def _create_buffers_like(buffers: Union[Dict, torch.Tensor], t_dim: int) -> Union[Dict, torch.Tensor]:
    if isinstance(buffers, dict):
        torch_buffers = {}
        for key, val in buffers.items():
            torch_buffers[key] = _create_buffers_like(val, t_dim)
        return torch_buffers
    else:
        buffers = buffers.unsqueeze(0).expand(t_dim, *[-1 for _ in range(len(buffers.shape))])
        return torch.empty_like(buffers).share_memory_()


def create_buffers(flags, example_info: Dict[str, Union[Dict, np.ndarray, torch.Tensor]]) -> Buffers:
    t = flags.unroll_length
    n = flags.n_actor_envs
    p = 2
    obs_specs = {}
    for key, spec in flags.obs_space(**flags.obs_space_kwargs).get_obs_spec().spaces.items():
        if isinstance(spec, gym.spaces.MultiBinary):
            dtype = torch.int64
        elif isinstance(spec, gym.spaces.MultiDiscrete):
            dtype = torch.int64
        elif isinstance(spec, gym.spaces.Box):
            dtype = torch.float32
        else:
            raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")
        obs_specs[key] = dict(size=(t + 1, n, *spec.shape), dtype=dtype)

    specs = dict(
        obs=obs_specs,
        reward=dict(size=(t + 1, n, p), dtype=torch.float32),
        done=dict(size=(t + 1, n), dtype=torch.bool),
        policy_logits={},
        baseline=dict(size=(t + 1, n, p), dtype=torch.float32),
        actions={},
    )
    act_space = flags.act_space()
    for key, expanded_shape in act_space.get_action_space_expanded_shape().items():
        specs["policy_logits"][key] = dict(size=(t + 1, n, *expanded_shape), dtype=torch.float32)
        final_actions_dim = min(expanded_shape[-1], MAX_OVERLAPPING_ACTIONS)
        specs["actions"][key] = dict(size=(t + 1, n, *expanded_shape[:-1], final_actions_dim), dtype=torch.int64)
    buffers: Buffers = []
    for _ in range(flags.num_buffers):
        new_buffer = _create_buffers_from_specs(specs)
        new_buffer["info"] = _create_buffers_like(example_info, t + 1)
        buffers.append(new_buffer)
    return buffers
