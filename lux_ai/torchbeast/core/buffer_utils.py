import numpy as np
import torch
from typing import *

Buffers = list[dict[str, Union[dict, torch.Tensor]]]


def fill_buffers_inplace(buffers: Union[dict, torch.Tensor], fill_vals: Union[dict, torch.Tensor], step: int):
    if isinstance(fill_vals, dict):
        for key, val in fill_vals.items():
            assert key in buffers.keys()
            fill_buffers_inplace(buffers[key], val, step)
    else:
        buffers[step, ...] = fill_vals


def stack_buffers(buffers: Buffers) -> dict[str, Union[dict, torch.Tensor]]:
    stacked_buffers = {}
    for key, val in buffers[0].keys():
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers])
        else:
            stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=1)
    return stacked_buffers


def slice_buffers(buffers: Union[dict, torch.Tensor], s: slice) -> Union[dict, torch.Tensor]:
    if isinstance(buffers, dict):
        sliced_buffers = {}
        for key, val in buffers.items():
            sliced_buffers[key] = slice_buffers(val, s)
        return sliced_buffers
    else:
        return buffers[s]


def _create_buffers_from_specs(specs: dict[str, Union[dict, tuple, torch.dtype]]) -> Union[dict, torch.Tensor]:
    if isinstance(specs, dict):
        new_buffers = {}
        for key, val in specs.items():
            new_buffers[key] = _create_buffers_from_specs(val)
        return new_buffers
    else:
        return torch.empty(**specs).share_memory_()


def _create_buffers_like(buffers: Union[dict, np.ndarray, torch.Tensor]) -> Union[dict, torch.Tensor]:
    if isinstance(buffers, dict):
        torch_buffers = {}
        for key, val in buffers.items():
            torch_buffers[key] = _create_buffers_like(val)
        return torch_buffers
    else:
        return torch.empty_like(torch.tensor(buffers)).share_memory_()


def create_buffers(flags, example_info: dict[str, Union[dict, np.ndarray, torch.Tensor]]) -> Buffers:
    t = flags.unroll_length
    n = flags.n_actor_envs
    p = 2
    specs = dict(
        observation={
            key: dict(size=(t + 1, n, *val.shape), dtype=torch.float32)
            for key, val in flags.observation_space.get_obs_spec().spaces.items()
        },
        reward=dict(size=(t + 1, n, p), dtype=torch.float32),
        done=dict(size=(t + 1, n), dtype=torch.bool),
        policy_logits={
            key: dict(size=(t + 1, n, *val), dtype=torch.float32)
            for key, val in flags.act_space.get_action_space_expanded_shape().items()
        },
        baseline=dict(size=(t + 1, n, p), dtype=torch.float32),
        action={
            key: dict(size=(t + 1, n, *val.shape), dtype=torch.int64)
            for key, val in flags.act_space.get_action_space().spaces.items()
        },
    )
    buffers: Buffers = []
    for _ in range(flags.num_buffers):
        new_buffer = _create_buffers_from_specs(specs)
        new_buffer["info"] = _create_buffers_like(example_info)
        buffers.append(new_buffer)
    return buffers
