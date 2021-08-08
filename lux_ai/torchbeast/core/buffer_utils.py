import gym
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


def stack_buffers(buffers: Buffers, dim: int) -> dict[str, Union[dict, torch.Tensor]]:
    stacked_buffers = {}
    for key, val in buffers[0].items():
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim)
        else:
            stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=dim)
    return stacked_buffers


def split_buffers(
        buffers: dict[str, Union[dict, torch.Tensor]],
        split_size_or_sections: Union[int, list[int]],
        dim: int,
        contiguous: bool,
) -> list[Union[dict, torch.Tensor]]:
    buffers_split = None
    for key, val in buffers.items():
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


def buffers_apply(buffers: Union[dict, torch.Tensor], func: Callable[[torch.Tensor], Any]) -> Union[dict, torch.Tensor]:
    if isinstance(buffers, dict):
        return {
            key: buffers_apply(val, func) for key, val in buffers.items()
        }
    else:
        return func(buffers)


def _create_buffers_from_specs(specs: dict[str, Union[dict, tuple, torch.dtype]]) -> Union[dict, torch.Tensor]:
    if isinstance(specs, dict) and "dtype" not in specs.keys():
        new_buffers = {}
        for key, val in specs.items():
            new_buffers[key] = _create_buffers_from_specs(val)
        return new_buffers
    else:
        return torch.empty(**specs).share_memory_()


def _create_buffers_like(buffers: Union[dict, torch.Tensor], t_dim: int) -> Union[dict, torch.Tensor]:
    if isinstance(buffers, dict):
        torch_buffers = {}
        for key, val in buffers.items():
            torch_buffers[key] = _create_buffers_like(val, t_dim)
        return torch_buffers
    else:
        buffers = buffers.unsqueeze(0).expand(t_dim, *[-1 for _ in range(len(buffers.shape))])
        return torch.empty_like(buffers).share_memory_()


def create_buffers(flags, example_info: dict[str, Union[dict, np.ndarray, torch.Tensor]]) -> Buffers:
    t = flags.unroll_length
    n = flags.n_actor_envs
    p = 2
    obs_specs = {}
    for key, spec in flags.obs_space().get_obs_spec().spaces.items():
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
        policy_logits={
            key: dict(size=(t + 1, n, *val), dtype=torch.float32)
            for key, val in flags.act_space().get_action_space_expanded_shape().items()
        },
        baseline=dict(size=(t + 1, n, p), dtype=torch.float32),
        actions={
            key: dict(size=(t + 1, n, *val.shape), dtype=torch.int64)
            for key, val in flags.act_space().get_action_space().spaces.items()
        },
    )
    buffers: Buffers = []
    for _ in range(flags.num_buffers):
        new_buffer = _create_buffers_from_specs(specs)
        new_buffer["info"] = _create_buffers_like(example_info, t + 1)
        buffers.append(new_buffer)
    return buffers
