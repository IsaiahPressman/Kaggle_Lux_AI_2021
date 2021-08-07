import torch
from typing import *

from .lux_env import LuxEnv
from .obs_spaces import ObsSpace
from .wrappers import PadEnv, LoggingEnv, VecEnv, PytorchEnv, DictEnv


def create_env(flags, device: torch.device, seed: Optional[int] = None) -> DictEnv:
    if seed is None:
        seed = flags.seed
    envs = []
    for i in range(flags.n_actor_envs):
        env = LuxEnv(
            act_space=flags.act_space,
            obs_space=flags.obs_space,
            reward_space=flags.reward_space(),
            seed=seed
        )
        env = flags.obs_space.wrap_env(env)
        env = PadEnv(env)
        env = LoggingEnv(env)
        envs.append(env)
    env = VecEnv(envs)
    env = PytorchEnv(env, device)
    env = DictEnv(env)
    return env
