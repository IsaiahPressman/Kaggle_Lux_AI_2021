import torch
from typing import Optional

from . import act_spaces, obs_spaces, reward_spaces, multi_subtask
from .lux_env import LuxEnv
from .wrappers import RewardSpaceWrapper, PadFixedShapeEnv, LoggingEnv, VecEnv, PytorchEnv, DictEnv

ACT_SPACES_DICT = {
    key: val for key, val in act_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, act_spaces.BaseActSpace)
}
OBS_SPACES_DICT = {
    key: val for key, val in obs_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, obs_spaces.BaseObsSpace)
}
REWARD_SPACES_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
}
REWARD_SPACES_DICT.update({
    key: val for key, val in multi_subtask.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
})
SUBTASKS_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.Subtask)
}
SUBTASK_SAMPLERS_DICT = {
    key: val for key, val in multi_subtask.__dict__.items()
    if isinstance(val, type) and issubclass(val, multi_subtask.SubtaskSampler)
}


def create_env(flags, device: torch.device, seed: Optional[int] = None) -> DictEnv:
    if seed is None:
        seed = flags.seed
    envs = []
    for i in range(flags.n_actor_envs):
        env = LuxEnv(
            act_space=flags.act_space(),
            obs_space=flags.obs_space(**flags.obs_space_kwargs),
            seed=seed
        )
        reward_space = create_reward_space(flags)
        env = RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env, reward_space)
        env = PadFixedShapeEnv(env)
        env = LoggingEnv(env, reward_space)
        envs.append(env)
    env = VecEnv(envs)
    env = PytorchEnv(env, device)
    env = DictEnv(env)
    return env


def create_reward_space(flags) -> reward_spaces.BaseRewardSpace:
    if flags.reward_space is multi_subtask.MultiSubtask:
        assert "subtasks" in flags.reward_space_kwargs and "subtask_sampler" in flags.reward_space_kwargs
        subtasks = [SUBTASKS_DICT[s] for s in flags.reward_space_kwargs["subtasks"]]
        subtask_sampler = SUBTASK_SAMPLERS_DICT[flags.reward_space_kwargs["subtask_sampler"]]
        return flags.reward_space(subtasks, subtask_sampler)

    return flags.reward_space(**flags.reward_space_kwargs)
