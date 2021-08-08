import torch
from types import SimpleNamespace

from .lux_gym import act_spaces, obs_spaces, reward_spaces


def flags_to_namespace(flags: dict) -> SimpleNamespace:
    flags = SimpleNamespace(**flags)

    # Env params
    flags.act_space = act_spaces.__dict__[flags.act_space]
    assert issubclass(flags.act_space, act_spaces.BaseActSpace), f"{flags.act_space}"
    flags.obs_space = obs_spaces.__dict__[flags.obs_space]
    assert issubclass(flags.obs_space, obs_spaces.BaseObsSpace), f"{flags.obs_space}"
    flags.reward_space = reward_spaces.__dict__[flags.reward_space]
    assert issubclass(flags.reward_space, reward_spaces.BaseRewardSpace), f"{flags.reward_space}"

    # Optimizer params
    flags.optimizer_class = torch.optim.__dict__[flags.optimizer_class]

    # Miscellaneous params
    flags.actor_device = torch.device(flags.actor_device)
    flags.learner_device = torch.device(flags.learner_device)

    return flags
