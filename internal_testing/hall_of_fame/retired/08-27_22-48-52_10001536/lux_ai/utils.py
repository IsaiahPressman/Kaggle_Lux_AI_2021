import torch
from types import SimpleNamespace
from typing import Dict

from .lux_gym import ACT_SPACES_DICT, OBS_SPACES_DICT, REWARD_SPACES_DICT


def flags_to_namespace(flags: Dict) -> SimpleNamespace:
    flags = SimpleNamespace(**flags)

    # Env params
    flags.act_space = ACT_SPACES_DICT[flags.act_space]
    flags.obs_space = OBS_SPACES_DICT[flags.obs_space]
    flags.reward_space = REWARD_SPACES_DICT[flags.reward_space]

    # Optimizer params
    flags.optimizer_class = torch.optim.__dict__[flags.optimizer_class]

    # Miscellaneous params
    flags.actor_device = torch.device(flags.actor_device)
    flags.learner_device = torch.device(flags.learner_device)

    return flags
