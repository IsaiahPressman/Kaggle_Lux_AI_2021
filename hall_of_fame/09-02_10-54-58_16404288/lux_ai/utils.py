import sys
import torch
from types import SimpleNamespace
from typing import Any, Dict, NoReturn, Tuple

from .lux.game_map import Position
from .lux_gym import ACT_SPACES_DICT, OBS_SPACES_DICT, REWARD_SPACES_DICT
from .utility_constants import LOCAL_EVAL


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


def in_bounds(pos: Position, board_dims: Tuple[int, int]) -> bool:
    return 0 <= pos.x < board_dims[0] and 0 <= pos.y < board_dims[1]


def DEBUG_MESSAGE(msg: Any) -> NoReturn:
    print(str(msg), file=sys.stderr)


def RUNTIME_DEBUG_MESSAGE(msg: Any) -> NoReturn:
    if not LOCAL_EVAL:
        DEBUG_MESSAGE(str(msg))


def RUNTIME_ASSERT(statement: bool, msg: Any = "") -> NoReturn:
    """
    Asserts a statement, but only raises an error during local evaluation.
    During competition evaluation, instead prints the error to the agent debug logs
    """
    if statement:
        return

    msg = str(msg)
    if LOCAL_EVAL:
        raise RuntimeError(msg)
    else:
        DEBUG_MESSAGE(msg)
