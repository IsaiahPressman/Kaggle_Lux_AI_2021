import atexit
from enum import auto, Enum
import gym
import json

import numpy as np
from kaggle_environments import make
import math
from pathlib import Path
import random
from scipy.stats import rankdata
from subprocess import Popen, PIPE
from typing import Optional, NoReturn

from .lux.game import Game

DIR_PATH = Path(__file__).parent


class ObsType(Enum):
    """
    An enum of all available obs_types
    WARNING: enum order is subject to change
    """
    MULTIHOT_PADDED_OBS = auto()

    def get_obs_spec(self) -> tuple[int, ...]:
        if self == ObsType.MULTIHOT_OBS:
            # 32x32 grid of categorical embeddings
            # 32x32 grid of multiplications?
            # ^ This should all be doable with a single matmul operation
            # Also, include turn info, day/night info, and time until day/night info?
            # ^ How to encode this info, perhaps another separate obs dimension?
            return gym.spaces # TODO
        else:
            raise NotImplementedError(f'ObsType not yet implemented: {self.name}')


def cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions


class _LuxEnvRaw(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, configuration: Optional[dict[str, any]] = None, seed: Optional[int] = None):
        super(_LuxEnvRaw, self).__init__()
        self.action_space = None
        self.observation_space = None

        self.game_state = Game()
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("lux_ai_2021").configuration
        if seed is not None:
            self.configuration["seed"] = seed
        elif "seed" not in self.configuration:
            self.configuration["seed"] = math.floor(random.random() * 1e9)
        self.done = False

        # 1.1: Initialize dimensions in the background
        self.dimension_process = Popen(
            ["node", str(DIR_PATH / "dimensions/main.js")],
            stdin=PIPE,
            stdout=PIPE
        )
        atexit.register(cleanup_dimensions_factory(self.dimension_process))

    def reset(self):
        # 1.2: Initialize a blank state game if new episode is starting
        self.configuration["seed"] += 1
        initiate = {
            "type": "start",
            "agent_names": [],  # unsure if this is provided?
            "config": self.configuration
        }
        self.dimension_process.stdin.write((json.dumps(initiate) + "\n").encode())
        self.dimension_process.stdin.flush()
        agent1res = json.loads(self.dimension_process.stdout.readline())
        _ = self.dimension_process.stdout.readline()

        self.game_state = Game()
        self.game_state._initialize(agent1res)
        self.game_state._update(agent1res[2:])
        self.done = False

        return self.obs, [0., 0.], self.done, self.info

    def step(self, action: list[list[str]]):
        # 2.: Pass in actions (json representation along with id of who made that action),
        #       and agent information (id, status) to dimensions via stdin
        # TODO: Does state need to include info other than actions?
        assert len(action) == 2
        state = [{'action': a} for a in action]
        self.dimension_process.stdin.write((json.dumps(state) + "\n").encode())
        self.dimension_process.stdin.flush()

        # 3.1 : Receive and parse the observations returned by dimensions via stdout
        agent1res = json.loads(self.dimension_process.stdout.readline())
        _ = self.dimension_process.stdout.readline()
        self.game_state._update(agent1res)

        # Check if done
        match_status = json.loads(self.dimension_process.stdout.readline())
        self.done = match_status["status"] == "finished"

        # 3.3 : handle rewards when done
        if self.done:
            # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
            rewards = [int(compute_reward(p)) for p in self.game_state.players]
            rewards = (rankdata(rewards) - 1.) * 2. - 1.
            rewards = list(rewards)
        else:
            rewards = [0., 0.]

        return self.obs, rewards, self.done, self.info

    def render(self, mode='human'):
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')

    @property
    def obs(self) -> Game:
        # TODO: Process game_state into a numpy array in some capacity?
        # TODO: Or just leave that for a wrapper?
        return self.game_state

    @property
    def info(self) -> dict[str, any]:
        # TODO
        return {}


class LuxEnv(_LuxEnvRaw):
    def __init__(self, obs_type: ObsType, *args, **kwargs):
        super(LuxEnv, self).__init__(*args, **kwargs)
        self.obs_type = obs_type
        # TODO: Define action and observation space using gym.spaces
        self.action_space = None
        self.observation_space = None

    @property
    def obs(self) -> np.ndarray:
        return

    @property
    def info(self) -> dict[str, any]:
        return


def compute_reward(player):
    ct_count = sum([len(v.citytiles) for k, v in player.cities.items()])
    unit_count = len(player.units)
    # max board size is 32 x 32 => 1024 max city tiles and units,
    # so this should keep it strictly so we break by city tiles then unit count
    return ct_count * 1000 + unit_count
