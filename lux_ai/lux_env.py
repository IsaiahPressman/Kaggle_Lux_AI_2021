import atexit
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
from .obs_types import ObsType

DIR_PATH = Path(__file__).parent


def cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions


class LuxEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(
            self,
            obs_type: ObsType,
            configuration: Optional[dict[str, any]] = None,
            seed: Optional[int] = None
    ):
        super(LuxEnv, self).__init__()
        self.obs_type = obs_type
        self.action_space = None
        self.observation_space = self.obs_type.get_obs_spec()

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
        self.action_space = None
        self.observation_space = self.obs_type.get_obs_spec((self.game_state.map_width, self.game_state.map_height))

        return self.obs, [0., 0.], self.done, self.info

    def step(self, action):
        self._step(self.process_actions(action))

        # 3.3 : handle rewards when done
        if self.done:
            # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
            rewards = [int(compute_reward(p)) for p in self.game_state.players]
            rewards = (rankdata(rewards) - 1.) * 2. - 1.
            rewards = list(rewards)
        else:
            rewards = [0., 0.]

        return self.obs, rewards, self.done, self.info

    def process_actions(self, action) -> list[list[str]]:
        # TODO
        return [[]]

    def _step(self, action: list[list[str]]) -> NoReturn:
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

    def render(self, mode='human'):
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')

    @property
    def obs(self) -> Game:
        return self.game_state

    @property
    def info(self) -> dict[str, any]:
        # TODO
        return {}


def compute_reward(player):
    ct_count = sum([len(v.citytiles) for k, v in player.cities.items()])
    unit_count = len(player.units)
    # max board size is 32 x 32 => 1024 max city tiles and units,
    # so this should keep it strictly so we break by city tiles then unit count
    return ct_count * 1000 + unit_count
