import atexit
import copy
import gym
import itertools
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
from .lux.game_objects import Unit
from .spaces.obs_spaces import ObsSpace, MAX_BOARD_SIZE
from .spaces.act_spaces import get_action_space, get_unit_action, get_city_tile_action

DIR_PATH = Path(__file__).parent


def _cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions


def _compute_reward(player):
    ct_count = sum([len(v.citytiles) for k, v in player.cities.items()])
    unit_count = len(player.units)
    # max board size is 32 x 32 => 1024 max city tiles and units,
    # so this should keep it strictly so we break by city tiles then unit count
    return ct_count * 1000 + unit_count


def _generate_pos_to_unit_dict(game_state: Game) -> dict[tuple, Optional[Unit]]:
    pos_to_unit_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for unit in reversed(player.units):
            pos_to_unit_dict[(unit.pos.x, unit.pos.y)] = unit

    return pos_to_unit_dict


class LuxEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(
            self,
            obs_type: ObsSpace,
            configuration: Optional[dict[str, any]] = None,
            seed: Optional[int] = None
    ):
        super(LuxEnv, self).__init__()
        self.obs_type = obs_type
        self.action_space = get_action_space()
        self.observation_space = self.obs_type.get_obs_spec()
        self.board_dims = MAX_BOARD_SIZE

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
        self.info = {}

        # 1.1: Initialize dimensions in the background
        self.dimension_process = Popen(
            ["node", str(DIR_PATH / "dimensions/main.js")],
            stdin=PIPE,
            stdout=PIPE
        )
        atexit.register(_cleanup_dimensions_factory(self.dimension_process))

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
        self.board_dims = (self.game_state.map_height, self.game_state.map_width)
        self.action_space = get_action_space(self.board_dims)
        self.observation_space = self.obs_type.get_obs_spec(self.board_dims)
        self.info = {
            "actions_taken": {
                space: np.zeros(space.shape, dtype=bool) for space in self.action_space
            },
        }

        return self.obs, [0., 0.], self.done, copy.copy(self.info)

    def step(self, action: dict[str, np.ndarray]):
        self._step(self.process_actions(action))

        # 3.3 : handle rewards when done
        if self.done:
            # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
            rewards = [int(_compute_reward(p)) for p in self.game_state.players]
            rewards = (rankdata(rewards) - 1.) * 2. - 1.
            rewards = list(rewards)
        else:
            rewards = [0., 0.]

        return self.obs, rewards, self.done, self.info

    def process_actions(self, action_tensors_dict: dict[str, np.ndarray]) -> list[list[str]]:
        action_strs = [[], []]
        self.info["actions_taken"] = {
            space: np.zeros(space.shape, dtype=bool) for space in self.action_space
        }
        pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        for player in self.game_state.players:
            p_id = player.team
            worker_actions_taken_count = np.zeros(self.board_dims, dtype=int)
            cart_actions_taken_count = np.zeros_like(worker_actions_taken_count)
            for unit in player.units:
                if unit.can_act():
                    x, y = unit.pos.x, unit.pos.y
                    if unit.is_worker():
                        # Action plane is selected for stacked units
                        action_plane = worker_actions_taken_count[x, y]
                        action_idx = action_tensors_dict["worker"][action_plane, p_id, x, y]
                        action = get_unit_action(unit, action_idx, pos_to_unit_dict)
                        self.info["actions_taken"]["worker"][action_plane, p_id, x, y] = True
                        # None: No-op
                        # "": Invalid transfer action - fed to game as no-op
                        if action is not None and action != "":
                            action_strs[p_id].append(action)
                        worker_actions_taken_count[x, y] += 1
                    elif unit.is_cart():
                        action_plane = cart_actions_taken_count[x, y]
                        action_idx = action_tensors_dict["cart"][action_plane, p_id, x, y]
                        action = get_unit_action(unit, action_idx, pos_to_unit_dict)
                        self.info["actions_taken"]["cart"][action_plane, p_id, x, y] = True
                        # None: No-op
                        # "": Invalid transfer action - fed to game as no-op
                        if action is not None and action != "":
                            action_strs[p_id].append(action)
                        cart_actions_taken_count[x, y] += 1
                    else:
                        raise NotImplementedError(f'New unit type: {unit}')
            for city in player.cities.values():
                for city_tile in city.citytiles:
                    if city_tile.can_act():
                        x, y = city_tile.pos.x, city_tile.pos.y
                        action_idx = action_tensors_dict["city_tile"][0, p_id, x, y]
                        action = get_city_tile_action(city_tile, action_idx)
                        self.info["actions_taken"]["city_tile"][0, p_id, x, y] = True
                        # None: No-op
                        if action is not None:
                            action_strs[p_id].append(action)
        return action_strs

    def _step(self, action: list[list[str]]) -> NoReturn:
        # 2.: Pass in actions (json representation along with id of who made that action),
        #       and agent information (id, status) to dimensions via stdin
        assert len(action) == 2
        # TODO: Does state need to include info other than actions?
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
