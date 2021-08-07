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
from subprocess import Popen, PIPE
from typing import NoReturn, Optional

from ..lux.game import Game
from ..lux.game_objects import Unit, CityTile
from ..lux_gym.act_spaces import BaseActSpace, ACTION_MEANINGS, ACTION_MEANINGS_TO_IDX, DIRECTIONS, RESOURCES
from ..lux_gym.obs_spaces import ObsSpace, MAX_BOARD_SIZE
from ..lux_gym.reward_spaces import BaseRewardSpace


DIR_PATH = Path(__file__).parent.parent


def _cleanup_dimensions_factory(dimension_process: Popen) -> NoReturn:
    def cleanup_dimensions():
        if dimension_process is not None:
            dimension_process.kill()
    return cleanup_dimensions


def _generate_pos_to_unit_dict(game_state: Game) -> dict[tuple, Optional[Unit]]:
    pos_to_unit_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for unit in reversed(player.units):
            pos_to_unit_dict[(unit.pos.x, unit.pos.y)] = unit

    return pos_to_unit_dict


def _generate_pos_to_city_tile_dict(game_state: Game) -> dict[tuple, Optional[CityTile]]:
    pos_to_city_tile_dict = {(cell.pos.x, cell.pos.y): None for cell in itertools.chain(*game_state.map.map)}
    for player in game_state.players:
        for city in player.cities.values():
            for city_tile in city.citytiles:
                pos_to_city_tile_dict[(city_tile.pos.x, city_tile.pos.y)] = city_tile

    return pos_to_city_tile_dict


class LuxEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: ObsSpace,
            reward_space: BaseRewardSpace,
            configuration: Optional[dict[str, any]] = None,
            seed: Optional[int] = None,
            run_game_automatically: bool = True,
    ):
        super(LuxEnv, self).__init__()
        self.obs_space = obs_space
        self.action_space = act_space
        self.reward_space = reward_space
        self.observation_space = self.obs_space.get_obs_spec()
        self.board_dims = MAX_BOARD_SIZE
        self.run_game_automatically = run_game_automatically

        self.game_state = Game()
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = make("lux_ai_2021").configuration
        if seed is not None:
            self.seed(seed)
        elif "seed" not in self.configuration:
            self.seed()
        self.done = False
        self.info = {}
        self.pos_to_unit_dict = {}
        self.pos_to_city_tile_dict = {}

        if self.run_game_automatically:
            # 1.1: Initialize dimensions in the background
            self.dimension_process = Popen(
                ["node", str(DIR_PATH / "dimensions/main.js")],
                stdin=PIPE,
                stdout=PIPE
            )
            atexit.register(_cleanup_dimensions_factory(self.dimension_process))
        else:
            self.dimension_process = None

    def reset(self, observation_updates: Optional[list[str]] = None):
        self.game_state = Game()
        if self.run_game_automatically:
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

            self.game_state._initialize(agent1res)
            self.game_state._update(agent1res[2:])
            assert observation_updates is None, "Game is being run automatically"
        else:
            self.game_state._initialize(observation_updates)
            self.game_state._update(observation_updates[2:])

        self.done = False
        self.board_dims = (self.game_state.map_width, self.game_state.map_height)
        self.observation_space = self.obs_space.get_obs_spec(self.board_dims)
        self.info = {
            "actions_taken": {
                key: np.zeros(space.shape, dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            },
            "available_actions_mask": {
                key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
                for key, space in self.action_space.get_action_space(self.board_dims).spaces.items()
            }
        }
        self.pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        self.pos_to_city_tile_dict = _generate_pos_to_city_tile_dict(self.game_state)
        self._update_available_actions_mask()

        rewards = self.reward_space.compute_rewards(self.game_state, self.done)
        return self.obs, rewards, self.done, copy.copy(self.info)

    def step(self, action: dict[str, np.ndarray]):
        if self.run_game_automatically:
            actions_processed, actions_taken = self.process_actions(action)
            self._step(actions_processed)
            self.info["actions_taken"] = actions_taken

        self.pos_to_unit_dict = _generate_pos_to_unit_dict(self.game_state)
        self.pos_to_city_tile_dict = _generate_pos_to_city_tile_dict(self.game_state)
        self._update_available_actions_mask()

        rewards = self.reward_space.compute_rewards(self.game_state, self.done)
        return self.obs, rewards, self.done, copy.copy(self.info)

    def manual_step(self, observation_updates: list[str]) -> NoReturn:
        assert not self.run_game_automatically
        self.game_state._update(observation_updates)

    def process_actions(self, action: dict[str, np.ndarray]) -> tuple[list[list[str]], dict[str, np.ndarray]]:
        return self.action_space.process_actions(
            action,
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict
        )

    def _step(self, action: list[list[str]]) -> NoReturn:
        # 2.: Pass in actions (json representation along with id of who made that action),
        #       and agent information (id, status) to dimensions via stdin
        assert len(action) == 2
        # TODO: Does dimension process state need to include info other than actions?
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

    def _update_available_actions_mask(self) -> NoReturn:
        self.info["available_actions_mask"] = self.action_space.get_available_actions_mask(
            self.game_state,
            self.board_dims,
            self.pos_to_unit_dict,
            self.pos_to_city_tile_dict
        )

    def seed(self, seed: Optional[int] = None) -> NoReturn:
        if seed is not None:
            # Seed is incremented on reset()
            self.configuration["seed"] = seed - 1
        else:
            self.configuration["seed"] = math.floor(random.random() * 1e9)

    def get_seed(self) -> int:
        return self.configuration["seed"]

    def render(self, mode='human'):
        raise NotImplementedError('LuxEnv rendering is not implemented. Use the Lux visualizer instead.')

    @property
    def obs(self) -> Game:
        return self.game_state
