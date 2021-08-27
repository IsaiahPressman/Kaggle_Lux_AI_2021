import numpy as np
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from typing import *
import yaml

from ..lux_gym import create_reward_space, LuxEnv, wrappers
from ..lux_gym.act_spaces import MAX_BOARD_SIZE, ACTION_MEANINGS
from ..handcrafted_agents.utils import get_city_tiles, DEBUG_MESSAGE, RUNTIME_DEBUG_MESSAGE, RUNTIME_ASSERT
from ..handcrafted_agents.utility_constants import MAX_RESEARCH
from ..nns import create_model
from ..utils import flags_to_namespace

from ..lux.game import Game
from ..lux.game_map import Cell
from ..lux.constants import Constants
from ..lux.game_constants import GAME_CONSTANTS
from ..lux.game_objects import CityTile, Unit
from ..lux import annotate

DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES
MODEL_CONFIG_PATH = Path(__file__).parent / "config.yaml"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))
AGENT = None

os.environ["OMP_NUM_THREADS"] = "1"


def pos_to_loc(pos: Tuple[int, int], board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> int:
    return pos[0] * board_dims[1] + pos[1]


class RLAgent:
    def __init__(self, obs, conf):
        with open(MODEL_CONFIG_PATH, 'r') as stream:
            self.model_flags = flags_to_namespace(yaml.safe_load(stream))
        with open(RL_AGENT_CONFIG_PATH, 'r') as stream:
            self.agent_flags = SimpleNamespace(**yaml.safe_load(stream))
        self.device = torch.device(self.agent_flags.device) if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        env = LuxEnv(
            act_space=self.model_flags.act_space(),
            obs_space=self.model_flags.obs_space(),
            configuration=conf,
            run_game_automatically=False
        )
        reward_space = create_reward_space(self.model_flags)
        env = wrappers.RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env, reward_space)
        env = wrappers.PadFixedShapeEnv(env)
        env = wrappers.VecEnv([env])
        env = wrappers.PytorchEnv(env, self.device)
        env = wrappers.DictEnv(env)

        self.env = env
        self.env.reset(observation_updates=obs["updates"], force=True)
        self.action_placeholder = {
            key: torch.zeros(space.shape)
            for key, space in self.unwrapped_env.action_space.get_action_space().spaces.items()
        }

        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]
        self.my_city_tile_mat = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        # NB: loc = pos[0] * n_cols + pos[1]
        self.loc_to_actionable_city_tiles = {}
        self.loc_to_actionable_workers = {}
        self.loc_to_actionable_carts = {}

    def __call__(self, obs, conf) -> List[str]:
        self.preprocess(obs, conf)
        env_output = self.get_env_output()
        with torch.no_grad():
            agent_output = self.model.select_best_actions(env_output, actions_per_square=None)
            # agent_output = self.model.sample_actions(env_output, actions_per_square=None)

        if self.agent_flags.use_collision_detection:
            actions = self.resolve_collision_detection(obs, agent_output)
        else:
            actions, _ = self.unwrapped_env.process_actions({
                key: value.squeeze(0).cpu().numpy() for key, value in agent_output["actions"].items()
            })
            actions = actions[obs.player]

        """
        # you can add debug annotations using the functions in the annotate object
        i = game_state.turn + observation.player
        actions.append(annotate.circle(i % width, i % height))
        i += 1
        actions.append(annotate.x(i % width, i % height))
        i += 1
        actions.append(annotate.line(i % width, i % height, (i + 3) % width, (i + 3) % height))
        actions.append(annotate.text(0, 1, f"{game_state.turn}_Text!"))
        actions.append(annotate.sidetext(f"Research points: {player.research_points}"))
        """

        value = agent_output["baseline"].squeeze().cpu().numpy()[obs.player]
        info_msg = f"Turn: {self.game_state.turn} - Predicted value: {value:.2f}"
        actions.append(annotate.sidetext(info_msg))
        RUNTIME_DEBUG_MESSAGE(info_msg)
        return actions

    def preprocess(self, obs, conf) -> NoReturn:
        self.unwrapped_env.manual_step(obs["updates"])
        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]

        self.my_city_tile_mat[:] = False
        self.loc_to_actionable_city_tiles: Dict[int, CityTile] = {}
        self.loc_to_actionable_workers: Dict[int, List[Unit]] = {}
        self.loc_to_actionable_carts: Dict[int, List[Unit]] = {}
        for unit in self.me.units:
            if unit.can_act():
                if unit.is_worker():
                    dictionary = self.loc_to_actionable_workers
                elif unit.is_cart():
                    dictionary = self.loc_to_actionable_carts
                else:
                    DEBUG_MESSAGE(f"Unrecognized unit type: {unit}")
                    continue
                dictionary.setdefault(pos_to_loc(unit.pos.astuple()), []).append(unit)
        for city_tile in get_city_tiles(self.me):
            self.my_city_tile_mat[city_tile.pos.x, city_tile.pos.y] = True
            if city_tile.can_act():
                self.loc_to_actionable_city_tiles[pos_to_loc(city_tile.pos.astuple())] = city_tile

    def get_env_output(self) -> Dict:
        return self.env.step(self.action_placeholder)

    def resolve_collision_detection(self, obs, agent_output) -> List[str]:
        # Get log_probs for all of my actions
        flat_log_probs = {
            key: torch.flatten(
                F.log_softmax(val.squeeze(0).squeeze(0), dim=-1),
                start_dim=-3,
                end_dim=-2
            ).cpu()
            for key, val in agent_output["policy_logits"].items()
        }
        my_flat_log_probs = {
            key: val[obs.player] for key, val in flat_log_probs.items()
        }
        my_flat_actions = {
            key: torch.flatten(
                val.squeeze(0).squeeze(0)[obs.player],
                start_dim=-3,
                end_dim=-2
            ).cpu()
            for key, val in agent_output["actions"].items()
        }
        # Use actions with highest prob/log_prob as highest priority
        city_tile_priorities = torch.argsort(my_flat_log_probs["city_tile"].max(dim=-1)[0], dim=-1, descending=True)

        # First handle city tile actions, ensuring the unit cap and research cap is not exceeded
        units_to_build = max(self.me.city_tile_count - len(self.me.units), 0)
        research_remaining = max(MAX_RESEARCH - self.me.research_points, 0)
        for loc in city_tile_priorities:
            loc = loc.item()
            actions = my_flat_actions["city_tile"][loc]
            if self.loc_to_actionable_city_tiles.get(loc, None) is not None:
                for i, act in enumerate(actions):
                    illegal_action = False
                    action_meaning = ACTION_MEANINGS["city_tile"][act]
                    if action_meaning.startswith("BUILD_"):
                        if units_to_build > 0:
                            units_to_build -= 1
                        else:
                            illegal_action = True
                    elif action_meaning == "RESEARCH":
                        if research_remaining > 0:
                            research_remaining -= 1
                        else:
                            illegal_action = True
                    if illegal_action:
                        my_flat_log_probs["city_tile"][loc, act] = float("-inf")
                    else:
                        break
        # Then handle unit actions, ensuring that no units try to move to the same square
        occupied_squares = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        max_loc_val = MAX_BOARD_SIZE[0] * MAX_BOARD_SIZE[1]
        combined_unit_log_probs = torch.cat(
            [my_flat_log_probs["worker"].max(dim=-1)[0], my_flat_log_probs["cart"].max(dim=-1)[0]],
            dim=-1
        )
        unit_priorities = torch.argsort(combined_unit_log_probs, dim=-1, descending=True)
        for loc in unit_priorities:
            loc = loc.item()
            if loc >= max_loc_val:
                unit_type = "cart"
                actionable_dict = self.loc_to_actionable_carts
            else:
                unit_type = "worker"
                actionable_dict = self.loc_to_actionable_workers
            loc = loc % max_loc_val
            actions = my_flat_actions[unit_type][loc]
            actionable_list = actionable_dict.get(loc, None)
            if actionable_list is not None:
                acted_count = 0
                for i, act in enumerate(actions):
                    illegal_action = False
                    action_meaning = ACTION_MEANINGS[unit_type][act]
                    if action_meaning.startswith("MOVE_"):
                        direction = action_meaning.split("_")[1]
                        new_pos = actionable_list[acted_count].pos.translate(direction, 1)
                    else:
                        new_pos = actionable_list[acted_count].pos

                    if occupied_squares[new_pos.x, new_pos.y] and not self.my_city_tile_mat[new_pos.x, new_pos.y]:
                        illegal_action = True
                    else:
                        occupied_squares[new_pos.x, new_pos.y] = True

                    if illegal_action:
                        my_flat_log_probs[unit_type][loc, act] = float("-inf")
                    else:
                        acted_count += 1

                    if acted_count >= len(actionable_list):
                        break
        # Finally, get new actions from modified log_probs
        actions_tensors = {
            key: val.view(1, *val.shape[:-2], *MAX_BOARD_SIZE, -1).argsort(dim=-1, descending=True)
            for key, val in flat_log_probs.items()
        }
        actions, _ = self.unwrapped_env.process_actions({
            key: value.numpy() for key, value in actions_tensors.items()
        })
        actions = actions[obs.player]
        return actions

    @property
    def unwrapped_env(self) -> LuxEnv:
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> Game:
        return self.unwrapped_env.game_state

    # Helper functions for debugging
    def set_to_turn(self, obs, conf, turn: int) -> NoReturn:
        self.game_state.turn = turn - 1
        self(obs, conf)


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = RLAgent(obs, conf)
    return AGENT(obs, conf)
