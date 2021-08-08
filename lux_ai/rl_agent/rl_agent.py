from pathlib import Path
import torch
from typing import *
import yaml

from ..lux_gym import LuxEnv, wrappers
from ..nns import create_model
from ..utils import flags_to_namespace

from ..lux.game import Game
from ..lux.game_map import Cell
from ..lux.constants import Constants
from ..lux.game_constants import GAME_CONSTANTS
from ..lux import annotate

"""
if __package__ == "":
    # not sure how to fix this atm
    from ..lux.game import Game
    from ..lux.game_map import Cell
    from ..lux.constants import Constants
    from ..lux.game_constants import GAME_CONSTANTS
    from ..lux import annotate
else:
    from lux_ai.lux.game import Game
    from lux_ai.lux.game_map import Cell
    from lux_ai.lux.constants import Constants
    from lux_ai.lux.game_constants import GAME_CONSTANTS
    from lux_ai.lux import annotate
"""

DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES
CONFIG_PATH = Path(__file__).parent / "config.yaml"
CHECKPOINT_PATH, = list(Path(__file__).parent.glob('*.pt'))
AGENT = None


class RLAgent:
    def __init__(self, obs, conf):
        with open(CONFIG_PATH, 'r') as stream:
            self.flags = flags_to_namespace(yaml.safe_load(stream))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device('cpu')
        env = LuxEnv(
            act_space=self.flags.act_space(),
            obs_space=self.flags.obs_space(),
            reward_space=self.flags.reward_space(),
            run_game_automatically=False
        )
        env = env.obs_space.wrap_env(env)
        env = wrappers.PadEnv(env)
        env = wrappers.VecEnv([env])
        env = wrappers.PytorchEnv(env, self.device)
        env = wrappers.DictEnv(env)

        self.env = env
        self.env.reset(observation_updates=obs["updates"], force=True)
        self.action_placeholder = {
            key: torch.zeros(space.shape)
            for key, space in self.unwrapped_env.action_space.get_action_space().spaces.items()
        }

        self.model = create_model(self.flags, self.device)
        checkpoint_states = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

    def __call__(self, obs, conf) -> List[str]:
        self.preprocess(obs, conf)
        env_output = self.env.step(self.action_placeholder)
        agent_output = self.model.select_best_actions(env_output)
        # agent_output = self.model.sample_actions(env_output)
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

        actions.append(annotate.sidetext(f"Turn: {self.game_state.turn}"))
        return actions

    def preprocess(self, obs, conf) -> NoReturn:
        # Do not edit
        self.unwrapped_env.manual_step(obs["updates"])

    @property
    def unwrapped_env(self) -> LuxEnv:
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> Game:
        return self.unwrapped_env.game_state


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = RLAgent(obs, conf)
    return AGENT(obs, conf)
