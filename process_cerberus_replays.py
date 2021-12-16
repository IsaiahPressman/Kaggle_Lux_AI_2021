import json
from copy import deepcopy
from functools import partial

import numpy as np
import requests
from kaggle_environments import structify
from tqdm import tqdm

from lux_ai.rl_agent import rl_agent

REPLAYS = [
    # replay_id, player_id
    (34014720, 0),
    (33947998, 1),
    (34222068, 0),
]
STEP_RANGE = [0, 361]
USE_DELTA_CACHE = True


def get_delta_with_cache(replay_id, agent, obs, config, skip_uncached=False):
    filepath = f"cerberus_replays/deltas/{replay_id}-{obs['step']:03}.npy"
    try:
        assert USE_DELTA_CACHE
        delta = np.load(filepath)
    except (FileNotFoundError, AssertionError):
        size = obs["width"]
        delta = np.zeros((size, size))
        if skip_uncached:
            return delta
        agent.game_state.skip = None
        out = agent(obs, config, True)
        val = float(out["baseline"][0][agent.game_state.id])
        occupied_pos = find_occupied_pos(agent.game_state)
        for i, j in tqdm(occupied_pos):
            agent.game_state.skip = (i, j)
            out = agent(obs, config, True)
            delta[i, j] = float(out["baseline"][0][agent.game_state.id]) - val
        np.save(filepath, delta)
    return delta


def find_occupied_pos(state):
    occupied_pos = set()
    for player in state.players:
        for unit in player.units:
            i, j = unit.pos.x, unit.pos.y
            occupied_pos.add((i, j))
    size = state.map.width
    for i in range(size):
        for j in range(size):
            cell = state.map.get_cell(i, j)
            if cell.resource is None:
                if cell.citytile is None:
                    # ignoring roads!
                    continue
            occupied_pos.add((i, j))
    return occupied_pos


def get_replay_with_cache(replay_id):
    filepath = f"cerberus_replays/{replay_id}.json"
    try:
        with open(filepath, "r") as f:
            replay = f.read()
    except FileNotFoundError:
        replay = requests.get(
            f"https://www.kaggleusercontent.com/episodes/{replay_id}.json"
        ).text
        with open(filepath, "w") as f:
            f.write(replay)
    replay = json.loads(replay)
    return replay


def extract_obs(step, player_id):
    obs = step[0]["observation"]
    obs.update(step[player_id]["observation"])
    return structify(obs)


if __name__ == "__main__":

    # USE_DELTA_CACHE = False

    for replay_id, player_id in REPLAYS:
        print(f"{replay_id=}")
        replay = get_replay_with_cache(replay_id)
        steps = replay["steps"]
        config = replay["configuration"]

        deltas = list()
        obs = extract_obs(steps[0], player_id)
        agent = rl_agent.RLAgent(obs, config)

        for step in steps[STEP_RANGE[0] : STEP_RANGE[1]]:
            obs = extract_obs(step, player_id)
            print(f"{obs.step=}")
            delta = get_delta_with_cache(replay_id, agent, obs, config)
