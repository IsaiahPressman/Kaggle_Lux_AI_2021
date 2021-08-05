import copy
import gym
import numpy as np
import torch
from typing import *

from .obs_spaces import MAX_BOARD_SIZE


class PadEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_board_size: tuple[int, int] = MAX_BOARD_SIZE):
        super(PadEnv, self).__init__(env)
        self.max_board_size = max_board_size
        self.observation_space = self.unwrapped.obs_space.get_obs_spec(max_board_size)
        self.input_mask = np.zeros((1,) + max_board_size, dtype=bool)
        self.input_mask[:, :self.orig_board_dims[0], :self.orig_board_dims[1]] = 1

    def _pad(self, x: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
        if isinstance(x, dict):
            return {key: self._pad(val) for key, val in x.items()}
        elif x.ndim == 4 and x.shape[2:4] == self.orig_board_dims:
            return np.pad(x, pad_width=self.base_pad_width, constant_values=0.)
        elif x.ndim == 5 and x.shape[2:4] == self.orig_board_dims:
            return np.pad(x, pad_width=self.base_pad_width + ((0, 0),), constant_values=0.)
        else:
            return x

    def observation(self, observation: dict[str, Union[dict, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            key: self._pad(val) for key, val in observation.items()
        }

    def info(self, info: dict[str, Union[dict, np.ndarray]]) -> dict[str, np.ndarray]:
        info = {
            key: self._pad(val) for key, val in info.items()
        }
        assert "input_mask" not in info.keys()
        info["input_mask"] = self.input_mask
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(PadEnv, self).reset(**kwargs)
        self.input_mask[:] = 0
        self.input_mask[:, :self.orig_board_dims[0], :self.orig_board_dims[1]] = 1
        return self.observation(obs), reward, done, self.info(info)

    def step(self, action: dict[str, np.ndarray]):
        action = {
            key: val[..., :self.orig_board_dims[0], :self.orig_board_dims[1]] for key, val in action.items()
        }
        obs, reward, done, info = super(PadEnv, self).step(action)
        return self.observation(obs), reward, done, self.info(info)

    @property
    def orig_board_dims(self) -> tuple[int, int]:
        return self.unwrapped.board_dims

    @property
    def base_pad_width(self):
        return (
            (0, 0),
            (0, 0),
            (0, self.max_board_size[0] - self.orig_board_dims[0]),
            (0, self.max_board_size[1] - self.orig_board_dims[1])
        )


class LoggingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(LoggingEnv, self).__init__(env)
        self.vals_peak = {}
        self.reward_sums = [0., 0.]
        # TODO: Resource mining % like in visualizer?
        # self.resource_count = {"wood", etc...}

    def info(self, info: dict[str, np.ndarray], rewards: list[int]) -> dict[str, np.ndarray]:
        info = copy.copy(info)
        game_state = self.env.unwrapped.game_state
        logs = {
            "step": [game_state.turn],
            "city_tiles": [p.city_tile_count for p in game_state.players],
            "separate_cities": [len(p.cities) for p in game_state.players],
            "workers": [sum(u.is_worker() for u in p.units) for p in game_state.players],
            "carts": [sum(u.is_cart() for u in p.units) for p in game_state.players],
            "research_points": [p.research_points for p in game_state.players],
        }
        self.vals_peak = {
            key: np.maximum(val, logs[key]) for key, val in self.vals_peak.items()
        }
        logs.update({f"{key}_peak": val.copy() for key, val in self.vals_peak.items()})

        self.reward_sums = [r + s for r, s in zip(rewards, self.reward_sums)]
        logs["reward_sums"] = [np.mean(self.reward_sums)]
        logs["reward_sum_magnitudes"] = [np.mean(np.abs(self.reward_sums))]
        info.update({f"logging_{key}": np.array(val, dtype=np.float32) for key, val in logs.items()})
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(LoggingEnv, self).reset(**kwargs)
        self._reset_peak_vals()
        self.reward_sums = [0., 0.]
        return obs, reward, done, self.info(info, reward)

    def step(self, action: dict[str, np.ndarray]):
        obs, reward, done, info = super(LoggingEnv, self).step(action)
        return obs, reward, done, self.info(info, reward)

    def _reset_peak_vals(self) -> NoReturn:
        self.vals_peak = {
            key: np.array([0., 0.])
            for key in [
                "city_tiles",
                "separate_cities",
                "workers",
                "carts",
            ]
        }


class VecEnv(gym.Env):
    def __init__(self, envs: list[gym.Env]):
        self.envs = envs
        self.last_outs = [() for _ in range(len(self.envs))]

    @staticmethod
    def _stack_dict(x: list[Union[dict, np.ndarray]]) -> Union[dict, np.ndarray]:
        if isinstance(x[0], dict):
            return {key: VecEnv._stack_dict([i[key] for i in x]) for key in x[0].keys()}
        else:
            return np.stack([arr for arr in x], axis=0)

    @staticmethod
    def _vectorize_env_outs(env_outs: list[tuple]) -> tuple:
        obs_list, reward_list, done_list, info_list = zip(*env_outs)
        obs_stacked = VecEnv._stack_dict(obs_list)
        reward_stacked = np.array(reward_list)
        done_stacked = np.array(done_list)
        info_stacked = VecEnv._stack_dict(info_list)
        return obs_stacked, reward_stacked, done_stacked, info_stacked

    def reset(self, force: bool = False, **kwargs):
        if force:
            # noinspection PyArgumentList
            self.last_outs = [env.reset(**kwargs) for env in self.envs]
            return VecEnv._vectorize_env_outs(self.last_outs)

        for i, env in enumerate(self.envs):
            # Check if env finished
            if self.last_outs[i][2]:
                # noinspection PyArgumentList
                self.last_outs[i] = env.reset(**kwargs)
        return VecEnv._vectorize_env_outs(self.last_outs)

    def step(self, action: dict[str, np.ndarray]):
        actions = [
            {key: val[i] for key, val in action.items()} for i in range(len(self.envs))
        ]
        self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
        return VecEnv._vectorize_env_outs(self.last_outs)

    def render(self, idx: int, mode: str = "human", **kwargs):
        # noinspection PyArgumentList
        return self.envs[idx].render(mode, **kwargs)

    def close(self):
        return [env.close() for env in self.envs]

    def seed(self, seed: int) -> list:
        return [env.seed(seed + i) for i, env in enumerate(self.envs)]

    @property
    def unwrapped(self) -> list[gym.Env]:
        return [env.unwrapped for env in self.envs]

    @property
    def action_space(self) -> list[gym.spaces.Dict]:
        return [env.action_space for env in self.envs]

    @property
    def observation_space(self) -> list[gym.spaces.Dict]:
        return [env.observation_space for env in self.envs]

    @property
    def metadata(self) -> list[dict]:
        return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):
    def __init__(self, env: Union[gym.Env, VecEnv], device: torch.device = torch.device("cpu")):
        super(PytorchEnv, self).__init__(env)
        self.device = device
        
    def reset(self, **kwargs):
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).reset(**kwargs)])
        
    def step(self, action: dict[str, torch.Tensor]):
        action = {
            key: val.cpu().numpy() for key, val in action.items()
        }
        return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).step(action)])

    def _to_tensor(self, x: Union[dict, np.ndarray]) -> dict[str, Union[dict, torch.Tensor]]:
        if isinstance(x, dict):
            return {key: self._to_tensor(val) for key, val in x.items()}
        else:
            return torch.from_numpy(x).to(self.device, non_blocking=True)


class DictEnv(gym.Wrapper):
    @staticmethod
    def _dict_env_out(env_out: tuple) -> dict:
        obs, reward, done, info = env_out
        assert "obs" not in info.keys()
        assert "reward" not in info.keys()
        assert "done" not in info.keys()
        return dict(
            obs=obs,
            reward=reward,
            done=done,
            info=info
        )

    def reset(self, **kwargs):
        return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

    def step(self, action):
        return DictEnv._dict_env_out(super(DictEnv, self).step(action))
