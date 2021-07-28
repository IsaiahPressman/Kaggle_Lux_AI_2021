import gym
import numpy as np
import torch
from typing import *

from .obs_spaces import MAX_BOARD_SIZE


class PadEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_board_size: tuple[int, int] = MAX_BOARD_SIZE):
        super(PadEnv, self).__init__(env)
        self.max_board_size = max_board_size
        self.observation_space = self.unwrapped.obs_type.get_obs_spec(max_board_size)
        self.board_mask = np.zeros(max_board_size, dtype=bool)
        self.board_mask[:self.orig_board_dims[0], :self.orig_board_dims[1]] = 1

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            key: np.pad(val, pad_width=self.pad_width, constant_values=0.) for key, val in observation.items()
        }

    def info(self, info: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        assert "board_mask" not in info.keys()
        info["board_mask"] = self.board_mask
        return info

    def reset(self, **kwargs):
        obs, reward, done, info = super(PadEnv, self).reset(**kwargs)
        self.board_mask[:] = 0
        self.board_mask[:self.orig_board_dims[0], :self.orig_board_dims[1]] = 1
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
    def pad_width(self):
        return (
            (0, 0),
            (0, 0),
            (0, self.max_board_size[0] - self.orig_board_dims[0]),
            (0, self.max_board_size[1] - self.orig_board_dims[1])
        )


class VecEnv:
    def __init__(self, envs: list[gym.Env]):
        self.envs = envs
        self.last_outs = [() for _ in range(len(self.envs))]

    @staticmethod
    def _vectorize_env_outs(env_outs: list[tuple]) -> tuple:
        obs_list, reward_list, done_list, info_list = zip(*env_outs)
        obs_stacked = {
            key: np.stack([obs[key] for obs in obs_list], axis=0) for key in obs_list[0].keys()
        }
        reward_stacked = np.array(reward_list)
        done_stacked = np.array(done_list)
        info_stacked = {
            key: np.stack([info[key] for info in info_list], axis=0) for key in info_list[0].keys()
        }
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

    def render(self, idx: int, mode: str = 'human', **kwargs):
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
    def __init__(self, env: Union[gym.Env, VecEnv], device: torch.device = torch.device('cpu')):
        super(PytorchEnv, self).__init__(env)
        self.device = device
        
    def reset(self, **kwargs):
        obs, reward, done, info = super(PytorchEnv, self).reset(**kwargs)
        return self.to_tensor(obs), reward, done, self.to_tensor(info)
        
    def step(self, action: dict[str, torch.Tensor]):
        action = {
            key: val.cpu().numpy() for key, val in action.items()
        }
        obs, reward, done, info = super(PytorchEnv, self).step(action)
        return self.to_tensor(obs), reward, done, self.to_tensor(info)

    def to_tensor(self, d: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(val).to(self.device, non_blocking=True) for key, val in d.items()
        }


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
            **info
        )

    def reset(self, **kwargs):
        return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

    def step(self, action):
        return DictEnv._dict_env_out(super(DictEnv, self).step(action))
