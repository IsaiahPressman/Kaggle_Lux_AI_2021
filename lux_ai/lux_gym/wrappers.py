import gym
import numpy as np
import torch

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
        obs, reward, done, info = super(PadEnv, self).step(**kwargs)
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


class PytorchEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, device: torch.device = torch.device('cpu')):
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

    def to_tensor(self, observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(val).to(self.device, non_blocking=True) for key, val in observation.items()
        }
