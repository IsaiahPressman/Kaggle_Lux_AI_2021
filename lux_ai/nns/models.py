import gym
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import *

from .in_blocks import DictInputLayer
from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.reward_spaces import RewardSpec


class DictActor(nn.Module):
    def __init__(
            self,
            in_channels: int,
            action_space: gym.spaces.Dict,
    ):
        super(DictActor, self).__init__()
        if not all([isinstance(space, gym.spaces.MultiDiscrete) for space in action_space.spaces.values()]):
            act_space_types = {key: type(space) for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must be MultiDiscrete. Found: {act_space_types}")
        if not all([len(space.shape) == 4 for space in action_space.spaces.values()]):
            act_space_ndims = {key: space.shape for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must have 4 dimensions. Found: {act_space_ndims}")
        if not all([space.nvec.min() == space.nvec.max() for space in action_space.spaces.values()]):
            act_space_n_acts = {key: np.unique(space.nvec) for key, space in action_space.spaces.items()}
            raise ValueError(f"Each action space must have the same number of actions throughout the space. "
                             f"Found: {act_space_n_acts}")
        self.n_actions = {
            key: space.nvec.max() for key, space in action_space.spaces.items()
        }
        # An action plane shape usually takes the form (n,), where n >= 1 and is used when multiple stacked units
        # must output different actions.
        self.action_plane_shapes = {
            key: space.shape[:-3] for key, space in action_space.spaces.items()
        }
        assert all([len(aps) == 1 for aps in self.action_plane_shapes.values()])
        self.actors = nn.ModuleDict({
            key: nn.Conv2d(
                in_channels,
                n_act * np.prod(self.action_plane_shapes[key]),
                (1, 1)
            ) for key, n_act in self.n_actions.items()
        })

    def forward(
            self,
            x: torch.Tensor,
            available_actions_mask: dict[str, torch.Tensor],
            sample: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Expects an input of shape batch_size * 2, n_channels, h, w
        This input will be projected by the actors, and then converted to shape batch_size, n_channels, 2, h, w
        """
        policy_logits_out = {}
        actions_out = {}
        b, _, h, w = x.shape
        for key, actor in self.actors.items():
            n_actions = self.n_actions[key]
            action_plane_shape = self.action_plane_shapes[key]
            logits = actor(x).view(b // 2, 2, n_actions, *action_plane_shape, h, w)
            # Move the logits dimension to the end and swap the player and channel dimensions
            logits = logits.permute(0, 3, 1, 4, 5, 2).contiguous()
            # In case all actions are masked, unmask all actions
            aam_filled = torch.where(
                (~available_actions_mask[key]).all(dim=-1, keepdim=True),
                torch.ones_like(available_actions_mask[key]),
                available_actions_mask[key]
            )
            assert logits.shape == aam_filled.shape
            logits = logits + torch.where(
                aam_filled,
                torch.zeros_like(logits),
                torch.zeros_like(logits) + float("-inf")
            )
            actions = DictActor.logits_to_actions(logits.view(-1, n_actions), sample)
            policy_logits_out[key] = logits
            actions_out[key] = actions.view(*logits.shape[:-1])
        return policy_logits_out, actions_out

    @staticmethod
    def logits_to_actions(logits: torch.Tensor, sample: bool) -> torch.Tensor:
        if sample:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        else:
            return logits.argmax(dim=-1)


class MultiLinear(nn.Module):
    # TODO: Add support for subtask float weightings instead of integer indices
    def __init__(self, num_layers: int, in_features: int, out_features: int, bias: bool = True):
        super(MultiLinear, self).__init__()
        self.weights = nn.Parameter(torch.empty((num_layers, in_features, out_features)))
        if bias:
            self.biases = nn.Parameter(torch.empty((num_layers, out_features)))
        else:
            self.register_parameter("biases", None)
        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.biases is not None:
            # noinspection PyProtectedMember
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x: torch.Tensor, embedding_idxs: torch.Tensor) -> torch.Tensor:
        weights = self.weights[embedding_idxs]
        if self.biases is None:
            biases = 0.
        else:
            biases = self.biases[embedding_idxs]
        return torch.matmul(x.unsqueeze(1), weights).squeeze(1) + biases


class BaselineLayer(nn.Module):
    def __init__(self, in_channels: int, reward_space: RewardSpec, n_value_heads: int):
        super(BaselineLayer, self).__init__()
        assert n_value_heads >= 1
        self.reward_min = reward_space.reward_min
        self.reward_max = reward_space.reward_max
        self.multi_headed = n_value_heads > 1
        if self.multi_headed:
            self.linear = MultiLinear(n_value_heads, in_channels, 1)
        else:
            self.linear = nn.Linear(in_channels, 1)
        if reward_space.zero_sum:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()
        if not reward_space.only_once:
            # Expand reward space to n_steps for rewards that occur more than once
            reward_space_expanded = GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
            self.reward_min *= reward_space_expanded
            self.reward_max *= reward_space_expanded

    def forward(self, x: torch.Tensor, value_head_idxs: Optional[torch.Tensor]):
        """
        Expects an input of shape b * 2, n_channels
        Returns an output of shape b, 2
        """
        # Project and reshape input
        if self.multi_headed:
            x = self.linear(x, value_head_idxs.squeeze()).view(-1, 2)
        else:
            x = self.linear(x).view(-1, 2)
        # Rescale to [0, 1], and then to the desired reward space
        x = self.activation(x)
        return x * (self.reward_max - self.reward_min) + self.reward_min


class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            base_out_channels: int,
            action_space: gym.spaces.Dict,
            reward_space: RewardSpec,
            actor_critic_activation: Callable = nn.ReLU,
            n_action_value_layers: int = 2,
            n_value_heads: int = 1,
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.dict_input_layer = DictInputLayer()
        self.base_model = base_model
        self.base_out_channels = base_out_channels

        actor_layers = []
        baseline_layers = []
        for i in range(n_action_value_layers - 1):
            actor_layers.append(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            actor_layers.append(actor_critic_activation())
            baseline_layers.append(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            baseline_layers.append(actor_critic_activation())

        self.actor_base = nn.Sequential(*actor_layers)
        self.actor = DictActor(self.base_out_channels, action_space)

        baseline_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        baseline_layers.append(nn.Flatten(start_dim=1, end_dim=-1))
        self.baseline_base = nn.Sequential(*baseline_layers)
        self.baseline = BaselineLayer(self.base_out_channels, reward_space, n_value_heads)

    def forward(
            self,
            x: dict[str, Union[dict, torch.Tensor]],
            sample: bool = True
    ) -> dict[str, Any]:
        x, input_mask, available_actions_mask, subtask_embeddings = self.dict_input_layer(x)
        base_out, _ = self.base_model((x, input_mask))
        if subtask_embeddings is not None:
            subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)
        policy_logits, actions = self.actor(
            self.actor_base(base_out),
            available_actions_mask=available_actions_mask,
            sample=sample
        )
        baseline = self.baseline(self.baseline_base(base_out), subtask_embeddings)
        return dict(
            actions=actions,
            policy_logits=policy_logits,
            baseline=baseline
        )

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)
