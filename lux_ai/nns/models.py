import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import *


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
            key: space.nvec.min() for key, space in action_space.spaces.items()
        }
        # An action plane shape usually takes the form (n, 2), where n >= 1 and is used when multiple stacked units
        # must output different actions. The 2 players also use different action planes
        self.action_plane_shapes = {
            key: space.shape[:-2] for key, space in action_space.spaces.items()
        }
        assert all([aps.ndim == 2 for aps in self.action_plane_shapes.values()])
        self.actors = nn.ModuleDict({
            key: nn.Conv2d(
                in_channels,
                n_act * np.prod(self.action_plane_shapes[key]),
                (1, 1)
            ) for key, n_act in self.n_actions.items()
        })

    def forward(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        policy_logits_out = {}
        actions_out = {}
        b, _, h, w = x.shape
        for key, actor in self.actors.items():
            n_actions = self.n_actions[key]
            action_plane_shape = self.action_plane_shapes[key]
            logits = actor(x).view(b, n_actions, *action_plane_shape, h, w)
            # Move the logits dimension to the end
            actions = self.logits_to_actions(logits.permute(0, 2, 3, 4, 5, 1).view(-1, n_actions))
            policy_logits_out[key] = logits
            actions_out[key] = actions
        return policy_logits_out, actions_out

    def logits_to_actions(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        else:
            return logits.argmax(dim=-1)


class ValueActivation(nn.Module):
    def __init__(self, dim: int):
        super(ValueActivation, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rescale to [-1, 1]
        return 2 * F.softmax(x, self.dim) - 1.


class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            base_out_channels: int,
            action_space: gym.spaces.Dict,
            actor_critic_activation: Callable = nn.ReLU,
            n_action_value_layers: int = 2,
            final_value_activation: nn.Module = ValueActivation(dim=-1),
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.base_model = base_model
        self.base_out_channels = base_out_channels

        actor_layers = []
        baseline_layers = []
        for i in range(n_action_value_layers - 1):
            actor_layers.append(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            actor_layers.append(actor_critic_activation())
            baseline_layers.append(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            baseline_layers.append(actor_critic_activation())

        actor_layers.append(DictActor(self.base_out_channels, action_space))
        self.actor = nn.Sequential(*actor_layers)

        baseline_layers.append(nn.AdaptiveAvgPool2d(1))
        baseline_layers.append(nn.Linear(self.base_out_channels, 2))
        baseline_layers.append(final_value_activation)
        self.baseline = nn.Sequential(*baseline_layers)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> dict[str, Any]:
        base_out, _ = self.base(x, input_mask)
        policy_logits, actions = self.actor(base_out)
        baseline = self.baseline(base_out)
        return dict(
            actions=actions,
            policy_logits=policy_logits,
            baseline=baseline
        )
