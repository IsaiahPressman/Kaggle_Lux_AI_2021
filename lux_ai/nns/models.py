import gym
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union

from .in_blocks import DictInputLayer
from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.act_spaces import MAX_OVERLAPPING_ACTIONS
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
            available_actions_mask: Dict[str, torch.Tensor],
            sample: bool,
            actions_per_square: Optional[int] = MAX_OVERLAPPING_ACTIONS
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
            # We first have to cast it to an int tensor to avoid errors in kaggle environment
            aam = available_actions_mask[key]
            orig_dtype = aam.dtype
            aam_new_type = aam.to(dtype=torch.int64)
            aam_filled = torch.where(
                (~aam).all(dim=-1, keepdim=True),
                torch.ones_like(aam_new_type),
                aam_new_type.to(dtype=torch.int64)
            ).to(orig_dtype)
            assert logits.shape == aam_filled.shape
            logits = logits + torch.where(
                aam_filled,
                torch.zeros_like(logits),
                torch.zeros_like(logits) + float("-inf")
            )
            actions = DictActor.logits_to_actions(logits.view(-1, n_actions), sample, actions_per_square)
            policy_logits_out[key] = logits
            actions_out[key] = actions.view(*logits.shape[:-1], -1)
        return policy_logits_out, actions_out

    @staticmethod
    @torch.no_grad()
    def logits_to_actions(logits: torch.Tensor, sample: bool, actions_per_square: Optional[int]) -> torch.Tensor:
        if actions_per_square is None:
            actions_per_square = logits.shape[-1]
        if sample:
            probs = F.softmax(logits, dim=-1)
            # In case there are fewer than MAX_OVERLAPPING_ACTIONS available actions, we add a small eps value
            probs = torch.where(
                (probs > 0.).sum(dim=-1, keepdim=True) >= actions_per_square,
                probs,
                probs + 1e-10
            )
            return torch.multinomial(
                probs,
                num_samples=min(actions_per_square, probs.shape[-1]),
                replacement=False
            )
        else:
            return logits.argsort(dim=-1, descending=True)[..., :actions_per_square]


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
    def __init__(self, in_channels: int, reward_space: RewardSpec, n_value_heads: int, rescale_input: bool):
        super(BaselineLayer, self).__init__()
        assert n_value_heads >= 1
        self.reward_min = reward_space.reward_min
        self.reward_max = reward_space.reward_max
        self.multi_headed = n_value_heads > 1
        self.rescale_input = rescale_input
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

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor, value_head_idxs: Optional[torch.Tensor]):
        """
        Expects an input of shape b * 2, n_channels, x, y
        Returns an output of shape b, 2
        """
        # Average feature planes
        if self.rescale_input:
            x = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1)
            x = x / torch.flatten(input_mask, start_dim=-2, end_dim=-1).sum(dim=-1)
        else:
            x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
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
            rescale_value_input: bool = True
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.dict_input_layer = DictInputLayer()
        self.base_model = base_model
        self.base_out_channels = base_out_channels

        if n_action_value_layers < 2:
            raise ValueError("n_action_value_layers must be >= 2 in order to use spectral_norm")

        """
        actor_layers = []
        baseline_layers = []
        for i in range(n_action_value_layers - 1):
            actor_layers.append(
                nn.utils.spectral_norm(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            )
            actor_layers.append(actor_critic_activation())
            baseline_layers.append(
                nn.utils.spectral_norm(nn.Conv2d(self.base_out_channels, self.base_out_channels, (1, 1)))
            )
            baseline_layers.append(actor_critic_activation())

        self.actor_base = nn.Sequential(*actor_layers)
        self.actor = DictActor(self.base_out_channels, action_space)

        self.baseline_base = nn.Sequential(*baseline_layers)"""

        self.actor_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=self.base_out_channels,
            activation=actor_critic_activation
        )
        self.actor = DictActor(self.base_out_channels, action_space)

        self.baseline_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=self.base_out_channels,
            activation=actor_critic_activation
        )
        self.baseline = BaselineLayer(
            in_channels=self.base_out_channels,
            reward_space=reward_space,
            n_value_heads=n_value_heads,
            rescale_input=rescale_value_input
        )

    def forward(
            self,
            x: Dict[str, Union[dict, torch.Tensor]],
            sample: bool = True,
            **actor_kwargs
    ) -> Dict[str, Any]:
        x, input_mask, available_actions_mask, subtask_embeddings = self.dict_input_layer(x)
        base_out, input_mask = self.base_model((x, input_mask))
        if subtask_embeddings is not None:
            subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)
        policy_logits, actions = self.actor(
            self.actor_base(base_out),
            available_actions_mask=available_actions_mask,
            sample=sample,
            **actor_kwargs
        )
        baseline = self.baseline(self.baseline_base(base_out), input_mask, subtask_embeddings)
        return dict(
            actions=actions,
            policy_logits=policy_logits,
            baseline=baseline
        )

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)

    @staticmethod
    def make_spectral_norm_head_base(n_layers: int, n_channels: int, activation: Callable) -> nn.Module:
        """
        Returns the base of an action or value head, with the final layer of the base/the semifinal layer of the
        head spectral normalized.
        NB: this function actually returns a base with n_layer - 1 layers, leaving the final layer to be filled in
        with the proper action or value output layer.
        """
        assert n_layers >= 2
        layers = []
        for i in range(n_layers - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, (1, 1)))
            layers.append(activation())
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, (1, 1)))
        )
        layers.append(activation())

        return nn.Sequential(*layers)
