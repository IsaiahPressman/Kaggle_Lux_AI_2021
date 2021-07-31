import numpy as np
import gym.spaces
import torch
from torch import nn
from typing import *


def _index_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    out = embedding_layer.weight.index_select(0, x.view(-1))
    return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    return embedding_layer(x)


def _get_select_func(use_index_select: bool) -> Callable:
    """Use index select instead of default forward to possibly speed up embedding."""
    if use_index_select:
        return _index_select
    else:
        return _forward_select


class DictInputLayer(nn.Module):
    @staticmethod
    def forward(x: dict[str, Union[dict, torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return x["obs"], x["info"]["input_mask"]


class ConvEmbeddingInputLayer(nn.Module):
    def __init__(self, obs_space: gym.spaces.Dict, embedding_dim: int, use_index_select: bool = True):
        super(ConvEmbeddingInputLayer, self).__init__()

        embeddings = {}
        n_continuous_channels = 0
        self.keys_to_op = {}
        for key, val in obs_space.spaces.items():
            if key.endswith("_COUNT"):
                if key[:-6] not in obs_space.spaces.keys():
                    raise ValueError(f"{key} was found in obs_space without an associated {key[:-6]}.")
                self.keys_to_op[key] = "count"
            elif isinstance(val, gym.spaces.MultiBinary):
                assert embedding_dim % np.prod(val.shape[:2]) == 0, f"{key}: {embedding_dim}, {val.shape[:2]}"
                embeddings[key] = nn.Embedding(2, embedding_dim // np.prod(val.shape[:2]), padding_idx=0)
                self.keys_to_op[key] = "embedding"
            elif isinstance(val, gym.spaces.MultiDiscrete):
                assert embedding_dim % np.prod(val.shape[:2]) == 0, f"{embedding_dim}, {val.shape[:2]}"
                if val.nvec.min() != val.nvec.max():
                    raise ValueError(f"MultiDiscrete observation spaces must all have the same number of embeddings. "
                                     f"Found: {np.unique(val.nvec)}")
                raise NotImplementedError("What should be the padding index, if any?")
                embeddings[key] = nn.Embedding(val.nvec.ravel()[0], embedding_dim // np.prod(val.shape[:2]))
                self.keys_to_op[key] = "embedding"
            elif isinstance(val, gym.spaces.Box):
                n_continuous_channels += np.prod(val.shape[:2])
                self.keys_to_op[key] = "continuous"
            else:
                raise NotImplementedError(f"{type(val)} is not an accepted observation space.")

        self.embeddings = nn.ModuleDict(embeddings)
        self.continuous_space_embedding = nn.Conv2d(n_continuous_channels, embedding_dim, (1, 1))
        self.select = _get_select_func(use_index_select)

    def forward(self, x: tuple[dict[str, torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, input_mask = x
        continuous_outs = []
        embedding_outs = {}
        for key, op in self.keys_to_op.items():
            # Input should be of size (b, n, p, x, y) OR (b, n, p)
            # Combine channel and player dims
            out = torch.flatten(x[key], start_dim=1, end_dim=2)
            # Size is now (b, n*p, x, y) or (b, n*p)
            if op == "count":
                embedding_expanded = embedding_outs[key[:-6]]
                embedding_expanded = embedding_expanded.view(-1, 2, embedding_expanded.shape[1] // 2,
                                                             *embedding_expanded.shape[-2:])
                embedding_outs[key[:-6]] = torch.flatten(
                    embedding_expanded * out.unsqueeze(2),
                    start_dim=1,
                    end_dim=2
                )
            elif op == "embedding":
                # Embedding out produces tensor of shape (b, n*p, ..., d/(n*p))
                # This should be reshaped to size (b, d, ...)
                out = self.select(self.embeddings[key], out)
                # In case out is of size (b, n*p, d/(n*p)), expand it to (b, n*p, x, y, d/(n*p))
                if len(out.shape) == 3:
                    out = out.unsqueeze(-2).unsqueeze(-2)
                assert len(out.shape) == 5
                embedding_outs[key] = torch.flatten(
                    out.permute(0, 1, 4, 2, 3),
                    start_dim=1,
                    end_dim=2
                ) * input_mask
            elif op == "continuous":
                if len(out.shape) == 2:
                    out = out.unsqueeze(-1).unsqueeze(-1)
                assert len(out.shape) == 4
                continuous_outs.append(out * input_mask)
            else:
                raise RuntimeError(f"Unknown operation: {op}")
            """
            while len(val.shape) < len(input_mask.shape):
                val = val.unsqueeze(-1)
            val = val.expand_as(input_mask)
            # Size is now (b, n*p, x, y)
            assert len(val.shape) == len(input_mask.shape)
            """
        continuous_out_combined = self.continuous_space_embedding(torch.cat(continuous_outs, dim=1))
        embedding_outs_combined = torch.stack([v for v in embedding_outs.values()], dim=-1).sum(dim=-1)
        return continuous_out_combined + embedding_outs_combined, input_mask
