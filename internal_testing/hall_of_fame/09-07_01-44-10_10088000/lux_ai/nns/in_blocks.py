import gym.spaces
import numpy as np
import torch
from torch import nn
from typing import Callable, Dict, Optional, Tuple, Union


def _index_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    out = embedding_layer.weight.index_select(0, x.view(-1))
    return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    return embedding_layer(x)


def _get_select_func(use_index_select: bool) -> Callable:
    """
    Use index select instead of default forward to possibly speed up embedding.
    NB: This disables padding_idx functionality
    """
    if use_index_select:
        return _index_select
    else:
        return _forward_select


class DictInputLayer(nn.Module):
    @staticmethod
    def forward(
            x: Dict[str, Union[Dict, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        return (x["obs"],
                x["info"]["input_mask"],
                x["info"]["available_actions_mask"],
                x["info"].get("subtask_embeddings", None))


class ConvEmbeddingInputLayer(nn.Module):
    def __init__(
            self,
            obs_space: gym.spaces.Dict,
            embedding_dim: int,
            n_merge_layers: int = 1,
            use_index_select: bool = True,
            activation: Callable = nn.LeakyReLU
    ):
        super(ConvEmbeddingInputLayer, self).__init__()

        embeddings = {}
        n_continuous_channels = 0
        n_embedding_channels = 0
        self.keys_to_op = {}
        for key, val in obs_space.spaces.items():
            assert val.shape[0] == 1
            if key.endswith("_COUNT"):
                if key[:-6] not in obs_space.spaces.keys():
                    raise ValueError(f"{key} was found in obs_space without an associated {key[:-6]}.")
                self.keys_to_op[key] = "count"
            elif isinstance(val, gym.spaces.MultiBinary) or isinstance(val, gym.spaces.MultiDiscrete):
                # assert embedding_dim % np.prod(val.shape[:2]) == 0, f"{key}: {embedding_dim}, {val.shape[:2]}"
                if isinstance(val, gym.spaces.MultiBinary):
                    n_embeddings = 2
                    padding_idx = 0
                elif isinstance(val, gym.spaces.MultiDiscrete):
                    if val.nvec.min() != val.nvec.max():
                        raise ValueError(
                            f"MultiDiscrete observation spaces must all have the same number of embeddings. "
                            f"Found: {np.unique(val.nvec)}")
                    n_embeddings = val.nvec.ravel()[0]
                    padding_idx = None
                else:
                    raise NotImplementedError(f"Got gym space: {type(val)}")
                n_players = val.shape[1]
                n_embeddings = n_players * (n_embeddings - 1) + 1
                embeddings[key] = nn.Embedding(n_embeddings, embedding_dim, padding_idx=padding_idx)
                n_embedding_channels += embedding_dim
                self.keys_to_op[key] = "embedding"
            elif isinstance(val, gym.spaces.Box):
                n_continuous_channels += np.prod(val.shape[:2])
                self.keys_to_op[key] = "continuous"
            else:
                raise NotImplementedError(f"{type(val)} is not an accepted observation space.")

        self.embeddings = nn.ModuleDict(embeddings)
        continuous_space_embedding_layers = []
        embedding_merger_layers = []
        merger_layers = []
        for i in range(n_merge_layers - 1):
            continuous_space_embedding_layers.extend([
                nn.Conv2d(n_continuous_channels, n_continuous_channels, (1, 1)),
                activation()
            ])
            embedding_merger_layers.extend([
                nn.Conv2d(n_embedding_channels, n_embedding_channels, (1, 1)),
                activation()
            ])
            merger_layers.extend([
                nn.Conv2d(embedding_dim * 2, embedding_dim * 2, (1, 1)),
                activation()
            ])
        continuous_space_embedding_layers.extend([
            nn.Conv2d(n_continuous_channels, embedding_dim, (1, 1)),
            activation()
        ])
        embedding_merger_layers.extend([
            nn.Conv2d(n_embedding_channels, embedding_dim, (1, 1)),
            activation()
        ])
        merger_layers.append(nn.Conv2d(embedding_dim * 2, embedding_dim, (1, 1)))
        self.continuous_space_embedding = nn.Sequential(*continuous_space_embedding_layers)
        self.embedding_merger = nn.Sequential(*embedding_merger_layers)
        self.merger = nn.Sequential(*merger_layers)
        self.select = _get_select_func(use_index_select)

    def forward(self, x: Tuple[Dict[str, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expects x to be a dictionary of tensors of shape (b, n, p|1, x, y) or (b, n, p|1)
        Returns an output of shape (b * 2, embedding_dim, x, y) where the observation has been duplicated and the
        player axes swapped for the opposing players
        """
        x, input_mask = x
        input_mask = torch.repeat_interleave(input_mask, 2, dim=0)
        continuous_outs = []
        embedding_outs = {}
        for key, op in self.keys_to_op.items():
            # Input should be of size (b, n, p|1, x, y) OR (b, n, p|1)
            in_tensor = x[key]
            assert in_tensor.shape[2] <= 2
            # First we duplicate each batch entry and swap player axes when relevant
            in_tensor = in_tensor[
                        :,
                        :,
                        [np.arange(in_tensor.shape[2]), np.arange(in_tensor.shape[2])[::-1]],
                        ...
                        ]
            # Then we swap the new dims and channel dims so we can combine them with the batch dims
            in_tensor = torch.flatten(
                in_tensor.transpose(1, 2),
                start_dim=0,
                end_dim=1
            )
            # Finally, combine channel and player dims
            out = torch.flatten(in_tensor, start_dim=1, end_dim=2)
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
                # Embedding out produces tensor of shape (b, p|1, ..., d)
                # This should be reshaped to size (b, d, ...)
                # First, we take all embeddings from the opponent and increment them
                if out.shape[1] == 2:
                    # noinspection PyTypeChecker
                    out[:, 1] = torch.where(
                        out[:, 1] != 0,
                        out[:, 1] + (self.embeddings[key].num_embeddings - 1) // 2,
                        out[:, 1]
                    )
                out = self.select(self.embeddings[key], out)
                # In case out is of size (b, p, d), expand it to (b, p, 1, 1, d)
                if len(out.shape) == 3:
                    out = out.unsqueeze(-2).unsqueeze(-2)
                assert len(out.shape) == 5
                embedding_outs[key] = out.permute(0, 1, 4, 2, 3).sum(dim=1) * input_mask
            elif op == "continuous":
                if len(out.shape) == 2:
                    out = out.unsqueeze(-1).unsqueeze(-1)
                assert len(out.shape) == 4
                continuous_outs.append(out * input_mask)
            else:
                raise RuntimeError(f"Unknown operation: {op}")
        continuous_out_combined = self.continuous_space_embedding(torch.cat(continuous_outs, dim=1))
        embedding_outs_combined = self.embedding_merger(torch.cat([v for v in embedding_outs.values()], dim=1))
        merged_outs = self.merger(torch.cat([continuous_out_combined, embedding_outs_combined], dim=1))
        return merged_outs, input_mask
