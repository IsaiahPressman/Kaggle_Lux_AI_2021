from omegaconf import OmegaConf
from pathlib import Path
import torch
from torch import nn
from typing import *

from ..lux_gym.obs_spaces import MAX_BOARD_SIZE
from .models import BasicActorCriticNetwork
from .in_blocks import DictInputLayer, ConvEmbeddingInputLayer
from .conv_blocks import FullConvResidualBlock


def create_model(flags, device: torch.device) -> nn.Module:
    if flags.model_arch == "dummy_conv_model":
        base_model = nn.Sequential(
            DictInputLayer(),
            ConvEmbeddingInputLayer(
                obs_space=flags.obs_space.get_obs_spec(),
                embedding_dim=flags.dim,
                use_index_select=flags.use_index_select
            ),
            FullConvResidualBlock(
                in_channels=flags.dim,
                out_channels=flags.dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=3
            ),
        )
        model = BasicActorCriticNetwork(
            base_model=base_model,
            base_out_channels=flags.dim,
            action_space=flags.act_space.get_action_space()
        )
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    model.to(device=device)
    return model


def load_model(load_dir: Union[Path, str], model_checkpoint: str, device: torch.device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = Path(load_dir) / (model_checkpoint + ".pt")
    model = create_model(flags, device)
    print(flags.checkpoint)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    return model
