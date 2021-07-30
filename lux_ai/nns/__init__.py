# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf
from pathlib import Path
import torch
from torch import nn
from typing import *

from ..lux_gym.obs_spaces import ObsSpace, MAX_BOARD_SIZE
from ..lux_gym import act_spaces
from .models import BasicActorCriticNetwork
from .in_blocks import ConvEmbeddingInputLayer
from .conv_blocks import FullConvResidualBlock


def create_model(flags, device: torch.device) -> nn.Module:
    obs_space = ObsSpace[flags.obs_space]
    act_space = act_spaces.__dict__[flags.act_space]()
    assert isinstance(act_space, act_spaces.BaseActSpace), f"{act_space}"

    if flags.model_arch == "dummy_conv_model":
        base_model = nn.Sequential(
            ConvEmbeddingInputLayer(
                obs_space=obs_space,
                embedding_dim=flags.dim,
                use_index_select=flags.use_index_select
            ),
            FullConvResidualBlock(
                in_channels=flags.dim,
                out_channels=flags.dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1]
            ),
        )
        model = BasicActorCriticNetwork(
            base_model=base_model,
            base_out_channels=flags.dim,
            action_space=act_space
        )
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    model.to(device=device)
    return model


def load_model(load_dir: Union[Path, str], device: torch.device):
    flags = OmegaConf.load(load_dir + "/config.yaml")
    flags.checkpoint = load_dir + "/checkpoint.tar"
    model = create_model(flags, device)
    print(flags.checkpoint)
    checkpoint_states = torch.load(flags.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    return model
