import logging

import torch
from torch import nn

from .models import BasicActorCriticNetwork
from .in_blocks import ConvEmbeddingInputLayer
from .attn_blocks import ViTBlock, RPSA, GPSA
from .conv_blocks import ResidualBlock, ParallelDilationResidualBlock
from .unet import UNET
from ..lux_gym.obs_spaces import SUBTASK_ENCODING
from ..lux_gym.multi_subtask import MultiSubtask
from ..utility_constants import MAX_BOARD_SIZE


def create_model(flags, device: torch.device) -> nn.Module:
    obs_space = flags.obs_space()
    act_space = flags.act_space()
    if flags.model_arch == "conv_model":
        base_model = nn.Sequential(
            ConvEmbeddingInputLayer(
                obs_space=obs_space.get_obs_spec(),
                embedding_dim=flags.hidden_dim,
                n_merge_layers=flags.n_merge_layers,
                use_index_select=flags.use_index_select
            ),
            *[ResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "pd_conv_model":
        logging.warning("Dilation is slow for some Pytorch/CUDNN versions.")
        base_model = nn.Sequential(
            ConvEmbeddingInputLayer(
                obs_space=obs_space.get_obs_spec(),
                embedding_dim=flags.hidden_dim,
                n_merge_layers=flags.n_merge_layers,
                use_index_select=flags.use_index_select
            ),
            *[ParallelDilationResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "unet_model":
        base_model = nn.Sequential(
            ConvEmbeddingInputLayer(
                obs_space=obs_space.get_obs_spec(),
                embedding_dim=flags.hidden_dim,
                n_merge_layers=flags.n_merge_layers,
                use_index_select=flags.use_index_select
            ),
            UNET(
                n_blocks_per_reduction=flags.n_blocks_per_reduction,
                in_out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                # Residual block kwargs
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            )
        )
    elif flags.model_arch == "RPSA_model":
        base_model = nn.Sequential(
            ConvEmbeddingInputLayer(
                obs_space=obs_space.get_obs_spec(),
                embedding_dim=flags.hidden_dim,
                n_merge_layers=flags.n_merge_layers,
                use_index_select=flags.use_index_select
            ),
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                mhsa_layer=RPSA(
                    in_channels=flags.hidden_dim,
                    heads=flags.n_heads,
                    height=MAX_BOARD_SIZE[0],
                    width=MAX_BOARD_SIZE[1]
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "GPSA_model":
        base_model = nn.Sequential(
            ConvEmbeddingInputLayer(
                obs_space=obs_space.get_obs_spec(),
                embedding_dim=flags.hidden_dim,
                n_merge_layers=flags.n_merge_layers,
                use_index_select=flags.use_index_select
            ),
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                mhsa_layer=GPSA(
                    dim=flags.hidden_dim,
                    n_heads=flags.n_heads,
                    height=MAX_BOARD_SIZE[0],
                    width=MAX_BOARD_SIZE[1]
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    if flags.reward_space is MultiSubtask:
        n_value_heads = len(SUBTASK_ENCODING)
    else:
        n_value_heads = 1

    model = BasicActorCriticNetwork(
        base_model=base_model,
        base_out_channels=flags.hidden_dim,
        action_space=act_space.get_action_space(),
        reward_space=flags.reward_space.get_reward_spec(),
        n_value_heads=n_value_heads,
        rescale_value_input=flags.rescale_value_input
    )
    return model.to(device=device)
