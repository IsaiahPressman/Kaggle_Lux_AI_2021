import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Tuple


class RelPosSelfAttention(nn.Module):
    """
    Relative Position Self Attention
    Forked from: https://gist.github.com/ShoufaChen/ec7b70038a6fdb488da4b34355380569
    """

    def __init__(self, h: int, w: int, dim: int, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = weights + torch.where(
            attn_mask.view(-1, 1, 1, 1, h * w),
            torch.zeros_like(weights),
            torch.zeros_like(weights) + float('-inf')
        )
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(
            q,
            self.rel_emb_w,
            transpose_mask=[0, 1, 2, 4, 3, 5]
        )
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(
            q.permute(0, 1, 3, 2, 4),
            self.rel_emb_h,
            transpose_mask=[0, 1, 4, 2, 5, 3]
        )
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype, device=x.device)
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1])
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype, device=x.device)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class GroupPointWise(nn.Module):
    """"""
    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01)

    def forward(self, x):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        x = x.permute(0, 2, 3, 1)
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', x, self.w)
        return out


class MHSA(nn.Module):
    """
    """
    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative'):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        o = self.self_attention(q=q, k=k, v=v, attn_mask=input_mask).permute(0, 3, 1, 2)
        return o


class ViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mhsa_heads: int,
        height: int,
        width: int,
        normalize: bool = True,
        activation: Callable = nn.GELU
    ):
        super(ViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.mhsa = MHSA(
            in_channels=in_channels,
            heads=mhsa_heads,
            curr_h=height,
            curr_w=width
        )

        self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1)
            ),
            activation(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1)
            )
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, input_mask = x
        identity = x
        x = self.mhsa(self.norm1(x), input_mask)
        x = x + identity
        return (self.mlp(self.norm2(x)) + x) * input_mask, input_mask
