"""Pure-PyTorch port of the mmpose ``ViT`` backbone used by SMPLer-X.

Matches ``main/transformer_utils/mmpose/models/backbones/vit.py`` for the
SMPLer-X-H32 config (img 256x192, patch 16, embed 1280, depth 32, heads 16,
31 task tokens) with no mmcv/mmpose dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdanyone_smplerx.config import DEPTH, DROP_PATH_RATE, FEAT_DIM, MLP_RATIO, NUM_HEADS, TASK_TOKENS_NUM


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0, b=2.0)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob is None or self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x / keep * random


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: float | None = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: tuple[int, int] = (256, 192), patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, ratio: int = 1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (ratio**2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size // ratio, padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int] = (256, 192),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = FEAT_DIM,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        mlp_ratio: float = MLP_RATIO,
        qkv_bias: bool = True,
        drop_path_rate: float = DROP_PATH_RATE,
        task_tokens_num: int = TASK_TOKENS_NUM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.task_tokens_num = task_tokens_num

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.task_tokens = nn.Parameter(torch.zeros(1, task_tokens_num, embed_dim))
        trunc_normal_(self.task_tokens)

        # +1 for the pretrained class-token slot
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed)

        dpr = [item for item in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=dpr[i]) for i in range(depth)]
        )
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        task_tokens = self.task_tokens.repeat(B, 1, 1)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        x = torch.cat((task_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)

        task_out = x[:, : self.task_tokens_num]
        xp = x[:, self.task_tokens_num :]
        xp = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp, task_out
