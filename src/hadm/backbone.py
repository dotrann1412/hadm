"""
EVA-02 Vision Transformer backbone + Simple Feature Pyramid.

Pure PyTorch re-implementation matching the Detectron2/EVA-02 architecture
and state_dict key names used in HADM (arXiv:2411.13842).

No detectron2, no xformers, no mmcv — only torch + torchvision ops.
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm on channel dim of (B, C, H, W) tensors."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Conv2dNorm(nn.Conv2d):
    """Conv2d with an optional ``.norm`` sub-module (matches Detectron2 key naming)."""
    def __init__(self, *args, norm_layer: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.norm(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop entire residual branch)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


# ---------------------------------------------------------------------------
# Vision Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class VisionRotaryEmbedding(nn.Module):
    """2-D rotary position embedding compatible with EVA-02.

    Produces ``freqs_cos`` and ``freqs_sin`` buffers of shape ``(L², head_dim)``
    where ``L = ft_seq_len`` (window or global grid side length).
    """

    def __init__(self, dim: int, pt_seq_len: int = 16,
                 ft_seq_len: Optional[int] = None, theta: float = 10000.0):
        super().__init__()
        ft_seq_len = ft_seq_len or pt_seq_len

        idx = torch.arange(0, dim, 2).float()[: dim // 2]
        base_freqs = 1.0 / (theta ** (idx / dim))

        t = torch.arange(ft_seq_len).float() / ft_seq_len * pt_seq_len
        freqs_1d = torch.outer(t, base_freqs)
        freqs_1d = freqs_1d.repeat_interleave(2, dim=-1)  # (L, dim)

        freqs_h = freqs_1d.unsqueeze(1).expand(-1, ft_seq_len, -1)
        freqs_w = freqs_1d.unsqueeze(0).expand(ft_seq_len, -1, -1)
        freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1).reshape(
            ft_seq_len * ft_seq_len, 2 * dim
        )

        self.register_buffer("freqs_cos", freqs_2d.cos())
        self.register_buffer("freqs_sin", freqs_2d.sin())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.freqs_cos + _rotate_half(t) * self.freqs_sin  # type: ignore


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with separate Q/K/V projections (EVA-02 style).

    - Q and V have learned biases; K has no bias.
    - Optional RoPE applied to Q and K.
    - Uses ``F.scaled_dot_product_attention`` (PyTorch ≥ 2.0).
    """

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor,
                rope: Optional[VisionRotaryEmbedding] = None) -> torch.Tensor:
        B, H, W, C = x.shape
        N = H * W
        x_flat = x.reshape(B, N, C)

        q = F.linear(x_flat, self.q_proj.weight, self.q_bias)
        k = F.linear(x_flat, self.k_proj.weight)
        v = F.linear(x_flat, self.v_proj.weight, self.v_bias)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            q = rope(q).type_as(v)
            k = rope(k).type_as(v)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x.reshape(B, H, W, C)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward with sub-layer norm (EVA-02 default)."""

    def __init__(self, in_features: int, hidden_features: int,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.ffn_ln = norm_layer(hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.ffn_ln(self.act(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Pre-norm transformer block: LN → Attention → residual → LN → SwiGLU → residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.667,
                 drop_path: float = 0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor,
                rope: Optional[VisionRotaryEmbedding] = None,
                window_size: int = 0) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        pad_hw = None
        if window_size > 0:
            x, pad_hw = _window_partition(x, window_size)

        x = self.attn(x, rope=rope)

        if window_size > 0 and pad_hw is not None:
            x = _window_unpartition(x, window_size, pad_hw,
                                    (shortcut.shape[1], shortcut.shape[2]))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Window partitioning helpers
# ---------------------------------------------------------------------------

def _window_partition(x: torch.Tensor, ws: int):
    B, H, W, C = x.shape
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = (x.view(B, Hp // ws, ws, Wp // ws, ws, C)
          .permute(0, 1, 3, 2, 4, 5)
          .reshape(-1, ws, ws, C))
    return x, (Hp, Wp)


def _window_unpartition(x: torch.Tensor, ws: int, pad_hw, orig_hw):
    Hp, Wp = pad_hw
    H, W = orig_hw
    B = x.shape[0] // (Hp // ws * Wp // ws)
    x = (x.view(B, Hp // ws, Wp // ws, ws, ws, -1)
          .permute(0, 1, 3, 2, 4, 5)
          .reshape(B, Hp, Wp, -1))
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)   # (B, H, W, C)


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------

def _get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: tuple):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    old_l = abs_pos.shape[1]
    old_h = old_w = int(math.sqrt(old_l))
    if old_h == h and old_w == w:
        return abs_pos.reshape(1, h, w, -1)
    pos = abs_pos.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2).float()
    pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
    return pos.permute(0, 2, 3, 1)


class ViT(nn.Module):
    """EVA-02 Vision Transformer (state_dict keys: ``blocks.{i}.*``, etc.)."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 2.667,
        drop_path_rate: float = 0.3,
        window_size: int = 16,
        window_block_indexes: tuple = (),
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_block_indexes = set(window_block_indexes)
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        num_patches = (pretrain_img_size // patch_size) ** 2
        num_positions = num_patches + (1 if pretrain_use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

        half_head_dim = embed_dim // num_heads // 2
        grid_size = img_size // patch_size
        self.rope_win = VisionRotaryEmbedding(half_head_dim, 16, window_size)
        self.rope_glb = VisionRotaryEmbedding(half_head_dim, 16, grid_size)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dpr[i], norm_layer)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        Ht, Wt = x.shape[1], x.shape[2]
        if self.pos_embed is not None:
            x = x + _get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (Ht, Wt)
            ).to(x.dtype)
        for i, blk in enumerate(self.blocks):
            is_win = i in self.window_block_indexes
            x = blk(x,
                     rope=self.rope_win if is_win else self.rope_glb,
                     window_size=self.window_size if is_win else 0)
        return x.permute(0, 3, 1, 2)  # (B, C, Ht, Wt)


# ---------------------------------------------------------------------------
# Combined backbone: ViT + Simple Feature Pyramid
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """``backbone.*`` in the HADM checkpoint: ViT + Simple Feature Pyramid.

    Produces multi-scale features ``{p2, p3, p4, p5, p6}`` from a single ViT
    feature map via deconv / pool + 1×1 + 3×3 conv stages (with LN).
    """

    OUT_CHANNELS = 256
    STRIDES = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 2.667,
        drop_path_rate: float = 0.3,
        window_size: int = 16,
        window_block_indexes: tuple = (),
    ):
        super().__init__()
        oc = self.OUT_CHANNELS
        self.net = ViT(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate, window_size=window_size,
            window_block_indexes=window_block_indexes,
        )

        def _ln(c):
            return nn.LayerNorm(c, eps=1e-6)

        # p2  (stride 4) — 2× deconv upsample
        self.simfp_2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),        # 0
            ChannelLayerNorm(embed_dim // 2, eps=1e-6),                 # 1
            nn.GELU(),                                                  # 2
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, 2),   # 3
            Conv2dNorm(embed_dim // 4, oc, 1, bias=False, norm_layer=_ln(oc)),  # 4
            Conv2dNorm(oc, oc, 3, padding=1, bias=False, norm_layer=_ln(oc)),   # 5
        )
        # p3  (stride 8) — 1× deconv upsample
        self.simfp_3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2),        # 0
            Conv2dNorm(embed_dim // 2, oc, 1, bias=False, norm_layer=_ln(oc)),  # 1
            Conv2dNorm(oc, oc, 3, padding=1, bias=False, norm_layer=_ln(oc)),   # 2
        )
        # p4  (stride 16) — identity spatial
        self.simfp_4 = nn.Sequential(
            Conv2dNorm(embed_dim, oc, 1, bias=False, norm_layer=_ln(oc)),       # 0
            Conv2dNorm(oc, oc, 3, padding=1, bias=False, norm_layer=_ln(oc)),   # 1
        )
        # p5  (stride 32) — 2× maxpool downsample
        self.simfp_5 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                                 # 0
            Conv2dNorm(embed_dim, oc, 1, bias=False, norm_layer=_ln(oc)),       # 1
            Conv2dNorm(oc, oc, 3, padding=1, bias=False, norm_layer=_ln(oc)),   # 2
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.net(x)  # (B, C, Ht, Wt)
        out = {
            "p2": self.simfp_2(feat),
            "p3": self.simfp_3(feat),
            "p4": self.simfp_4(feat),
            "p5": self.simfp_5(feat),
        }
        out["p6"] = F.max_pool2d(out["p5"], kernel_size=1, stride=2)
        return out
