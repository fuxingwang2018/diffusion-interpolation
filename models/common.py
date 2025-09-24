from __future__ import annotations
from typing import Optional
import math
import torch
from torch import nn
import torch.nn.functional as F

# ---------- small helpers ----------

def exists(x): return x is not None

def default(val, dflt):
    return val if exists(val) else (dflt() if callable(dflt) else dflt)

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding for a 1D scalar per-batch (e.g., log-σ, or normalized t).
    t: (B,)
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / max(half - 1, 1)))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

 

# ---------- compact U-Net with FiLM ----------

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, c_out * 2))  # gamma, beta
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        gamma, beta = self.emb(emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(self.act(h))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.pool = nn.Conv2d(c, c, 3, stride=2, padding=1)
    def forward(self, x): return self.pool(x)

class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class SelfAttention2D(nn.Module):
    """
    Multi-head self-attention over 2D feature maps.
    Follows the spirit of lucidrains' DDPM attention:
      - 1x1 qkv projections
      - scaled dot-product attention across HW tokens
      - final 1x1 projection + residual
    """
    def __init__(self, channels: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, inner * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        qkv = self.to_qkv(x)                         # (B, 3*inner, H, W)
        q, k, v = qkv.chunk(3, dim=1)                # (B, inner, H, W) each

        # reshape to (B, heads, dim_head, N)
        def reshape_heads(t):
            b, c, h, w = t.shape
            t = t.view(b, self.heads, c // self.heads, h * w)
            return t
        q, k, v = map(reshape_heads, (q, k, v))

        q = q * self.scale
        # attn = softmax(q^T k) over tokens
        # (B, heads, dim, N) @ (B, heads, dim, N) -> (B, heads, N, N)
        attn = torch.einsum('b h d n, b h d m -> b h n m', q, k).softmax(dim=-1)

        # out = (attn @ v^T)^T => (B, heads, dim, N)
        out = torch.einsum('b h n m, b h d m -> b h d n', attn, v)

        # merge heads back to (B, inner, H, W)
        out = out.contiguous().view(b, -1, h, w)
        out = self.to_out(out)
        return out + x_in


class UNet2D(nn.Module):
    """
    Tiny U-Net with FiLM from time/noise embedding + optional 2D attention.
    in_ch: channels of [noisy_target || conditioning ]
    out_ch: channels of target only (predict noise on targets)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base: int = 64,
        emb_dim: int = 256,
        # --- NEW attention controls ---
        use_attention: bool = False,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        # where to apply attention: subset of {"down1","down2","mid","up1","up2"}
        attn_levels: tuple[str, ...] = ("mid",),
    ):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.attn_levels = set(attn_levels)

        # time embedding MLP (FiLM)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # ---------- Encoder ----------
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, emb_dim)          # level: base
        self.down1 = Down(base)                           # /2
        self.rb2 = ResBlock(base, base*2, emb_dim)        # level: base*2
        self.down2 = Down(base*2)                         # /4
        self.rb3 = ResBlock(base*2, base*4, emb_dim)      # bottleneck in our 3-level UNet

        # (optional) attention in encoder/bottleneck
        if self.use_attention:
            self.attn_down1 = SelfAttention2D(base)              if "down1" in self.attn_levels else nn.Identity()
            self.attn_down2 = SelfAttention2D(base*2, attn_heads, attn_dim_head) if "down2" in self.attn_levels else nn.Identity()
            self.attn_mid   = SelfAttention2D(base*4, attn_heads, attn_dim_head) if "mid"   in self.attn_levels else nn.Identity()
        else:
            self.attn_down1 = self.attn_down2 = self.attn_mid = nn.Identity()

        # ---------- Decoder ----------
        self.up2 = Up(base*4, base*2)
        self.rb4 = ResBlock(base*4, base*2, emb_dim)      # concat with h2
        self.up1 = Up(base*2, base)
        self.rb5 = ResBlock(base*2, base, emb_dim)        # concat with h1
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

        # (optional) attention in decoder
        if self.use_attention:
            self.attn_up2 = SelfAttention2D(base*2, attn_heads, attn_dim_head) if "up2" in self.attn_levels else nn.Identity()
            self.attn_up1 = SelfAttention2D(base,   attn_heads, attn_dim_head) if "up1" in self.attn_levels else nn.Identity()
        else:
            self.attn_up2 = self.attn_up1 = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t_emb)  # (B, emb_dim)

        # enc
        h0 = self.in_conv(x)
        h1 = self.rb1(h0, temb)
        h1 = self.attn_down1(h1)                 # optional
        h2 = self.rb2(self.down1(h1), temb)
        h2 = self.attn_down2(h2)                 # optional
        h3 = self.rb3(self.down2(h2), temb)
        h3 = self.attn_mid(h3)                   # optional

        # dec
        u2 = self.up2(h3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb4(u2, temb)
        u2 = self.attn_up2(u2)                   # optional

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb5(u1, temb)
        u1 = self.attn_up1(u1)                   # optional

        return self.out(u1)