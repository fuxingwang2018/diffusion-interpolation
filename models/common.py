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

def make_coord_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return (2, H, W) normalized coords in [-1,1]."""
    ys = torch.linspace(-1., 1., steps=h, device=device)
    xs = torch.linspace(-1., 1., steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)

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

class UNet2D(nn.Module):
    """
    Tiny U-Net with FiLM from time/noise embedding.
    in_ch: channels of [noisy_target || conditioning || optional extras]
    out_ch: channels of target only (predict noise on targets)
    """
    def __init__(self, in_ch: int, out_ch: int, base: int = 64, emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        # Enc
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, emb_dim)
        self.down1 = Down(base)
        self.rb2 = ResBlock(base, base*2, emb_dim)
        self.down2 = Down(base*2)
        self.rb3 = ResBlock(base*2, base*4, emb_dim)
        # Dec
        self.up2 = Up(base*4, base*2)
        self.rb4 = ResBlock(base*4, base*2, emb_dim)
        self.up1 = Up(base*2, base)
        self.rb5 = ResBlock(base*2, base, emb_dim)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t_emb)
        h0 = self.in_conv(x)
        h1 = self.rb1(h0, temb)
        h2 = self.rb2(self.down1(h1), temb)
        h3 = self.rb3(self.down2(h2), temb)
        u2 = self.up2(h3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb4(u2, temb)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb5(u1, temb)
        return self.out(u1)
