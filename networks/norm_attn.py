# diffusion/models/norm_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, max(1, num_channels // min_channels_per_group))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)

class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4, use_gn=True):
        super().__init__()
        assert channels % num_heads == 0
        self.h = num_heads; self.scale = (channels // num_heads) ** -0.5
        self.norm = GroupNorm(channels) if use_gn else nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3*channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.norm(x)
        q,k,v = self.qkv(x).view(b, self.h, 3*(c//self.h), h*w).chunk(3, dim=2)
        q = q.permute(0,1,3,2) * self.scale   # (B,H,HW,C/H)
        k = k.permute(0,1,3,2)
        v = v.permute(0,1,3,2)
        attn = torch.softmax(torch.einsum('bhqc,bhkc->bhqk', q, k), dim=-1)
        out  = torch.einsum('bhqk,bhkc->bhqc', attn, v).permute(0,1,3,2).contiguous().view(b,c,h,w)
        return self.proj(out) + x
