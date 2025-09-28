# networks/norm_atten.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_info

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
    def __init__(self, channels, num_heads=4, use_gn=True, attn_dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.h = num_heads
        self.d = channels // num_heads
        self.use_gn = use_gn
        self.attn_dropout = attn_dropout

        self.norm = GroupNorm(channels) if use_gn else nn.BatchNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  -> returns (B, C, H, W)
        Uses F.scaled_dot_product_attention for memory efficiency when available.
        """
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x)  # (B, 3C, H, W)
        # reshape -> (B, 3, heads, head_dim, HW)
        qkv = qkv.view(b, 3, self.h, self.d, h * w)
        q, k, v = qkv.unbind(dim=1)  # each: (B, H, D, HW)

        # SDPA expects (..., L, E) x (..., S, E) -> (..., L, E)
        # reshape to (B*H, HW, D)
        q = q.permute(0, 1, 3, 2).reshape(b * self.h, h * w, self.d)  # (B*H, HW, D)
        k = k.permute(0, 1, 3, 2).reshape(b * self.h, h * w, self.d)  # (B*H, HW, D)
        v = v.permute(0, 1, 3, 2).reshape(b * self.h, h * w, self.d)  # (B*H, HW, D)

        out: torch.Tensor
        try:
            # Prefer flash/memory-efficient kernels when available
            # You can also force-enable with:
            #   torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )  # (B*H, HW, D)
        except Exception:
            # Fallback: naive attention (may OOM on large HW)
            # attn = softmax(q @ k^T / sqrt(D)) @ v
            scale = self.d ** -0.5
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # (B*H, HW, HW)
            attn = scores.softmax(dim=-1)
            if self.attn_dropout and self.training:
                attn = F.dropout(attn, p=self.attn_dropout)
            out = torch.matmul(attn, v)  # (B*H, HW, D)

        # back to (B, C, H, W)
        out = out.reshape(b, self.h, h * w, self.d).permute(0, 1, 3, 2).contiguous()
        out = out.view(b, c, h, w)
        return self.proj(out) + residual
