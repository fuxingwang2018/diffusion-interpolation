# networks/unet.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_atten import GroupNorm, SelfAttention2D


class FiLM(nn.Module):
    def __init__(self, emb_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * ch))

    def forward(self, emb: torch.Tensor):
        s, b = self.net(emb).chunk(2, dim=1)
        return s, b


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0, use_gn: bool = True):
        super().__init__()
        Norm = GroupNorm if use_gn else nn.BatchNorm2d
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = Norm(out_ch)
        self.film = FiLM(emb_dim, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = Norm(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.c1(x)
        s, b = self.film(emb)
        h = self.n1(h) * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        h = self.drop(h)
        h = self.c2(h)
        h = self.n2(h)
        return F.silu(h + self.skip(x))


class UpLevel(nn.Module):
    """
    One upsampling level:
      - upsample: ConvTranspose2d (or Identity for the top)
      - merge: 1x1 conv to bring cat(h, skip) from 2*cur -> cur
      - blocks: ResBlocks (+ optional attention) that follow after merge
    """
    def __init__(
        self,
        cur_ch: int,
        out_ch: int,
        emb_dim: int,
        num_res_blocks: int,
        num_heads: int,
        use_gn: bool,
        use_attn: bool,
        do_upsample: bool,
        dropout: float,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(cur_ch, cur_ch, 4, stride=2, padding=1) if do_upsample else nn.Identity()
        # After upsample we concatenate skip: channels become cur_ch + cur_ch = 2*cur_ch
        self.merge = nn.Conv2d(cur_ch * 2, cur_ch, kernel_size=1) if do_upsample else nn.Identity()

        blocks = []
        ch_in = cur_ch  # after merge projection
        for _ in range(num_res_blocks + 1):
            blocks.append(ResBlock(ch_in, out_ch, emb_dim, dropout=dropout, use_gn=use_gn))
            ch_in = out_ch
            if use_attn:
                blocks.append(SelfAttention2D(ch_in, num_heads=num_heads, use_gn=use_gn))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, h: torch.Tensor, emb: torch.Tensor, skip: torch.Tensor | None):
        # upsample
        h = self.upsample(h)
        # merge with skip (if provided)
        if not isinstance(self.merge, nn.Identity) and skip is not None:
            if skip.shape[-2:] != h.shape[-2:]:
                skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = self.merge(h)
        # run blocks
        for m in self.blocks:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        return h


class UNet2D(nn.Module):
    """
    Pure nn.Module UNet backbone with FiLM conditioning.
    Fixed up path: after concatenating a skip, we project 2*cur -> cur via 1x1 conv.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_ch: int = 128,
        ch_mults: tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        emb_dim: int = 256,
        attn_resolutions: tuple[int, ...] = (8, 16),
        num_heads: int = 4,
        dropout: float = 0.0,
        use_gn: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.ch_mults = ch_mults
        self.num_res_blocks = num_res_blocks
        self.emb_dim = emb_dim
        self.attn_resolutions = attn_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_gn = use_gn
        
        chs = [base_ch * m for m in ch_mults]

        # Stem
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # ----------------
        # Down path (store features AFTER each downsample conv to use as skips)
        # ----------------
        downs = []
        cur = chs[0]
        for i, ch in enumerate(chs):
            # residual blocks at current scale
            for _ in range(num_res_blocks):
                downs.append(ResBlock(cur, ch, emb_dim, dropout, use_gn))
                cur = ch
                if (2 ** i) in attn_resolutions:
                    downs.append(SelfAttention2D(cur, num_heads, use_gn=use_gn))
            # downsample except for the last scale; we will store the downsampled feature as a skip
            if i != len(chs) - 1:
                downs.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))
        self.down = nn.ModuleList(downs)

        # ----------------
        # Middle
        # ----------------
        mid_blocks = [
            ResBlock(cur, cur, emb_dim, dropout, use_gn),
            SelfAttention2D(cur, num_heads, use_gn=use_gn) if (2 ** (len(chs) - 1) in attn_resolutions) else nn.Identity(),
            ResBlock(cur, cur, emb_dim, dropout, use_gn),
        ]
        self.mid = nn.ModuleList(mid_blocks)

        # ----------------
        # Up path (structured as levels with explicit merge conv)
        # ----------------
        up_levels = []
        for i, ch in list(reversed(list(enumerate(chs)))):
            use_attn = (2 ** i) in attn_resolutions
            do_upsample = i != 0  # all but the topmost level
            # Each level takes current channels (cur) and produces ch at this scale
            up_levels.append(
                UpLevel(
                    cur_ch=cur,
                    out_ch=ch,
                    emb_dim=emb_dim,
                    num_res_blocks=num_res_blocks,
                    num_heads=num_heads,
                    use_gn=use_gn,
                    use_attn=use_attn,
                    do_upsample=do_upsample,
                    dropout=dropout,
                )
            )
            cur = ch  # update for the next (higher) level
        self.up_levels = nn.ModuleList(up_levels)

        # Head
        self.out = nn.Sequential(
            GroupNorm(cur) if use_gn else nn.BatchNorm2d(cur),
            nn.SiLU(),
            nn.Conv2d(cur, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = self.in_conv(x)

        # down: run modules; whenever we hit a stride-2 conv, push the result as a skip
        for m in self.down:
            if isinstance(m, ResBlock):
                h = m(h, emb)
            else:
                h = m(h)
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    skips.append(h)

        # mid
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)

        # up: traverse levels; each level consumes one skip (except the top level which has no upsample/merge)
        for lvl in self.up_levels:
            skip = skips.pop() if not isinstance(lvl.upsample, nn.Identity) and skips else None
            h = lvl(h, emb, skip)

        return self.out(h)
