# diffusion/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_attn import GroupNorm, SelfAttention2D

class FiLM(nn.Module):
    def __init__(self, emb_dim, ch):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2*ch))
    def forward(self, emb):
        s,b = self.net(emb).chunk(2, dim=1); return s,b

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0, use_gn=True):
        super().__init__()
        Norm = GroupNorm if use_gn else nn.BatchNorm2d
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = Norm(out_ch)
        self.f  = FiLM(emb_dim, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = Norm(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout)
    def forward(self, x, emb):
        h = self.c1(x); s,b = self.f(emb)
        h = self.n1(h)*(1+s[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h); h = self.drop(h); h = self.c2(h); h = self.n2(h)
        return F.silu(h + self.skip(x))

class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=128, ch_mults=(1,2,2,2), num_res_blocks=2,
                 emb_dim=256, attn_resolutions=(8,16), num_heads=4, dropout=0.0, use_gn=True):
        super().__init__()
        self.out_ch = out_ch
        chs = [base_ch*m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        downs = []
        cur = chs[0]
        feats = []
        for i, ch in enumerate(chs):
            for _ in range(num_res_blocks):
                downs.append(ResBlock(cur, ch, emb_dim, dropout, use_gn)); cur=ch
                if (2**i) in attn_resolutions:
                    downs.append(SelfAttention2D(cur, num_heads, use_gn=use_gn))
            if i != len(chs)-1:
                downs.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1)); feats.append(cur)
        self.down = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(cur, cur, emb_dim, dropout, use_gn),
            SelfAttention2D(cur, num_heads, use_gn=use_gn) if (2**(len(chs)-1) in attn_resolutions) else nn.Identity(),
            ResBlock(cur, cur, emb_dim, dropout, use_gn),
        ])

        ups = []
        for i, ch in list(reversed(list(enumerate(chs)))):
            for _ in range(num_res_blocks+1):
                ups.append(ResBlock(cur, ch, emb_dim, dropout, use_gn)); cur=ch
                if (2**i) in attn_resolutions:
                    ups.append(SelfAttention2D(cur, num_heads, use_gn=use_gn))
            if i != 0:
                ups.append(nn.ConvTranspose2d(cur, cur, 4, stride=2, padding=1))
        self.up = nn.ModuleList(ups)

        self.out = nn.Sequential(GroupNorm(cur) if use_gn else nn.BatchNorm2d(cur), nn.SiLU(),
                                 nn.Conv2d(cur, out_ch, 3, padding=1))

    def forward(self, x, emb):
        f = []
        h = self.in_conv(x)
        for m in self.down:
            if isinstance(m, ResBlock): h = m(h, emb)
            else:
                h = m(h)
                if isinstance(m, nn.Conv2d) and m.stride==(2,2): f.append(h)
        for m in self.mid:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        for m in self.up:
            if isinstance(m, nn.ConvTranspose2d):
                h = m(h)
                if f:
                    s = f.pop()
                    if s.shape[-2:] != h.shape[-2:]:
                        s = F.interpolate(s, size=h.shape[-2:], mode='nearest')
                    h = torch.cat([h, s], dim=1)
            elif isinstance(m, ResBlock): h = m(h, emb)
            else: h = m(h)
        return self.out(h)
