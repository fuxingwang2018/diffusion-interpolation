from __future__ import annotations
from typing import Optional, Tuple, List, Any, Dict

import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L


# -------------------------
# small helpers
# -------------------------

def exists(x): return x is not None

def default(val, dflt):
    return val if exists(val) else (dflt() if callable(dflt) else dflt)

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal time embedding (for noise level σ or any continuous scalar).
    t: (B,) tensor
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / max(half - 1, 1)))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        import torch.nn.functional as F  # local import to avoid polluting namespace
        emb = F.pad(emb, (0, 1))
    return emb

 

# -------------------------
# very compact U-Net with FiLM from time/noise embedding
# -------------------------

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, c_out * 2)  # gamma, beta for FiLM
        )
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
    in_ch: channels of [noisy_target || conditioning]
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
        self.down1 = Down(base)           # /2
        self.rb2 = ResBlock(base, base*2, emb_dim)
        self.down2 = Down(base*2)         # /4
        self.rb3 = ResBlock(base*2, base*4, emb_dim)

        # Dec
        self.up2 = Up(base*4, base*2)
        self.rb4 = ResBlock(base*4, base*2, emb_dim)
        self.up1 = Up(base*2, base)
        self.rb5 = ResBlock(base*2, base, emb_dim)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t_emb)  # (B, emb_dim)
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


# -------------------------
# EDM-style Lightning module for interpolation
# -------------------------

class EDMInterpolator(L.LightningModule):
    """
    Conditional diffusion (EDM-style) that learns to map:
      endpoints (first+last)  --->  internal lead-times

    Batch from your DataModule:
      x_cond  = endpoints channels  (B, Cx, H, W)
      y_clean = internal channels   (B, Cy, H, W)

    We add noise to y_clean only, then predict noise ε on y.
    """

    def __init__(
        self,
        # channels
        cond_channels: int,
        target_channels: int,

        # U-Net
        unet_base: int = 64,
        time_embed_dim: int = 256,

        # EDM noise / training
        sigma_data: float = 0.5,      # (kept for completeness; not used directly here)
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,         # log-normal sampling of sigma
        p_std: float = 1.2,
        loss_weighting: str = "edm",  # 'edm' (sigma^2) or 'none'

        # optimization
        lr: float = 3e-4,             # fallback LR if no optimizer_cfg is given
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,

        # sampling
        sample_steps: int = 20,       # number of EDM steps for inference
        rho: float = 7.0,             # EDM schedule exponent
        sample_every_val: int = 1,    # generate+save samples every N val epochs
        sample_save_npz: bool = False,

        # figure saving (from Hydra config)
        figures_cfg: Optional[dict] = None,
        # Back-compat alias (ignored if figures_cfg is provided):
        figures: Optional[dict] = None,
    ):
        super().__init__()
        # keep optimizer/scheduler configs out of saved hparams payload
        self.save_hyperparameters(ignore=["optimizer_cfg", "scheduler_cfg", "figures_cfg", "figures"])
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg
        self.fig_cfg: Dict[str, Any] = figures_cfg or figures or {}

        self.cond_channels = int(cond_channels)
        self.target_channels = int(target_channels)


        # build UNet
        in_ch = self.target_channels + self.cond_channels




        self.unet = UNet2D(in_ch=in_ch, out_ch=self.target_channels,
                           base=unet_base, emb_dim=time_embed_dim)

        # cache for val sampling
        self._val_cache: List[Tuple[torch.Tensor, torch.Tensor, Optional[dict]]] = []

    # -------- figure/path helpers (UPDATED for new meta keys) --------
    def _fig_root(self) -> str:
        return str((self.fig_cfg.get("root") if isinstance(self.fig_cfg, dict) else None) or "figures")

    def _bool(self, key: str, default: bool) -> bool:
        try:
            return bool(self.fig_cfg.get(key, default))
        except Exception:
            return default

    def _safe_str(self, x) -> str:
        import re
        s = str(x)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:128] or "na"

    def _select_first_meta(self, meta_in):
        """
        Robustly get one metadata dict from:
         - dict (possibly dict-of-lists, as in default collate)
         - list[dict]
         - None
        """
        if meta_in is None:
            return {}
        if isinstance(meta_in, dict):
            out = {}
            for k, v in meta_in.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    out[k] = v[0]
                else:
                    out[k] = v
            return out
        if isinstance(meta_in, (list, tuple)) and meta_in and isinstance(meta_in[0], dict):
            return meta_in[0]
        return {}

    def _build_stem_from_meta(self, meta: dict) -> str:
        """
        Build filename stem from the *new* meta keys:
        - date, window
        - member (or member_index)
        - lead_times (list)
        - rec_index
        """
        # Optional template override
        tmpl = self.fig_cfg.get("filename_template") if isinstance(self.fig_cfg, dict) else None
        if tmpl:
            class _D(dict):
                def __missing__(self, k): return "na"
            try:
                return self._safe_str(tmpl.format_map(_D(meta)))
            except Exception:
                pass

        parts = []
        if "date" in meta:          parts.append(f"date={self._safe_str(meta['date'])}")
        if "window" in meta:        parts.append(f"window={self._safe_str(meta['window'])}")
        if "member" in meta:        parts.append(f"member={self._safe_str(meta['member'])}")
        elif "member_index" in meta:parts.append(f"midx={self._safe_str(meta['member_index'])}")
        if "lead_times" in meta:
            try:
                lt = meta["lead_times"]
                if isinstance(lt, (list, tuple)) and lt:
                    parts.append("leads=" + "-".join(self._safe_str(x) for x in lt))
            except Exception:
                pass
        if "rec_index" in meta:     parts.append(f"rec={self._safe_str(meta['rec_index'])}")
        return "_".join(parts) or "sample"

    def _meta_subdir(self, meta: dict) -> str:
        """
        Optional hierarchical subdir: date/window/member (new meta keys).
        """
        if not self._bool("use_meta_subdirs", True):
            return ""
        bits = []
        for key in ("date", "window", "member"):
            if key in meta:
                bits.append(f"{key}={self._safe_str(meta[key])}")
        return "/".join(bits)

    def _ensure_dir(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)

    # ------------- EDM core utilities -------------

    @torch.no_grad()
    def get_sigma_schedule(self, steps: int, rho: float) -> torch.Tensor:
        """Karras scheduler (EDM). Returns (steps,) sigmas decreasing from sigma_max -> sigma_min."""
        s0, s1 = self.hparams.sigma_max, self.hparams.sigma_min
        ramp = torch.linspace(0, 1, steps, device=self.device)
        sigmas = (s0 ** (1 / rho) + ramp * (s1 ** (1 / rho) - s0 ** (1 / rho))) ** rho
        return torch.flip(sigmas, dims=[0])  # descending

    def sample_sigmas_train(self, b: int) -> torch.Tensor:
        """Draw σ ~ LogNormal(p_mean, p_std), clamp to [sigma_min, sigma_max]."""
        p_mean, p_std = self.hparams.p_mean, self.hparams.p_std
        sigma = torch.exp(torch.randn(b, device=self.device) * p_std + p_mean)
        sigma = sigma.clamp(self.hparams.sigma_min, self.hparams.sigma_max)
        return sigma


    # ------------- forward = one denoising pass -------------
    def forward(self, y_noisy: torch.Tensor, x_cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        y_noisy: (B, Cy, H, W), x_cond: (B, Cx, H, W), sigma: (B,)
        returns: eps_hat (B, Cy, H, W)
        """
        time_embed_dim = self.unet.time_mlp[0].in_features
        t_emb = sinusoidal_embedding(sigma.log(), time_embed_dim)  # (B, time_embed_dim)
        inp = torch.cat([y_noisy, x_cond], dim=1)          # concat along channels
        return self.unet(inp, t_emb)

    # ------------- training / validation -------------
    def _shared_step(self, batch: Any, stage: str):
        """
        batch = (x_cond, y_clean[, meta])  # meta is optional
        """
        meta = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x_cond, y_clean, meta = batch
        else:
            x_cond, y_clean = batch

        B = y_clean.shape[0]
        sigma = self.sample_sigmas_train(B)
        noise = torch.randn_like(y_clean)
        y_noisy = y_clean + noise * sigma[:, None, None, None]

        eps_hat = self(y_noisy, x_cond, sigma)

        if self.hparams.loss_weighting == "edm":
            w = (sigma ** 2)[:, None, None, None]
            loss = F.mse_loss(eps_hat * w.sqrt(), noise * w.sqrt(), reduction="mean")
        else:
            loss = F.mse_loss(eps_hat, noise, reduction="mean")

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)

        if stage == "val" and len(self._val_cache) < 1:
            # keep meta (could be dict, list[dict], or dict-of-lists after collate)
            self._val_cache.append((x_cond.detach(), y_clean.detach(), meta))
        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self._shared_step(batch, "val")

    # ------------- sampling (EDM steps, simple Euler) -------------
    @torch.no_grad()
    def sample_from_cond(self, x_cond: torch.Tensor, shape_target: Tuple[int, int, int], steps: Optional[int] = None) -> torch.Tensor:
        """
        Given endpoints x_cond (B, Cx, H, W), generate internals (B, Cy, H, W)
        using an EDM sigma schedule and Euler updates:
            x0_hat = y - sigma * eps_hat
            y <- x0_hat + sigma_next * eps_hat
        """
        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]
        steps = default(steps, self.hparams.sample_steps)

        sigmas = self.get_sigma_schedule(steps=steps, rho=self.hparams.rho)  # (S,)
        y = torch.randn((B, Cy, H, W), device=x_cond.device) * sigmas[0]     # init at highest sigma

        for i, sigma in enumerate(sigmas):
            sigma_b = torch.full((B,), float(sigma), device=x_cond.device)
            eps = self(y, x_cond, sigma_b)
            x0_hat = y - sigma * eps
            if i == len(sigmas) - 1:
                y = x0_hat
            else:
                y = x0_hat + sigmas[i + 1] * eps
        return y

    # ------------- end-of-epoch saving (UPDATED to use new meta) -------------

    def on_validation_epoch_end(self) -> None:
        # frequency / rank guard
        if (self.current_epoch + 1) % int(self.hparams.sample_every_val or 1) != 0:
            self._val_cache.clear()
            return
        if not self.trainer.is_global_zero:
            self._val_cache.clear()
            return
        if not self._val_cache:
            return

        cached = self._val_cache[0]
        if len(cached) == 3:
            x_cond, y_gt, meta_in = cached
        else:
            x_cond, y_gt = cached[:2]
            meta_in = None

        # normalize to a single dict with keys like:
        # date, window, lead_times, members | member/member_index, start_valid_time, end_valid_time, rec_index
        meta = self._select_first_meta(meta_in)

        x_cond = x_cond.to(self.device)
        y_gt = y_gt.to(self.device)
        y_pred = self.sample_from_cond(x_cond, shape_target=y_gt.shape[1:4])

        # ---- plotting ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            warnings.warn(f"Matplotlib not available for saving figures: {e}")
            self._val_cache.clear()
            return

        def _grid(t: torch.Tensor, title: str, max_ch=3):
            t = t.detach().float().cpu()
            b, c, h, w = t.shape
            ch = min(c, max_ch)
            fig, axes = plt.subplots(1, ch, figsize=(ch * 2.6, 2.6))
            if ch == 1:
                axes = [axes]
            for i in range(ch):
                axes[i].imshow(t[0, i].numpy(), cmap="viridis")
                axes[i].axis("off")
                axes[i].set_title(f"{title} ch#{i}", fontsize=8)
            fig.tight_layout()
            return fig

        # build naming & directories from new meta + config
        import os
        root = self._fig_root()
        stem = self._build_stem_from_meta(meta)
        subdir = self._meta_subdir(meta)
        local_dir = os.path.join(root, subdir) if subdir else root
        self._ensure_dir(local_dir)

        # final local paths (filenames reflect new meta fields)
        png_cond = os.path.join(local_dir, f"{stem}__cond.png")
        png_pred = os.path.join(local_dir, f"{stem}__pred.png")
        png_gt   = os.path.join(local_dir, f"{stem}__gt.png")
        npz_path = os.path.join(local_dir, f"{stem}__sample_e{self.current_epoch:04d}.npz")

        # titles augmented with light meta (new keys)
        title_bits = []
        for k in ("date", "window", "member", "member_index"):
            if k in meta:
                title_bits.append(f"{k}={meta[k]}")
        if "lead_times" in meta and isinstance(meta["lead_times"], (list, tuple)) and len(meta["lead_times"]) > 0:
            title_bits.append(f"leads={meta['lead_times']}")
        title_suffix = (" (" + ", ".join(map(str, title_bits)) + ")") if title_bits else ""

        figs = [
            (png_cond, _grid(x_cond, "cond endpoints" + title_suffix)),
            (png_pred, _grid(y_pred, "pred internals" + title_suffix)),
            (png_gt,   _grid(y_gt,   "gt internals"   + title_suffix)),
        ]

        # save PNGs
        try:
            for path, fig in figs:
                fig.savefig(path, dpi=120, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Saving figures failed: {e}")

        # save NPZ (optional)
        if bool(self.hparams.sample_save_npz):
            try:
                import numpy as np
                np.savez_compressed(
                    npz_path,
                    cond=x_cond[0].detach().cpu().numpy(),
                    pred=y_pred[0].detach().cpu().numpy(),
                    gt=y_gt[0].detach().cpu().numpy(),
                    # tiny meta snapshot for traceability
                    date=str(meta.get("date")),
                    window=str(meta.get("window")),
                    member=str(meta.get("member", meta.get("member_index", ""))),
                    rec_index=int(meta.get("rec_index")) if "rec_index" in meta else -1,
                )
            except Exception as e:
                warnings.warn(f"Saving NPZ failed: {e}")

        self._val_cache.clear()

    # ------------- optimizers -------------
    def configure_optimizers(self):
        """Hydra-friendly optimizer creation with a safe fallback."""
        from hydra.utils import instantiate
        if isinstance(self.optimizer_cfg, dict) and "_target_" in self.optimizer_cfg:
            opt = instantiate(self.optimizer_cfg, params=self.parameters())
        else:
            raise ValueError("no hydra optimizer target")

        if self.scheduler_cfg:
            try:
                sch = instantiate(self.scheduler_cfg, optimizer=opt)
                return {"optimizer": opt, "lr_scheduler": sch}
            except Exception:
                return opt
        return opt
