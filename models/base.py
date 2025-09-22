from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import os

from .common import exists, default, sinusoidal_embedding, make_coord_grid, UNet2D

class DiffusionBase(L.LightningModule):
    """
    Shared UNet backbone, extras (coords/time), and end-of-epoch plotting/saving.
    Subclasses implement their training loss and sampling method.
    """

    def __init__(
        self,
        # channels
        cond_channels: int,
        target_channels: int,
        # extras
        extra_coord_channels: bool = False,
        extra_phys_time_scalar: Optional[float] = None,
        # UNet
        unet_base: int = 64,
        time_embed_dim: int = 256,
        # logging / saving
        sample_every_val: int = 1,
        sample_save_npz: bool = False,
        figures_cfg: Optional[dict] = None,
        # optimization config (Hydra)
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer_cfg", "scheduler_cfg", "figures_cfg"])
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg
        self.fig_cfg: Dict[str, Any] = figures_cfg or {}

        self.cond_channels = int(cond_channels)
        self.target_channels = int(target_channels)

        # extras
        self.add_coords = bool(extra_coord_channels)
        self.phys_time_scalar = extra_phys_time_scalar

        # UNet io
        in_ch = self.target_channels + self.cond_channels
        if self.add_coords: in_ch += 2
        if exists(self.phys_time_scalar): in_ch += 1

        self.unet = UNet2D(in_ch=in_ch, out_ch=self.target_channels, base=unet_base, emb_dim=time_embed_dim)

        # val cache: (x_cond, y_clean, optional meta)
        self._val_cache: List[Tuple[torch.Tensor, torch.Tensor, Optional[dict]]] = []

    # -------- figure/path helpers (metadata-aware) --------
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
        if not self._bool("use_meta_subdirs", True):
            return ""
        bits = []
        for key in ("date", "window", "member"):
            if key in meta:
                bits.append(f"{key}={self._safe_str(meta[key])}")
        return "/".join(bits)

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    # -------- extras & forward --------
    def add_extras(self, x_like: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_like.shape
        extras: List[torch.Tensor] = []
        if self.add_coords:
            grid = make_coord_grid(H, W, x_like.device)              # (2,H,W)
            extras.append(grid.unsqueeze(0).expand(B, -1, -1, -1))   # (B,2,H,W)
        if exists(self.phys_time_scalar):
            tchan = torch.full((B, 1, H, W), float(self.phys_time_scalar), device=x_like.device)
            extras.append(tchan)
        if not extras:
            return torch.zeros((B, 0, H, W), device=x_like.device, dtype=x_like.dtype)
        return torch.cat(extras, dim=1)

    def forward(self, y_noisy: torch.Tensor, x_cond: torch.Tensor, t_embed_scalar: torch.Tensor) -> torch.Tensor:
        """
        Subclasses define the semantic of t_embed_scalar:
          - EDM: log(σ)
          - DDPM: normalized t in [0,1]
        """
        time_embed_dim = self.unet.time_mlp[0].in_features
        t_emb = sinusoidal_embedding(t_embed_scalar, time_embed_dim)  # (B, time_embed_dim)
        extras = self.add_extras(y_noisy)                              # (B, Cextra, H, W)
        inp = torch.cat([y_noisy, x_cond, extras], dim=1)              # concat along channels
        return self.unet(inp, t_emb)

    # -------- training / validation cache wrapper --------
    def _cache_val_batch(self, x_cond, y_clean, meta):
        if len(self._val_cache) < 1:
            self._val_cache.append((x_cond.detach(), y_clean.detach(), meta))

  # -------- plotting/saving at val end (compact names + colorbars) --------
    def on_validation_epoch_end(self) -> None:
        import os, re
        import numpy as np
    
        # ——— helpers (local to keep this method self-contained) ———
        def _to_scalar(x):
            # best-effort: Tensor -> item, list/tuple -> first, else as-is
            try:
                if hasattr(x, "item"):
                    return x.item()
            except Exception:
                pass
            if isinstance(x, (list, tuple)) and x:
                return _to_scalar(x[0])
            return x
    
        def _safe(s: str) -> str:
            # compact + filesystem safe
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")
    
        def _compact_date(s):
            # turn "2023-02-18T00:00:00Z" -> "20230218T0000Z"
            if not isinstance(s, str): 
                return _safe(s)
            out = s.replace(":", "").replace("-", "")
            out = re.sub(r"(\d{8})T(\d{4})\d{2}Z", r"\1T\2Z", out)  # drop seconds if present
            return _safe(out)
    
        # ——— guards ———
        if (self.current_epoch + 1) % int(self.hparams.sample_every_val or 1) != 0:
            self._val_cache.clear(); return
        if not getattr(self.trainer, "is_global_zero", True):
            self._val_cache.clear(); return
        if not self._val_cache:
            return
    
        cached = self._val_cache[0]
        x_cond, y_gt = cached[0], cached[1]
        meta_in = cached[2] if len(cached) > 2 else None
        meta = self._select_first_meta(meta_in)
    
        # extract compact meta bits
        date_raw   = meta.get("date", "na")
        window_raw = meta.get("window", "na")
        member_raw = meta.get("member", meta.get("member_index", "na"))
    
        date_key   = _compact_date(date_raw)
        window_key = _safe(window_raw)
        m_val      = _to_scalar(member_raw)
        try:
            m_str = f"m{int(m_val):03d}"
        except Exception:
            m_str = f"m{_safe(m_val)}"
    
        # tensors to device
        x_cond = x_cond.to(self.device)
        y_gt   = y_gt.to(self.device)
        y_pred = self.sample_from_cond(x_cond, shape_target=y_gt.shape[1:4])  # subclass-defined
    
        # ——— plotting ———
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
                im = axes[i].imshow(t[0, i].numpy(), cmap="viridis")
                axes[i].axis("off")
                axes[i].set_title(f"{title} ch#{i}", fontsize=8)
                # add per-panel colorbar (compact)
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            fig.tight_layout()
            return fig
    
        # title suffix (compact)
        lt = meta.get("lead_times")
        lt_str = ""
        if isinstance(lt, (list, tuple)) and lt:
            lt_str = f" | leads={list(lt)}"
        title_suffix = f" ({date_key}, {window_key}, {m_str}){lt_str}"
    
        figs = [
            (_grid(x_cond, "cond endpoints" + title_suffix), "cond"),
            (_grid(y_pred, "pred internals" + title_suffix), "pred"),
            (_grid(y_gt,   "gt internals"   + title_suffix), "gt"),
        ]
    
        # ——— folders & compact filenames ———
        root = self._fig_root()
        local_dir = os.path.join(root, date_key, window_key, m_str)
        self._ensure_dir(local_dir)
    
        # e.g. 20230218T0000Z_0-6_m003__pred.png
        stem = f"{date_key}_{window_key}_{m_str}"
        png_cond = os.path.join(local_dir, f"{stem}__cond.png")
        png_pred = os.path.join(local_dir, f"{stem}__pred.png")
        png_gt   = os.path.join(local_dir, f"{stem}__gt.png")
        npz_path = os.path.join(local_dir, f"{stem}__e{self.current_epoch:04d}.npz")
    
        # save PNGs
        try:
            for fig, tag in figs:
                path = {"cond": png_cond, "pred": png_pred, "gt": png_gt}[tag]
                fig.savefig(path, dpi=120, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Saving figures failed: {e}")
    
        # optional NPZ
        if bool(self.hparams.sample_save_npz):
            try:
                np.savez_compressed(
                    npz_path,
                    cond=x_cond[0].detach().cpu().numpy(),
                    pred=y_pred[0].detach().cpu().numpy(),
                    gt=y_gt[0].detach().cpu().numpy(),
                    date=str(date_raw),
                    window=str(window_raw),
                    member=str(m_val),
                )
            except Exception as e:
                warnings.warn(f"Saving NPZ failed: {e}")
    
        self._val_cache.clear()


    # -------- optimizers (Hydra-friendly) --------
    def configure_optimizers(self):
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
