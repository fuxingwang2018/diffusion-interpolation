from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import os

from .common import exists, sinusoidal_embedding, make_coord_grid, UNet2D

class DiffusionBase(L.LightningModule):
    """
    Shared UNet backbone, extras (coords/time), and end-of-epoch plotting/saving.
    This version supports 4-D BCHW and 5-D BTCHW inputs.

    Expected dataloader shapes:
      - x_cond: [B, 2, C, H, W]  (two endpoint frames, each with C channels)
      - y     : [B, T, C, H, W]  (T target frames, each with C channels)

    What the network actually sees (internally):
      - x_cond_flat: [B, 2*C, H, W]
      - y_flat     : [B, T*C, H, W]

    Notes:
      • Set model hyperparams to flattened channel counts:
          cond_channels   = 2 * C
          target_channels = T * C
      • The forward() will reshape 5-D inputs to BCHW, run the UNet, and
        reshape outputs back to BTCHW for convenience.
      • The diffusion step embedding (t_embed_scalar) is still a (B,) vector
        and refers to the *denoiser* step, not physical time.
      • If extra_phys_time_scalar is provided, a single (B,1,H,W) map is
        concatenated — same value for all target times.
    """

    def __init__(
        self,
        # channels (must be flattened counts if your data is BTCHW):
        cond_channels: int,     # e.g., cond_channels = 2*C
        target_channels: int,   # e.g., target_channels = T*C
        # extras
        extra_coord_channels: bool = False,
        extra_phys_time_scalar: Optional[float] = None,
        # UNet
        unet_base: int = 64,
        time_embed_dim: int = 256,
        # attention passthrough
        use_attention: bool = False,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        attn_levels: Optional[Tuple[str, ...]] = ("mid",),
        # logging / saving
        sample_every_val: int = 1,
        sample_save_npz: bool = False,
        figures_cfg: Optional[dict] = None,
        # optimization
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,

        init_with_ones: bool = False,  # for testing
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer_cfg", "scheduler_cfg", "figures_cfg"])
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg
        self.fig_cfg: Dict[str, Any] = figures_cfg or {}

        # These are the *flattened* channel counts the UNet will actually see
        self.cond_channels   = int(cond_channels)     # expect 2*C
        self.target_channels = int(target_channels)   # expect T*C

        # extras
        self.add_coords = bool(extra_coord_channels)          # adds 2 channels if True
        self.phys_time_scalar = extra_phys_time_scalar        # adds 1 channel if given
        self.init_with_ones = init_with_ones

        # UNet io channels
        in_ch = self.target_channels + self.cond_channels
        if self.add_coords: in_ch += 2
        if exists(self.phys_time_scalar): in_ch += 1

        self.unet = UNet2D(
            in_ch=in_ch,
            out_ch=self.target_channels,
            base=unet_base,
            emb_dim=time_embed_dim,
            use_attention=use_attention,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_levels=tuple(attn_levels or ("mid",)),
        )

        # small runtime guard so we only print mismatch help once
        self._shape_checked_once = False

    # -------- small helpers --------
    @staticmethod
    def _flatten_time_to_channel(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """
        Accept BCHW or BTCHW. Return BCHW plus (T,C) info if we flattened.
        """
        if x.dim() == 4:
            # BCHW
            return x, None
        if x.dim() == 5:
            # BTCHW -> BCHW
            B, T, C, H, W = x.shape
            return x.reshape(B, T * C, H, W), (T, C)
        raise ValueError(f"Expected 4-D (BCHW) or 5-D (BTCHW), got shape {tuple(x.shape)}")

    @staticmethod
    def _unflatten_channel_to_time(y: torch.Tensor, tc: Optional[Tuple[int, int]]) -> torch.Tensor:
        """
        If tc=(T,C) is provided, convert BCHW back to BTCHW.
        """
        if tc is None:
            return y
        T, C = tc
        B, TC, H, W = y.shape
        if TC != T * C:
            raise RuntimeError(f"Cannot reshape output with C={TC} to (T={T}, C={C}).")
        return y.view(B, T, C, H, W)

    def _extras(self, ref_bchw: torch.Tensor) -> torch.Tensor:
        """
        Build extras aligned with a BCHW reference tensor.
        Returns (B, C_extra, H, W)
        """
        B, _, H, W = ref_bchw.shape
        extras: List[torch.Tensor] = []
        if self.add_coords:
            grid = make_coord_grid(H, W, ref_bchw.device)           # (2,H,W)
            extras.append(grid.unsqueeze(0).expand(B, -1, -1, -1))  # (B,2,H,W)
        if exists(self.phys_time_scalar):
            tchan = torch.full((B, 1, H, W), float(self.phys_time_scalar), device=ref_bchw.device)
            extras.append(tchan)
        if not extras:
            return torch.zeros((B, 0, H, W), device=ref_bchw.device, dtype=ref_bchw.dtype)
        return torch.cat(extras, dim=1)

    # -------- forward --------
    def forward(self, y_noisy: torch.Tensor, x_cond: torch.Tensor, t_embed_scalar: torch.Tensor) -> torch.Tensor:
        """
        Inputs (from dataloader / training loop):
          y_noisy: [B, T, C, H, W]  or [B, T*C, H, W]
          x_cond : [B, 2, C, H, W]  or [B, 2*C, H, W]
          t_embed_scalar: [B,]      diffusion step in [0,1] (for DDPM) or log-σ (for EDM)

        Internally we use BCHW for the UNet:
          y_noisy_bchw = reshape(y_noisy)  # [B, T*C, H, W]
          x_cond_bchw  = reshape(x_cond)   # [B, 2*C, H, W]
          inp = concat([y_noisy_bchw, x_cond_bchw, extras], dim=1)

        Output:
          pred with the same time layout as y_noisy:
            - If input was BTCHW -> returns BTCHW
            - If input was BCHW  -> returns BCHW
        """
        # 1) Time embedding
        time_embed_dim = self.unet.time_mlp[0].in_features
        t_emb = sinusoidal_embedding(t_embed_scalar, time_embed_dim)  # (B, time_embed_dim)

        # 2) Flatten time to channels if needed
        y_bchw, y_tc = self._flatten_time_to_channel(y_noisy)  # y_tc = (T,C) or None
        x_bchw, _    = self._flatten_time_to_channel(x_cond)   # endpoints -> [B, 2*C, H, W]

        # 3) Extras on BCHW
        extras = self._extras(y_bchw)                           # (B, Cextra, H, W)

        # 4) Concatenate
        inp = torch.cat([y_bchw, x_bchw, extras], dim=1)        # (B, T*C + 2*C + Cextra, H, W)

        # 5) One-time helpful check for channel configuration mismatches
        if not self._shape_checked_once:
            expected_in = self.unet.in_conv.in_channels
            if inp.shape[1] != expected_in:
                msg = (
                    f"[Shape mismatch] The UNet was built for in_ch={expected_in}, "
                    f"but current input has {inp.shape[1]} channels "
                    f"(y={y_bchw.shape[1]} + x={x_bchw.shape[1]} + extras={extras.shape[1]}).\n"
                    "Reminder: when your data is BTCHW, you must pass *flattened* counts when constructing the model:\n"
                    "  cond_channels   = 2 * C\n"
                    "  target_channels = T * C\n"
                    "Example: C=3, T=5 -> cond_channels=6, target_channels=15."
                )
                raise RuntimeError(msg)
            self._shape_checked_once = True

        # 6) UNet
        pred_bchw = self.unet(inp, t_emb)                       # (B, T*C, H, W)

        # 7) Restore time dimension if the input had it
        return self._unflatten_channel_to_time(pred_bchw, y_tc)

    # -------- training / validation cache wrapper --------
    def _cache_val_batch(self, x_cond, y_clean, meta):
        if not hasattr(self, "_val_cache"):
            self._val_cache = []  # lazy create
        if len(self._val_cache) < 1:
            self._val_cache.append((x_cond.detach(), y_clean.detach(), meta))

    def on_load_checkpoint(self, checkpoint) -> None:
        self.print("Model Restored from checkpoint")

    # -------- plotting/saving at val end (compact names + colorbars) --------
    def on_validation_epoch_end(self) -> None:
        import re, hashlib
        import numpy as np

        # guards
        if (self.current_epoch + 1) % int(self.hparams.sample_every_val or 1) != 0:
            if hasattr(self, "_val_cache"): self._val_cache.clear()
            return
        if not getattr(self.trainer, "is_global_zero", True):
            if hasattr(self, "_val_cache"): self._val_cache.clear()
            return
        if not getattr(self, "_val_cache", None):
            return

        cached = self._val_cache[0]
        x_cond, y_gt = cached[0], cached[1]
        meta_in = cached[2] if len(cached) > 2 else None
        meta = self._select_first_meta(meta_in)

        # --- helpers for safe text ---
        def _to_scalar(x):
            try:
                if hasattr(x, "item"):
                    return x.item()
            except Exception:
                pass
            if isinstance(x, (list, tuple)) and x:
                return _to_scalar(x[0])
            return x

        def _safe(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

        def _compact_date(s):
            if not isinstance(s, str):
                return _safe(s)
            out = s.replace(":", "").replace("-", "")
            out = re.sub(r"(\d{8})T(\d{4})\d{2}Z", r"\1T\2Z", out)
            return _safe(out)

        def _member_short(m):
            m = _to_scalar(m)
            try:
                return f"m{int(m):03d}"
            except Exception:
                text = _safe(m)
                if len(text) <= 12:
                    return f"m{text}"
                h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
                return f"m{h}"

        # --- compact meta keys / folders ---
        date_key   = _compact_date(meta.get("date", "na"))
        window_raw = meta.get("window", "na")
        if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
            window_key = f"{window_raw[0]}-{window_raw[1]}"
        else:
            window_key = _safe(window_raw)
        m_str = _member_short(meta.get("member", meta.get("member_index", "na")))

        # tensors to device
        x_cond = x_cond.to(self.device)
        y_gt   = y_gt.to(self.device)

        # Build a shape_target the sampler can understand.
        # Many samplers expect BCHW channel counts; if y_gt is BTCHW, pass T*C.
        if y_gt.dim() == 5:
            _, T, C, H, W = y_gt.shape
            shape_target_for_sampler = (T * C, H, W)   # flattened for samplers that expect BCHW
        else:
            _, C, H, W = y_gt.shape
            shape_target_for_sampler = (C, H, W)

        with torch.amp.autocast("cuda", enabled=False):
            y_pred = self.sample_from_cond(x_cond, shape_target=shape_target_for_sampler)  # subclass-defined

        # If sampler returned BCHW but GT is BTCHW, reshape prediction for parity.
        if y_gt.dim() == 5 and y_pred.dim() == 4:
            B, TC, H, W = y_pred.shape
            assert TC == T * C, f"Sampler returned {TC} channels but GT implies {T*C}."
            y_pred = y_pred.view(B, T, C, H, W)

        # plotting
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            warnings.warn(f"Matplotlib not available for saving figures: {e}")
            self._val_cache.clear()
            return

        def _as_bchw(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 5:  # BTCHW -> BCHW for visualization
                B, T, C, H, W = t.shape
                t = t.view(B, T * C, H, W)
            return t

        def _grid(t: torch.Tensor, title: str, max_ch=3):
            t = _as_bchw(t).detach().float().cpu()
            b, c, h, w = t.shape
            ch = min(c, max_ch)
            fig, axes = plt.subplots(1, ch, figsize=(ch * 2.6, 2.6))
            if ch == 1:
                axes = [axes]
            for i in range(ch):
                im = axes[i].imshow(t[0, i].numpy(), cmap="viridis")
                axes[i].axis("off")
                axes[i].set_title(f"{title} ch#{i}", fontsize=8)
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            fig.tight_layout()
            return fig

        lt = meta.get("lead_times")
        lt_str = f" | leads={list(lt)}" if isinstance(lt, (list, tuple)) and lt else ""
        title_suffix = f" ({date_key}, {window_key}, {m_str}){lt_str}"

        title_suffix = f" ({date_key}, {window_key}, {m_str}){lt_str}"


        figs = [
            (_grid(x_cond, "cond endpoints" + title_suffix), "cond"),
            (_grid(y_pred, "pred internals" + title_suffix), "pred"),
            (_grid(y_gt,   "gt internals"   + title_suffix), "gt"),
        ]

        # folders & filenames
        root = self._fig_root()
        local_dir = os.path.join(root, date_key, window_key, m_str)
        try:
            self._ensure_dir(local_dir)
        except OSError:
            local_dir = os.path.join(root, date_key, window_key)
            self._ensure_dir(local_dir)

        stem = f"{date_key}_{window_key}_{m_str}_e{self.current_epoch:04d}"
        png_cond = os.path.join(local_dir, f"{stem}_cond.png")
        png_pred = os.path.join(local_dir, f"{stem}_pred.png")
        png_gt   = os.path.join(local_dir, f"{stem}_gt.png")
        npz_path = os.path.join(local_dir, f"{stem}.npz")

        try:
            for fig, tag in figs:
                path = {"cond": png_cond, "pred": png_pred, "gt": png_gt}[tag]
                fig.savefig(path, dpi=120, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Saving figures failed: {e}")

        if bool(self.hparams.sample_save_npz):
            try:
                # Save in the *natural* shapes (BTCHW if present)
                np.savez_compressed(
                    npz_path,
                    cond=x_cond[0].detach().cpu().numpy(),
                    pred=y_pred[0].detach().cpu().numpy(),
                    gt=y_gt[0].detach().cpu().numpy(),
                    date=str(meta.get("date")),
                    window=str(window_raw),
                    member=str(_to_scalar(meta.get("member", meta.get("member_index", "na")))),
                )
            except Exception as e:
                warnings.warn(f"Saving NPZ failed: {e}")

        self._val_cache.clear()

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

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

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
