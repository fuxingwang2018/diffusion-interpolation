from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import warnings
import torch
from torch import nn
import lightning as L
import os

from .common import exists, sinusoidal_embedding, UNet2D

class DiffusionBase(L.LightningModule):
    """
    Shared UNet backbone and end-of-epoch plotting/saving.
    This version supports 4-D BCHW and 5-D BTCHW inputs.

    Expected dataloader shapes:
      - x_cond: [B, 2, C, H, W]  (two endpoint frames, each with C channels)
      - y     : [B, T, C, H, W]  (T target frames, each with C channels)

    What the network sees internally (time fused onto channels):
      - x_cond_flat: [B, 2*C, H, W]
      - y_flat     : [B, T*C, H, W]

    Notes
    -----
    • Construct the model with *flattened* channel counts:
        cond_channels   = 2 * C
        target_channels = T * C
    • The diffusion step embedding (t_embed_scalar) is a (B,) vector referring to
      the *denoiser* step (e.g., normalized t for DDPM), not the physical time.
    """

    def __init__(
        self,
        # flattened channels:
        cond_channels: int,     # expect 2*C
        target_channels: int,   # expect T*C
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

        # flattened channel counts (what UNet actually receives)
        self.cond_channels   = int(cond_channels)     # 2*C
        self.target_channels = int(target_channels)   # T*C

 
        # UNet I/O channels
        in_ch = self.target_channels + self.cond_channels
 
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

        self._shape_checked_once = False  # to print a helpful error only once

    # -------- shape helpers --------
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
        raise ValueError(f"Expected 4-D (BCHW) or 5-D (BTCHW), got {tuple(x.shape)}")

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


    # -------- forward --------
    def forward(self, y_noisy: torch.Tensor, x_cond: torch.Tensor, t_embed_scalar: torch.Tensor) -> torch.Tensor:
        """
        Inputs (from dataloader / training loop):
          y_noisy: [B, T, C, H, W]  or [B, T*C, H, W]
          x_cond : [B,  2, C, H, W] or [B, 2*C, H, W]
          t_embed_scalar: [B,]      diffusion step (denoiser-time), e.g. normalized t for DDPM

        Internally:
          y_bchw = reshape(y_noisy)  # [B, T*C, H, W]
          x_bchw = reshape(x_cond)   # [B, 2*C, H, W]
          inp = concat([y_bchw, x_bchw], dim=1)

        Output:
          pred with the same time layout as y_noisy:
            - If input was BTCHW -> returns BTCHW
            - If input was BCHW  -> returns BCHW
        """
        # 1) time/step embedding for the denoiser
        time_embed_dim = self.unet.time_mlp[0].in_features
        t_emb = sinusoidal_embedding(t_embed_scalar, time_embed_dim)  # (B, time_embed_dim)

        # 2) flatten time to channels if needed
        y_bchw, y_tc = self._flatten_time_to_channel(y_noisy)  # y_tc=(T,C) or None
        x_bchw, _    = self._flatten_time_to_channel(x_cond)   # [B, 2*C, H, W]

    
        # 4) concatenate
        inp = torch.cat([y_bchw, x_bchw], dim=1)        # (B, T*C + 2*C , H, W)

        # 5) one-time helpful config check
        if not self._shape_checked_once:
            expected_in = self.unet.in_conv.in_channels
            if inp.shape[1] != expected_in:
                msg = (
                    f"[Shape mismatch] UNet in_ch={expected_in}, but input has {inp.shape[1]} channels "
                    f"(y={y_bchw.shape[1]} + x={x_bchw.shape[1]}).\n"
                    "Reminder: if your data is BTCHW, pass *flattened* counts to the constructor:\n"
                    "  cond_channels   = 2 * C\n"
                    "  target_channels = T * C\n"
                    "Example: C=3, T=5 -> cond_channels=6, target_channels=15."
                )
                raise RuntimeError(msg)
            self._shape_checked_once = True

        # 6) UNet forward
        pred_bchw = self.unet(inp, t_emb)                       # (B, T*C, H, W)

        # 7) restore time dimension if needed
        return self._unflatten_channel_to_time(pred_bchw, y_tc)

    # -------- training / validation cache wrapper --------
    def _cache_val_batch(self, x_cond, y_clean, meta):
        if not hasattr(self, "_val_cache"):
            self._val_cache = []  # lazy create
        if len(self._val_cache) < 1:
            self._val_cache.append((x_cond.detach(), y_clean.detach(), meta))

    def on_load_checkpoint(self, checkpoint) -> None:
        self.print("Model Restored from checkpoint")

    # -------- plotting/saving at val end (C rows × (T+2) cols, minimal titles) --------
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
        x_cond = x_cond.to(self.device)   # [B,2,C,H,W] or [B,2*C,H,W]
        y_gt   = y_gt.to(self.device)     # [B,T,C,H,W] or [B,T*C,H,W]

        # Build a shape_target for the sampler (often expects BCHW counts).
        if y_gt.dim() == 5:
            _, T, C, H, W = y_gt.shape
            shape_target_for_sampler = (T * C, H, W)
        else:
            _, C, H, W = y_gt.shape
            T = None  # unknown here
            shape_target_for_sampler = (C, H, W)

        # Generate prediction with the subclass sampler
        with torch.amp.autocast("cuda", enabled=False):
            y_pred = self.sample_from_cond(x_cond, shape_target=shape_target_for_sampler)  # subclass-defined

        # Make shapes consistent for plotting: ensure BTCHW for y_gt and y_pred
        def _ensure_btchw(t: torch.Tensor, fallback_TC: Optional[int] = None) -> torch.Tensor:
            if t.dim() == 5:
                return t
            # BCHW -> BTCHW by splitting channels to (T,C) using y_gt guide
            B, TC, H, W = t.shape
            if y_gt.dim() == 5:
                _, Ty, Cy, _, _ = y_gt.shape
                assert TC == Ty * Cy, f"Expected {Ty*Cy} channels, got {TC}."
                return t.view(B, Ty, Cy, H, W)
            # fallback not supported without (T,C)
            if fallback_TC is not None:
                raise RuntimeError("Cannot infer (T,C) for BCHW tensor; provide BTCHW ground truth.")
            return t

        y_gt_bt = _ensure_btchw(y_gt)
        y_pred_bt = _ensure_btchw(y_pred)

        # plotting
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            warnings.warn(f"Matplotlib not available for saving figures: {e}")
            self._val_cache.clear()
            return

        # ---- grid builder: rows=channels (C), cols=T+2 (start, internals, end)
        def _grid_btchw(x_cond_bt: torch.Tensor, y_bt: torch.Tensor, meta: dict, suptitle: str):
            """
            x_cond_bt: [B, 2, C, H, W]   (start,end)
            y_bt     : [B, T, C, H, W]   (internals)
            Titles per subplot: 'ch=<c> lt=<lead_time>'
            """
            assert x_cond_bt.dim() == 5 and x_cond_bt.shape[1] == 2, "x_cond must be [B,2,C,H,W]"
            assert y_bt.dim() == 5 and x_cond_bt.shape[2] == y_bt.shape[2], "channel count mismatch"

            B, Tloc, C, H, W = y_bt.shape
            # lead times per column: [start, internals..., end]
            lts = meta.get("lead_times", None)
            if isinstance(lts, (list, tuple)) and len(lts) == Tloc + 2:
                col_lts = list(lts)
            else:
                col_lts = list(range(Tloc + 2))  # fallback 0..T+1

            # first item for display
            x0 = x_cond_bt[0].detach().float().cpu()  # [2,C,H,W]
            y0 = y_bt[0].detach().float().cpu()       # [T,C,H,W]

            cols = Tloc + 2
            rows = C
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
            if rows == 1:
                axes = [axes]
            if cols == 1:
                axes = [[ax] for ax in axes]

            for cidx in range(C):
                start_img = x0[0, cidx].numpy()
                end_img   = x0[1, cidx].numpy()
                internals = y0[:, cidx].numpy()  # [T,H,W]

                # consistent color scale per channel

                # col 0: start
                im0 = axes[cidx][0].imshow(start_img, cmap="viridis", vmin=-1, vmax=1) # set vmin and vmax here 
                axes[cidx][0].axis("off")
                axes[cidx][0].set_title(f"ch={cidx} lt={col_lts[0]}", fontsize=8)

                # cols 1..T: internals
                for k in range(Tloc):
                    vmin = float(internals[k].min())
                    vmax = float(internals[k].max())
                    if vmin == vmax:
                        vmin, vmax = float(vmin - 1e-6), float(vmax + 1e-6)

                    ax = axes[cidx][1 + k]
                    ax.imshow(internals[k], cmap="viridis", vmin=-1, vmax=1)
                    ax.axis("off")
                    ax.set_title(f"ch={cidx} lt={col_lts[1 + k]} {vmin:.2f}-{vmax:.2f}", fontsize=8)

                # col T+1: end
                axes[cidx][Tloc + 1].imshow(end_img, cmap="viridis", vmin=-1, vmax=1)
                axes[cidx][Tloc + 1].axis("off")
                axes[cidx][Tloc + 1].set_title(f"ch={cidx} lt={col_lts[Tloc + 1]}", fontsize=8)

                # one colorbar per row
                fig.colorbar(im0, ax=axes[cidx], orientation="vertical", fraction=0.02, pad=0.01)

            # minimalist top title (optional)
            if suptitle:
                fig.suptitle(suptitle, fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            return fig

        # prepare inputs for grid: ensure x_cond is [B,2,C,H,W]
        if x_cond.dim() == 4:  # [B,2*C,H,W] -> [B,2,C,H,W]
            B, Cxc, Hc, Wc = x_cond.shape
            # infer C from y_gt_bt
            _, Tloc, Cinf, _, _ = y_gt_bt.shape
            assert Cxc == 2 * Cinf, f"Cannot split x_cond channels ({Cxc}) into (2,C={Cinf})."
            x_cond_bt = x_cond.view(B, 2, Cinf, Hc, Wc)
        else:
            x_cond_bt = x_cond

        # Draw figures
        try:
            cond_fig = _grid_btchw(x_cond_bt, y_gt_bt, meta, "cond endpoints")
            pred_fig = _grid_btchw(x_cond_bt, y_pred_bt, meta, "pred internals")
            gt_fig   = _grid_btchw(x_cond_bt, y_gt_bt, meta, "gt internals")
            figs = [(cond_fig, "cond"), (pred_fig, "pred"), (gt_fig, "gt")]
        except Exception as e:
            warnings.warn(f"Plotting failed: {e}")
            self._val_cache.clear()
            return

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
                # Save in natural shapes (BTCHW)
                np.savez_compressed(
                    npz_path,
                    cond=x_cond_bt[0].detach().cpu().numpy(),
                    pred=y_pred_bt[0].detach().cpu().numpy(),
                    gt=y_gt_bt[0].detach().cpu().numpy(),
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
