#!/usr/bin/env python3
"""
sample.py — Generate samples from validation or test split.

- Uses Hydra config (same as train.py) via `def main(cfg: DictConfig)`.
- Instantiates datamodule + model.
- Optionally loads a Lightning checkpoint (path).
- Runs inference on N batches from val/test.
- Saves 3 figures per batch:
    * <stem>__cond.png  (start/end endpoints)
    * <stem>__gt.png    (ground-truth internals)
    * <stem>__pred.png  (predicted internals)
Grid layout (per figure):
    rows   = C (channels)
    cols   = T+2 (first, internals..., last)
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra import main as hydra_main
from hydra.utils import instantiate
import numpy as np 

# ---------------- small helpers ----------------

def _ensure_btchw(y: torch.Tensor, guide_y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Ensure (B,T,C,H,W). If y is (B, T*C, H, W), use guide_y (BTCHW) to split."""
    if y.dim() == 5:
        return y
    if guide_y is not None and guide_y.dim() == 5:
        B, T, C, H, W = guide_y.shape
        BY, TC, HY, WY = y.shape
        if (B, H, W) != (BY, HY, WY) or TC != T * C:
            raise RuntimeError(f"Shape mismatch for reshape: y={tuple(y.shape)} guide_y={tuple(guide_y.shape)}")
        return y.view(B, T, C, H, W)
    raise RuntimeError("Cannot infer (T,C) for BCHW tensor — provide guide_y (BTCHW).")


def _cond_to_btchw_from_y(cond_bchw: torch.Tensor, y_btchw: torch.Tensor) -> torch.Tensor:
    """(B,2*C,H,W) -> (B,2,C,H,W) using C from y_btchw (BTCHW)."""
    if cond_bchw.dim() != 4 or y_btchw.dim() != 5:
        raise RuntimeError(f"Unexpected shapes: cond={tuple(cond_bchw.shape)} y={tuple(y_btchw.shape)}")
    B, C2, H, W = cond_bchw.shape
    _, T, C, HY, WY = y_btchw.shape
    if (H, W) != (HY, WY) or C2 != 2 * C:
        raise RuntimeError(f"cond/y channel mismatch: cond={C2}, y C={C}")
    return cond_bchw.view(B, 2, C, H, W)

def radial_psd_1d(img_2d: torch.Tensor, H: int, W: int, nbins: int = 256 , eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    img_2d: (H, W) tensor (float, on CPU)
    Returns:
        k:    (nbins,) radial frequency (0..Nyquist, cycles/pixel)
        ps1d: (nbins,) radial power spectrum (mean |F|^2 within each annulus)
    """
    x = img_2d.numpy()
    # 2D FFT and power
    F = np.fft.fft2(x)
    P = (F.real**2 + F.imag**2)

    # Build frequency radii in cycles/pixel (0 at DC, up to ~0.5 Nyquist)
    fy = np.fft.fftfreq(H)  # [-0.5,0.5) scaled to cycles/pixel
    fx = np.fft.fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R = np.sqrt(FX**2 + FY**2)  # radial frequency

    # Bin edges from 0 .. max radius (Nyquist)
    r_max = 0.5 * np.sqrt(2.0)  # diagonal Nyquist; we’ll clamp at 0.5 to show up to Nyquist
    # Better: cap to 0.5 (max along axes)
    r_cap = 0.5
    r = np.clip(R, 0.0, r_cap)

    # Choose bins uniformly in [0, r_cap]
    nb = min(nbins, max(8, min(H, W) // 2))
    edges = np.linspace(0.0, r_cap, nb + 1)
    idx = np.digitize(r.ravel(), edges) - 1  # 0..nb-1
    idx = np.clip(idx, 0, nb - 1)

    # Accumulate power and counts per radial bin
    num = np.bincount(idx, weights=P.ravel(), minlength=nb)
    den = np.bincount(idx, minlength=nb)
    den = np.maximum(den, 1)  # avoid div by zero
    ps1d = num / den

    # Bin centers
    k = 0.5 * (edges[:-1] + edges[1:])
    return k, ps1d + eps  # add eps for log stability

def _plot_per_channel_diff(
    gt_bt: torch.Tensor,     # (B,T,C,H,W)
    pred_bt: torch.Tensor,   # (B,T,C,H,W)
    outdir: str,
    stem: str,
    title_prefix: str = "",
    cmap: str = "coolwarm",
    vmax=None               # optional fixed |diff| max for row 1
):
    """
    For each channel c, save a figure with 2 rows and T columns:

      Row 0: diff image = pred - gt               (diverging colormap, symmetric around 0)
      Row 1: 1D radial power spectrum of diff     (semilogy plot of radial average of |FFT2(diff)|^2)

    Output files: <outdir>/<stem>__diff_ch{c:02d}.png
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmax is not None:
        assert vmax > 0, "vmax must be positive when provided"

    ygt  = gt_bt[0].detach().float().cpu()    # (T,C,H,W)
    ypr  = pred_bt[0].detach().float().cpu()  # (T,C,H,W)
    diff = ypr - ygt                           # (T,C,H,W)

    T, C, H, W = diff.shape
    eps = 1e-12

    # ---- radial (isotropic) 1D power spectrum helper ----

    for c in range(C):
        d = diff[:, c]  # (T,H,W)

        # symmetric color scale for diff row
        if vmax is not None:
            amax = float(vmax)
        else:
            amax = float(d.abs().max())
            if amax == 0:
                amax = 1e-6
        vmin, vmax_sym = -amax, amax

        vvmax = float(d.max())
        vvmin = float(d.min())

        fig, axes = plt.subplots(2, T, figsize=(T * 2.1, 2 * 2.2))
        if T == 1:
            axes = [[axes[0]], [axes[1]]]

        # Row 0: diff images + RMSE in titles
        for k in range(T):
            rmse = float(torch.sqrt(torch.mean(d[k] ** 2)))
            im = axes[0][k].imshow(d[k], cmap=cmap, vmin=vmin, vmax=vmax_sym)
            axes[0][k].axis("off")
            axes[0][k].set_title(f"t={k+1}  RMSE={rmse:.3f}", fontsize=8)

        # Row 1: 1D power spectra (semilogy)
        # Keep shared y-limits across columns for comparability
        ymins, ymaxs = [], []
        spectra = []
        freqs = None
        for k in range(T):
            fk, pk = radial_psd_1d(d[k].cpu(), H, W, eps=eps)
            spectra.append((fk, pk))
            freqs = fk if freqs is None else freqs
            ymins.append(pk.min())
            ymaxs.append(pk.max())

        ylo = max(min(ymins), eps)
        yhi = max(ymaxs)

        for k in range(T):
            fk, pk = spectra[k]
            axes[1][k].semilogy(fk, pk)
            axes[1][k].grid(True, alpha=0.3, linewidth=0.5)
            axes[1][k].set_ylim([ylo, yhi * 1.05])
            # x up to Nyquist (0.5 cycles/pixel)
            axes[1][k].set_xlim([0.0, 0.5])
            if k == 0:
                axes[1][k].set_ylabel("Power (log)", fontsize=8)
            axes[1][k].set_xlabel("freq (cycles/pixel)", fontsize=8)

        if title_prefix:
            fig.suptitle(f"{title_prefix} | diff (pred−gt) [{vvmin:.2f}, {vvmax:.2f}] ch={c}", fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(outdir, f"{stem}__diff_ch{c:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

def _plot_per_channel_two_rows(
    cond_bt: torch.Tensor,   # (B,2,C,H,W)
    gt_bt: torch.Tensor,     # (B,T,C,H,W)
    pred_bt: torch.Tensor,   # (B,T,C,H,W)
    outdir: str,
    stem: str,
    title_prefix: str = "",
    cmap = "viridis",
    vmin = None,
    vmax = None
):
    """
    For each channel c: make a figure with 2 rows, T+2 columns.
      Row 0 (GT):   start, gt[0..T-1], end
      Row 1 (Pred): start, pred[0..T-1], end
    Color scale is per-channel, shared across both rows.
    Saves: <outdir>/<stem>__ch{c:02d}.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    isvmin = vmin is not None
    isvmax = vmax is not None
    # Take first item of the batch for visualization
    x0 = cond_bt[0].detach().float().cpu()   # (2,C,H,W)
    ygt = gt_bt[0].detach().float().cpu()    # (T,C,H,W)
    ypr = pred_bt[0].detach().float().cpu()  # (T,C,H,W)

    T, C, H, W = ygt.shape
    cols = T + 2

    for c in range(C):
        # Compute per-channel vmin/vmax across start/end, gt internals, pred internals
        stack = torch.cat(
            [x0[0, c][None], ygt[:, c], ypr[:, c], x0[1, c][None]], dim=0
        )  # ((T+2)+T, H, W) = (2T+2, H, W)
        vvmax = float(stack.max())
        vvmin = float(stack.min())
        vmin  = vvmin if not isvmin else vmin
        vmax = vvmax if not isvmax else vmax
        if vmin == vmax:  # avoid degenerate color range
           vmin, vmax = vmin - 1e-6, vmax + 1e-6

        fig, axes = plt.subplots(2, cols, figsize=(cols * 2.1, 2 * 2.1))
        if cols == 1:
            axes = [[axes[0]], [axes[1]]]

        # --- Row 0: Ground truth ---
        im0 = axes[0][0].imshow(x0[0, c], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0][0].axis("off"); axes[0][0].set_title(f"ch={c} start", fontsize=8)
        for k in range(T):
            axes[0][1 + k].imshow(ygt[k, c], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0][1 + k].axis("off"); axes[0][1 + k].set_title(f"gt t={k+1}", fontsize=8)
        axes[0][T + 1].imshow(x0[1, c], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0][T + 1].axis("off"); axes[0][T + 1].set_title("end", fontsize=8)

        # --- Row 1: Prediction ---
        axes[1][0].imshow(x0[0, c], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1][0].axis("off"); axes[1][0].set_title("start", fontsize=8)
        for k in range(T):
            axes[1][1 + k].imshow(ypr[k, c], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1][1 + k].axis("off"); axes[1][1 + k].set_title(f"pred t={k+1}", fontsize=8)
        axes[1][T + 1].imshow(x0[1, c], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1][T + 1].axis("off"); axes[1][T + 1].set_title("end", fontsize=8)

        # One colorbar for the whole figure (right side)
        #fig.colorbar(im0, ax=axes, orientation="vertical", fraction=0.025, pad=0.01)

        if title_prefix:
            fig.suptitle(f"{title_prefix} | [{vvmin:0.2f} {vvmax:0.2f}] ch={c}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
       
        out_path = os.path.join(outdir, f"{stem}__ch{c:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)



# ---------------- Hydra entry ----------------

@hydra_main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Optional runtime flags (split/ckpt/batches/device/outdir)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--device", default="cuda:0")
    args, _ = ap.parse_known_args()

    L.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule & model from Hydra cfg
    dm = instantiate(cfg.datamodule, _recursive_=False)
    model = instantiate(cfg.model, _recursive_=False, _convert_="partial")

    # Optionally load checkpoint
    if cfg.ckpt_file:
        ckpt_path = Path(cfg.ckpt_file)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded ckpt: {ckpt_path}\n  missing={missing}\n  unexpected={unexpected}", flush=True)
        else:
            print(f"[warn] ckpt not found: {ckpt_path}", flush=True)

    # Prepare dataloaders
    dm.prepare_data()
    dm.setup(stage="test" if cfg.split == "test" else "validate")
    loader = dm.test_dataloader() if cfg.split == "test" and hasattr(dm, "test_dataloader") else dm.val_dataloader()

    device = torch.device(cfg.device)
    model.eval().to(device)

    # Output dir
    outdir = cfg.outdir 
    os.makedirs(outdir, exist_ok=True)

    num_done = 0
    print(cfg.batches)
    with torch.no_grad():
        
        for bidx, batch in enumerate(loader):
            print(bidx)
            # Unpack
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Pack to (cond, target)
            
       
            
            cond, target = model._pack_xy(x, y)
      
            # Predict (DiffusionBase.sample uses sampler; SimpleModel does direct forward)
            pred = model.sample(cond, target.shape)
             
            
           

            # Shapes for plotting
            target_bt = _ensure_btchw(target, guide_y=y)
            pred_bt   = _ensure_btchw(pred, guide_y=y)
            cond_bt   = _ensure_btchw(cond, guide_y=x)
      
            # Build and save figures — one per channel
            stem = f"{cfg.split}_b{bidx:04d}"
            _plot_per_channel_two_rows(
                cond_bt=cond_bt,
                gt_bt=target_bt,
                pred_bt=pred_bt,
                outdir=outdir,
                stem=stem,
                title_prefix=f"[{cfg.split}] batch={bidx}",
                vmin = None,
                vmax = None,
            )
            _plot_per_channel_diff(
                gt_bt=target_bt,
                pred_bt=pred_bt,
                outdir=outdir,
                stem=stem,
                title_prefix=f"[{cfg.split}] batch={bidx}",
                cmap="coolwarm",  # diverging; center at 0
                vmax=0.15,
            )
            print(f"Saved per-channel figs: {os.path.join(outdir, stem)}__ch**.png", flush=True)
            num_done += 1
            if num_done >= cfg.batches:
                break


if __name__ == "__main__":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
