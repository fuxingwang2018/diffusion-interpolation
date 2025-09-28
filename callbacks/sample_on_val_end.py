# callbacks/sample_on_val_end.py
from __future__ import annotations
import os
import warnings
from typing import Any, Optional, Tuple, Dict

import torch
import lightning as L
# safer: from lightning.pytorch.callbacks import Callback  # then subclass Callback

class SampleOnValEndCallback(L.Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        save_npz: bool = False,
        out_dir: Optional[str] = None,
        max_plot_channels: int = 3,
    ):
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_npz = bool(save_npz)
        self.out_dir = out_dir
        self.max_plot_channels = max_plot_channels
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]] = None

    def on_validation_batch_end( #  on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._cache is not None:
            return
        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch[:2]
                meta = None
        else:
            return
        self._cache = (x.detach().cpu(), y.detach().cpu(), meta)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            self._cache = None
            return
        if not trainer.is_global_zero:
            self._cache = None
            return
        if self._cache is None:
            return

        x, y, meta = self._cache
        # move to device
        x = x.to(pl_module.device, non_blocking=True)
        y = y.to(pl_module.device, non_blocking=True)

        # build cond/target using your model helper
        cond, target = pl_module._pack_xy(x, y)  # (B, 2*C, H, W), (B, T*C, H, W)

        # run sampler -> prediction shaped like target
        with torch.no_grad():
            y_pred = pl_module.sample(cond=cond, target_shape=target.shape)
        print(f"y_pred.shape: {y_pred.shape}, target.shape: {target.shape}")
        print(f"y_pred.abs().min(): {y_pred.abs().min()}, y_pred.abs().max(): {y_pred.abs().max()}")
        print(f"cond.abs().min(): {cond.abs().min()}, cond.abs().max(): {cond.abs().max()}")

        # simple plotting util (kept minimal)
        def _grid(t: torch.Tensor, title: str, max_ch=3):
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception as e:
                warnings.warn(f"Matplotlib not available: {e}")
                return None
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

        root = self.out_dir or os.path.join(trainer.default_root_dir, "val_samples")
        os.makedirs(root, exist_ok=True)
        stem = f"epoch{trainer.current_epoch:04d}"
        png_cond = os.path.join(root, f"{stem}__cond.png")
        png_pred = os.path.join(root, f"{stem}__pred.png")
        png_gt   = os.path.join(root, f"{stem}__gt.png")
        npz_path = os.path.join(root, f"{stem}__sample.npz")

        figs = [
            (png_cond, _grid(cond, "cond endpoints", self.max_plot_channels)),
            (png_pred, _grid(y_pred, "pred internals", self.max_plot_channels)),
            (png_gt,   _grid(target, "gt internals", self.max_plot_channels)),
        ]
        try:
            import matplotlib.pyplot as plt
            for path, fig in figs:
                if fig is None:
                    continue
                fig.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            warnings.warn(f"Saving figures failed: {e}")

        if self.save_npz:
            try:
                import numpy as np
                np.savez_compressed(
                    npz_path,
                    cond=cond[0].detach().cpu().numpy(),
                    pred=y_pred[0].detach().cpu().numpy(),
                    gt=target[0].detach().cpu().numpy(),
                    meta=meta if isinstance(meta, dict) else {},
                )
            except Exception as e:
                warnings.warn(f"Saving NPZ failed: {e}")

        self._cache = None
