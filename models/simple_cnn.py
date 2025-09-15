from __future__ import annotations
from typing import Any, Optional, Tuple, List

import io
import os
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix

# Optional: MLflow for custom artifact logging
try:
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless logging
    import matplotlib.pyplot as plt
except Exception as e:  # noqa: BLE001
    warnings.warn(f"Matplotlib not available; figure logging disabled: {e}")
    plt = None


class SimpleCNN(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        compile_cfg: Optional[dict] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
        figures: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["compile_cfg", "optimizer_cfg", "scheduler_cfg", "figures"]
        )
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg  # may be None
        self.fig_cfg = figures or {}
        self.example_limit: int = int(self.fig_cfg.get("max_misclassified", 36))

        c = 32
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Dropout(dropout),
            nn.Conv2d(c, 2 * c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(2 * c * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Metrics (DDP-safe via TorchMetrics)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

        # For figure generation
        self.val_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="none")
        self._mis_images: List[torch.Tensor] = []
        self._mis_targets: List[torch.Tensor] = []
        self._mis_preds: List[torch.Tensor] = []

        # optional torch.compile
        if compile_cfg and compile_cfg.get("enabled", False):
            try:
                self.net = torch.compile(
                    self.net,
                    mode=compile_cfg.get("mode", "default"),
                    backend=compile_cfg.get("backend", "inductor"),
                )
            except Exception as e:  # noqa: BLE001
                self.print(f"[warn] torch.compile disabled: {e}")

    # ---------- core ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)

        # metrics
        acc = self.train_acc(preds, y) if stage == "train" else self.val_acc(preds, y)

        # ddp-safe logs
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # figures: accumulate val confusion matrix + misclassified samples
        if stage == "val":
            self.val_cm.update(preds, y)
            if self.fig_cfg and self.example_limit > 0 and len(self._mis_images) < self.example_limit:
                mis_mask = preds.ne(y)
                if mis_mask.any():
                    need = self.example_limit - len(self._mis_images)
                    idx = mis_mask.nonzero(as_tuple=False).squeeze(-1)[:need]
                    # store detached CPU copies (still normalized; we’ll unnormalize on plot)
                    self._mis_images.extend(x[idx].detach().cpu())
                    self._mis_targets.extend(y[idx].detach().cpu())
                    self._mis_preds.extend(preds[idx].detach().cpu())

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        self._shared_step(batch, "val")

    # ---------- epoch-end visualizations ----------
    def on_validation_epoch_end(self) -> None:
        # Only log figures from global rank 0 and only if matplotlib+mlflow are available
        should_log = (
            self.trainer.is_global_zero
            and plt is not None
            and mlflow is not None
            and any(
                self.fig_cfg.get(k, False)
                for k in ["log_confusion_matrix", "log_per_class_error", "log_misclassified_grid"]
            )
        )
        if not should_log:
            # reset state and buffers regardless to avoid leaks across epochs
            self.val_cm.reset()
            self._mis_images.clear()
            self._mis_targets.clear()
            self._mis_preds.clear()
            return

        # Compute confusion matrix with distributed sync (TorchMetrics handles sync on compute)
        cm = self.val_cm.compute().detach().cpu()  # [C, C]
        num_classes = cm.shape[0]

        figs: list[Tuple[str, "plt.Figure"]] = []

        if self.fig_cfg.get("log_confusion_matrix", True):
            fig_cm = self._make_confusion_matrix_figure(cm)
            figs.append((f"figures/confusion_matrix_epoch_{self.current_epoch}.png", fig_cm))

        if self.fig_cfg.get("log_per_class_error", True):
            fig_err = self._make_per_class_error_figure(cm)
            figs.append((f"figures/per_class_error_epoch_{self.current_epoch}.png", fig_err))

        if self.fig_cfg.get("log_misclassified_grid", True) and len(self._mis_images) > 0:
            fig_grid = self._make_misclassified_grid_figure(
                self._mis_images, self._mis_targets, self._mis_preds
            )
            figs.append((f"figures/misclassified_grid_epoch_{self.current_epoch}.png", fig_grid))

        # Log figures to MLflow using the active Lightning run
        from lightning.pytorch.loggers import MLFlowLogger

        if isinstance(self.logger, MLFlowLogger):
            run_id = self.logger.run_id
            # ensure this process activates the Lightning-created run so mlflow.log_figure works
            try:
                with mlflow.start_run(run_id=run_id):
                    for path, fig in figs:
                        # You can stream to bytes to avoid tmp files
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        mlflow.log_figure(fig, path)
                        plt.close(fig)
            except Exception as e:  # noqa: BLE001
                self.print(f"[warn] MLflow figure logging failed: {e}")

        # reset for next epoch
        self.val_cm.reset()
        self._mis_images.clear()
        self._mis_targets.clear()
        self._mis_preds.clear()

    # ---------- optimizer ----------
    def configure_optimizers(self):
        from hydra.utils import instantiate

        opt = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg:
            sch = instantiate(self.scheduler_cfg, optimizer=opt)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
        return opt

    # ---------- plotting helpers ----------
    def _make_confusion_matrix_figure(self, cm: torch.Tensor):
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm.numpy(), interpolation="nearest")
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        # annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center")
        fig.tight_layout()
        return fig

    def _make_per_class_error_figure(self, cm: torch.Tensor):
        import numpy as np

        totals = cm.sum(dim=1).clamp(min=1)  # avoid /0
        correct = torch.diag(cm)
        per_class_error = 1.0 - (correct.float() / totals.float())
        xs = list(range(len(per_class_error)))

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(xs, per_class_error.numpy())
        ax.set_title("Per-class Error (1 - accuracy)")
        ax.set_xlabel("Class")
        ax.set_ylabel("Error")
        ax.set_xticks(xs)
        fig.tight_layout()
        return fig

    def _make_misclassified_grid_figure(
        self,
        imgs: List[torch.Tensor],
        targets: List[torch.Tensor],
        preds: List[torch.Tensor],
        ncols: int = 6,
    ):
        import math

        n = min(len(imgs), self.example_limit)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))
        axes = axes.flatten() if isinstance(axes, (list, tuple)) or hasattr(axes, "flatten") else [axes]

        # MNIST normalization used in datamodule
        mean, std = 0.1307, 0.3081
        for idx in range(n):
            ax = axes[idx]
            img = imgs[idx].clone()
            # unnormalize
            img = img * std + mean
            img = img.clamp(0, 1)
            ax.imshow(img.squeeze(0), cmap="gray")
            ax.set_title(f"p={int(preds[idx])} / y={int(targets[idx])}", fontsize=8)
            ax.axis("off")

        # hide any extra axes
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Misclassified Examples", y=0.98)
        fig.tight_layout()
        return fig
