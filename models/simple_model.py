# diffusion/base.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from hydra.utils import instantiate


class SimpleModel(L.LightningModule):
    """
    Deterministic model: learns y ~ f(x) with MSE loss.

    Shapes:
      x: (B, 2, C, H, W)      -> cond:   (B, 2*C, H, W)
      y: (B, T, C, H, W)      -> target: (B, T*C, H, W)

    The backbone `network` is instantiated from Hydra config and is expected
    to support either:
      forward(cond, emb)  with emb shape (B, emb_dim), or
      forward(cond)       (if no embedding is used).
    """

    def __init__(
        self,
        data_shape: dict[str, Tuple[int, int, int, int]],
        network_cfg: Optional[dict],
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
        *,
        emb_mode: str = "zeros",        # "learned" | "zeros"
        emb_dim: Optional[int] = None,    # fallback if network doesn't expose .emb_dim
        **kwargs,
    ):
        super().__init__()
        self.data_shape = data_shape
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Channels
        T_x, C_x, _, _ = data_shape["x_shape"]
        T_y, C_y, _, _ = data_shape["y_shape"]
        assert T_x == 2, f"x_shape first dim must be 2 (got {T_x})"
        self.cond_ch = 2 * C_x
        self.target_ch = T_y * C_y

        # Backbone
        self.network = instantiate(network_cfg, in_ch=self.cond_ch, out_ch=self.target_ch)

        # ---- Constant embedding setup ----
        # Determine embedding dim: prefer attribute on network; else constructor arg.
        net_emb_dim = getattr(self.network, "emb_dim", None)
        self.emb_dim = int(net_emb_dim if net_emb_dim is not None else (emb_dim or 0))
        self.use_embedding = self.emb_dim > 0

        self.emb_mode = emb_mode.lower().strip()
        if self.use_embedding:
            if self.emb_mode == "learned":
                # One learnable vector shared for all batches; expand on forward
                self.const_emb = nn.Parameter(torch.zeros(1, self.emb_dim))
                nn.init.normal_(self.const_emb, mean=0.0, std=0.02)
            elif self.emb_mode == "zeros":
                # Fixed zero vector (not trainable)
                self.register_buffer("const_emb", torch.zeros(1, self.emb_dim), persistent=False)
            else:
                raise ValueError(f"Unknown emb_mode='{emb_mode}', expected 'learned' or 'zeros'.")
        else:
            self.const_emb = None  # network likely doesn't take an embedding

    # ---------- forward & loss ----------
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: (B, 2*C, H, W) -> (B, T*C, H, W)"""
        if self.use_embedding:
            # Expand constant embedding to batch size
            B = cond.shape[0]
            emb = self.const_emb.expand(B, -1)
            return self.network(cond, emb)
        # Fallback if network ignores embedding
        return self.network(cond)

    def _compute_loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Deterministic MSE loss between prediction and target."""
        pred = self.forward(cond)
        if pred.shape != target.shape:
            raise RuntimeError(f"Pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        return F.mse_loss(pred, target)

    # ---------- helpers ----------
    def _pack_xy(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 2, C, H, W) or (2, C, H, W)
        y: (B, T, C, H, W) or (T, C, H, W)
        -> cond: (B, 2*C, H, W), target: (B, T*C, H, W)
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        B, two, C, H, W = x.shape
        _, T, C2, H2, W2 = y.shape
        if not (two == 2 and C2 == C and H2 == H and W2 == W):
            raise ValueError(f"Mismatch in (x,y) shapes: x={tuple(x.shape)} y={tuple(y.shape)}")
        cond = x.reshape(B, 2 * C, H, W)
        target = y.reshape(B, T * C, H, W)
        return cond, target

    def _shared_step(self, batch, stage: str):
        # batch = (x, y[, meta])
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        cond, target = self._pack_xy(x, y)
        loss = self._compute_loss(cond, target)
        bs = x.size(0)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    # ---------- Lightning hooks ----------
    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    # ---------- optimizers (Hydra-friendly) ----------
    def configure_optimizers(self):
        if isinstance(self.optimizer_cfg, dict) and "_target_" in self.optimizer_cfg:
            opt = instantiate(self.optimizer_cfg, params=self.parameters())
        else:
            raise ValueError("no hydra optimizer target in optimizer_cfg")
        if self.scheduler_cfg:
            try:
                sch = instantiate(self.scheduler_cfg, optimizer=opt)
                return {"optimizer": opt, "lr_scheduler": sch}
            except Exception:
                return opt
        return opt

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, target_shape: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """
        Deterministic inference.
        cond: (B, 2*C, H, W)
        target_shape: optional sanity check (B, T*C, H, W)
        """
        pred = self.forward(cond)
        if target_shape is not None and pred.shape != target_shape:
            raise RuntimeError(f"Pred shape {tuple(pred.shape)} != target_shape {tuple(target_shape)}")
        return pred
