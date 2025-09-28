# diffusion/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set, Tuple
from abc import ABC, abstractmethod
from hydra.utils import instantiate

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info


# =========================
# Types & small configs
# =========================

 


def fourier_embed(x: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """
    Fourier features on (log-)noise/time scalars.
    x: (B,)
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), half, device=x.device, dtype=x.dtype)
    )
    x = x.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# =========================
# Abstract Diffusion Base
# =========================

class DiffusionBase(L.LightningModule, ABC):
    """
    Abstract diffusion base (no VAE):
      - owns a UNet2D backbone (`self.net`) and a time/noise embedding MLP (`self.emb_mlp`)
      - packs (x,y) -> (cond, target) in **data space** (no latent encoding)
      - exposes optimizer/scheduler via Hydra-friendly dict configs
      - subclasses must implement:
          * allowed_samplers() -> Set[str]
          * _sample_sigma(...)
          * _denoise_target(...)
          * _compute_loss(...)
    """

    def __init__(
        self,
        data_shape: dict[str, Tuple[int, int, int, int]],                
        network_cfg: Optional[dict],                
        optimizer_cfg: Optional[dict] = None,   
        scheduler_cfg: Optional[dict] = None,   
        sampler_cfg: Optional[SamplerBase] = None, 
    ):
        super().__init__()

        # ---- keep for checkpoints/logging
        self.data_shape = data_shape
        self.sampler_cfg = sampler_cfg
        self._sampler = None

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        

 

        # Conditioning
        
        T, C, _ , _= data_shape['x_shape']
        self.cond_ch = 2 * C
        T, C, _ , _= data_shape['y_shape']
        self.target_ch = T * C

        

        self.network = instantiate(network_cfg,in_ch=self.cond_ch + self.target_ch,
            out_ch=self.target_ch, )
            

        # ---- time/noise embedding MLP
        self.emb_mlp = nn.Sequential(
            nn.Linear(64, self.network.emb_dim),
            nn.SiLU(),
            nn.Linear(self.network.emb_dim, self.network.emb_dim),
        )


    @property
    def sampler(self) -> SamplerBase:
        # lazily instantiate once
        if self._sampler is None:
            if self.sampler_cfg is None:
                raise ValueError("sampler_cfg is not set; cannot create sampler")
            s = instantiate(self.sampler_cfg)  # e.g. {"_target_": "samplers.HeunEDM", ...}
            if s.name not in self.allowed_samplers():
                raise ValueError(f"Sampler '{s.name}' not allowed for this model "
                                f"(allowed: {self.allowed_samplers()})")
            self._sampler = s
        return self._sampler
    # -------- training / validation cache wrapper --------



    # ---------- required API for subclasses ----------
    @abstractmethod
    def allowed_samplers(self) -> Set[str]:
        """Set of sampler names allowed for this model, e.g. {'heun_edm'} or {'ddim','iddpm'}."""
        ...

    @abstractmethod
    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ...

    @abstractmethod
    def _denoise_target(
        self, cond_clean: torch.Tensor, target_noisy: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def _compute_loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...

    # ---------- helpers ----------
    def _pack_xy(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 2, C, H, W) or (2, C, H, W)
        y: (B, K, C, H, W) or (K, C, H, W)
        returns:
          cond:   (B, 2*C, H, W)
          target: (B, K*C, H, W)
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        B, two, C, H, W = x.shape
        _, K, C2, H2, W2 = y.shape
        if not (two == 2 and C2 == C and H2 == H and W2 == W):
            raise ValueError(f"Mismatch in (x,y) shapes: x={tuple(x.shape)} y={tuple(y.shape)}")
        cond = x.reshape(B, 2 * C, H, W)
        target = y.reshape(B, K * C, H, W)
        return cond, target


    def _shared_step(self, batch, stage):
        """
        batch = (x, y[, meta])
        """
        meta = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
        cond, target = self._pack_xy(x, y)
        loss = self._compute_loss(cond, target)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        return loss
    # ---------- Lightning ----------
    def training_step(self, batch, _):
        return  self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    # ---------- optimizers (Hydra-friendly) ----------
    def configure_optimizers(self):
        from hydra.utils import instantiate
        if isinstance(self.optimizer_cfg, dict) and "_target_" in self.optimizer_cfg:
            opt = instantiate(self.optimizer_cfg, params=self.parameters())
        else:
            raise ValueError("no hydra optimizer target in optimizer_cfg")
        if self.scheduler_cfg:
            try:
                sch = instantiate(self.scheduler_cfg, optimizer=opt)
                return {"optimizer": opt, "lr_scheduler": sch}
            except Exception:
                # If scheduler instantiation fails, just return the optimizer
                return opt
        return opt

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        cond: (B, cond_ch, H, W)  clean conditioning
        target_shape: (B, target_ch, H, W)  desired target tensor shape
        """
        self.eval()
        out = self.sampler.sample(self, cond, target_shape, cond.device)  # sampler drives the schedule
        self.train()  # restore if you want to stay in train mode normally
        return out

   