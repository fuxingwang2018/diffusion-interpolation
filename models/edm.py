from __future__ import annotations
from typing import Any, Optional, Tuple, List, Dict
import torch
import torch.nn.functional as F

from .base import DiffusionBase
from .common import default

class EDMInterpolator(DiffusionBase):
    """
    Conditional diffusion (EDM-style) that learns to map:
      endpoints (first+last)  --->  internal lead-times
    We add noise to y_clean only, predict ε(y, x_cond, σ).
    """

    def __init__(
        self,
        # channels/extras/UNet/logging are handled by DiffusionBase
        cond_channels: int,
        target_channels: int,
        extra_coord_channels: bool = False,
        extra_phys_time_scalar: Optional[float] = None,
        unet_base: int = 64,
        time_embed_dim: int = 256,
        sample_every_val: int = 1,
        sample_save_npz: bool = False,
        figures_cfg: Optional[dict] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,
        # --- EDM noise/schedule ---
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        loss_weighting: str = "edm",  # 'edm' or 'none'
        # sampling schedule
        sample_steps: int = 20,
        rho: float = 7.0,
        # --- SDEdit knobs ---
        sdedit_enabled: bool = False,
        sdedit_sigma: Optional[float] = None,  # if None: use schedule midpoint
    ):
        super().__init__(
            cond_channels=cond_channels, target_channels=target_channels,
            extra_coord_channels=extra_coord_channels, extra_phys_time_scalar=extra_phys_time_scalar,
            unet_base=unet_base, time_embed_dim=time_embed_dim,
            sample_every_val=sample_every_val, sample_save_npz=sample_save_npz,
            figures_cfg=figures_cfg, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg,
        )
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.loss_weighting = str(loss_weighting)

        self.sample_steps = int(sample_steps)
        self.rho = float(rho)

        self.sdedit_enabled = bool(sdedit_enabled)
        self.sdedit_sigma = sdedit_sigma

    # ----- EDM utilities -----
    @torch.no_grad()
    def get_sigma_schedule(self, steps: int, rho: float) -> torch.Tensor:
        s0, s1 = self.sigma_max, self.sigma_min
        ramp = torch.linspace(0, 1, steps, device=self.device)
        sigmas = (s0 ** (1 / rho) + ramp * (s1 ** (1 / rho) - s0 ** (1 / rho))) ** rho
        return torch.flip(sigmas, dims=[0])  # descending

    def sample_sigmas_train(self, b: int) -> torch.Tensor:
        sigma = torch.exp(torch.randn(b, device=self.device) * self.p_std + self.p_mean)
        return sigma.clamp(self.sigma_min, self.sigma_max)

    # ----- training / validation -----
    def _shared_step(self, batch: Any, stage: str):
        """
        batch = (x_cond, y_clean[, meta])
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

        # forward expects log(σ)
        eps_hat = super().forward(y_noisy, x_cond, sigma.log())

        if self.loss_weighting == "edm":
            w = (sigma ** 2)[:, None, None, None]
            loss = F.mse_loss(eps_hat * w.sqrt(), noise * w.sqrt(), reduction="mean")
        else:
            loss = F.mse_loss(eps_hat, noise, reduction="mean")

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        if stage == "val":
            self._cache_val_batch(x_cond, y_clean, meta)
        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self._shared_step(batch, "val")

    # ----- sampling (EDM Euler) with optional SDEdit -----
    @torch.no_grad()
    def sample_from_cond(
        self,
        x_cond: torch.Tensor,
        shape_target: Tuple[int, int, int],
        steps: Optional[int] = None,
        init_y: Optional[torch.Tensor] = None,   # if provided, also pass start_sigma
        start_sigma: Optional[float] = None,
        y0_clean: Optional[torch.Tensor] = None, # SDEdit clean target (if enabled)
    ) -> torch.Tensor:

        steps = default(steps, self.sample_steps)
        sigmas = self.get_sigma_schedule(steps=steps, rho=self.rho)  # (S,)

        # --- SDEdit: construct init_y from y0_clean if requested ---
        if self.sdedit_enabled and (y0_clean is not None) and (init_y is None):
            sigma_use = float(self.sdedit_sigma) if self.sdedit_sigma is not None else float(sigmas[len(sigmas)//2])
            init_y = y0_clean + sigma_use * torch.randn_like(y0_clean)
            start_sigma = sigma_use

        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]

        if init_y is not None and start_sigma is not None:
            s_idx = torch.argmin((sigmas - float(start_sigma)).abs()).item()
            y = init_y.clone().to(x_cond.device)
            start = s_idx
        else:
            y = torch.randn((B, Cy, H, W), device=x_cond.device) * sigmas[0]
            start = 0

        for i in range(start, len(sigmas)):
            sigma = sigmas[i]
            eps = super().forward(y, x_cond, torch.full((B,), float(sigma), device=x_cond.device).log())
            x0_hat = y - sigma * eps
            if i == len(sigmas) - 1:
                y = x0_hat
            else:
                y = x0_hat + sigmas[i + 1] * eps
        return y
