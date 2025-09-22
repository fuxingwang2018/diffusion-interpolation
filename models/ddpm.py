from __future__ import annotations
from typing import Any, Optional, Tuple, Dict
import torch
import torch.nn.functional as F

from .base import DiffusionBase
from .common import default

def _cosine_beta_schedule(T: int, s: float = 0.008, device=None):
    t = torch.linspace(0, T, T+1, device=device)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.999)

def _linear_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02, device=None):
    return torch.linspace(beta_start, beta_end, T, device=device)

class DDPMInterpolator(DiffusionBase):
    """
    Conditional diffusion with discrete timesteps (DDPM / DDIM).
    Train to predict ε or v; sample via DDPM or DDIM.
    """

    def __init__(
        self,
        # base (channels/extras/UNet/logging)
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
        # DDPM specific
        T: int = 1000,
        schedule: str = "cosine",  # 'cosine' or 'linear'
        predict: str = "eps",      # 'eps' or 'v' (implicit)
        loss_weighting: str = "none",
        # SDEdit knobs
        sdedit_enabled: bool = False,
        sdedit_start_t: Optional[int] = None,
        sdedit_start_pct: Optional[float] = None,
    ):
        super().__init__(
            cond_channels=cond_channels, target_channels=target_channels,
            extra_coord_channels=extra_coord_channels, extra_phys_time_scalar=extra_phys_time_scalar,
            unet_base=unet_base, time_embed_dim=time_embed_dim,
            sample_every_val=sample_every_val, sample_save_npz=sample_save_npz,
            figures_cfg=figures_cfg, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg,
        )

        self.T = int(T)
        self.predict = predict
        self.loss_weighting = str(loss_weighting)

        # schedule buffers
        device = torch.device("cpu")
        if schedule == "cosine":
            betas = _cosine_beta_schedule(self.T, device=device)
        elif schedule == "linear":
            betas = _linear_beta_schedule(self.T, device=device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # SDEdit
        self.sdedit_enabled = bool(sdedit_enabled)
        self.sdedit_start_t = sdedit_start_t
        self.sdedit_start_pct = sdedit_start_pct

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
        device = y_clean.device
        t = torch.randint(0, self.T, (B,), device=device)

        abar_t = self.alphas_cumprod[t]  # (B,)
        noise = torch.randn_like(y_clean)
        y_noisy = abar_t.sqrt()[:, None, None, None] * y_clean + (1 - abar_t).sqrt()[:, None, None, None] * noise

        # forward expects normalized t in [0,1]
        t_norm = (t.float() / float(self.T)).clamp(0, 1)
        pred = super().forward(y_noisy, x_cond, t_norm)

        if self.predict == "eps":
            target = noise
            loss = F.mse_loss(pred, target, reduction="mean")
        elif self.predict == "v":
            # v = a*y - sqrt(1-a)*eps, a = sqrt(abar)
            v_target = abar_t.sqrt()[:, None, None, None] * y_noisy - (1 - abar_t).sqrt()[:, None, None, None] * noise
            loss = F.mse_loss(pred, v_target, reduction="mean")
        else:
            raise ValueError(f"Unknown predict='{self.predict}'")

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)
        if stage == "val":
            self._cache_val_batch(x_cond, y_clean, meta)
        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self._shared_step(batch, "val")

    # ----- samplers -----
    @torch.no_grad()
    def ddpm_sample(self, x_cond: torch.Tensor, shape_target: Tuple[int, int, int]) -> torch.Tensor:
        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]
        y = torch.randn((B, Cy, H, W), device=x_cond.device)
        for t_int in reversed(range(self.T)):
            t = torch.full((B,), t_int, device=x_cond.device, dtype=torch.long)
            t_norm = (t.float() / float(self.T)).clamp(0, 1)
            bet = self.betas[t]
            abar_t = self.alphas_cumprod[t]
            eps_or_v = super().forward(y, x_cond, t_norm)
            if self.predict == "eps":
                eps = eps_or_v
            else:
                v = eps_or_v
                # eps = (a*y - v)/sqrt(1-a), a = sqrt(abar)
                eps = (abar_t.sqrt()[:, None, None, None] * y - v) / (1 - abar_t).sqrt()[:, None, None, None]

            # mean step (DDPM equation)
            y0 = (y - (1 - abar_t).sqrt()[:, None, None, None] * eps) / abar_t.sqrt()[:, None, None, None]
            coef1 = (self.alphas_cumprod_prev[t].sqrt() * self.betas[t] / (1 - abar_t))
            coef2 = ((1 - self.alphas_cumprod_prev[t]) * self.alphas[t].sqrt() / (1 - abar_t))
            mean = coef1[:, None, None, None] * y0 + coef2[:, None, None, None] * y
            if t_int > 0:
                var = self.betas[t]
                noise = torch.randn_like(y)
                y = mean + var.sqrt()[:, None, None, None] * noise
            else:
                y = mean
        return y

    @torch.no_grad()
    def ddim_sample(self, x_cond: torch.Tensor, shape_target: Tuple[int, int, int], eta: float = 0.0, steps: int = 50) -> torch.Tensor:
        # choose subset of timesteps
        device = x_cond.device
        ts = torch.linspace(0, self.T - 1, steps, device=device).long()
        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]
        y = torch.randn((B, Cy, H, W), device=device)

        for i in reversed(range(len(ts))):
            t = ts[i].expand(B)
            t_norm = (t.float() / float(self.T)).clamp(0, 1)
            abar_t = self.alphas_cumprod[t]

            pred = super().forward(y, x_cond, t_norm)
            if self.predict == "eps":
                eps = pred
                x0 = (y - (1 - abar_t).sqrt()[:, None, None, None] * eps) / abar_t.sqrt()[:, None, None, None]
            else:
                v = pred
                x0 = (abar_t.sqrt())[:, None, None, None] * y - ((1 - abar_t).sqrt())[:, None, None, None] * v

            if i == 0:
                y = x0
                break

            t_prev = ts[i - 1].expand(B)
            abar_prev = self.alphas_cumprod[t_prev]

            # DDIM update
            sigma = eta * (((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).clamp(min=0).sqrt())
            dir_xt = ((1 - abar_prev - sigma**2).clamp(min=0).sqrt())[:, None, None, None] * \
                     (y - abar_t.sqrt()[:, None, None, None] * x0) / (1 - abar_t).sqrt()[:, None, None, None]
            y = abar_prev.sqrt()[:, None, None, None] * x0 + dir_xt
            if eta > 0:
                y = y + sigma[:, None, None, None] * torch.randn_like(y)
        return y

    # ----- sampling wrapper with optional SDEdit -----
    @torch.no_grad()
    def sample_from_cond(
        self,
        x_cond: torch.Tensor,
        shape_target: Tuple[int, int, int],
        sampler: str = "ddim",
        eta: float = 0.0,
        steps: int = 50,
        init_y: Optional[torch.Tensor] = None,
        start_t: Optional[int] = None,
        y0_clean: Optional[torch.Tensor] = None,  # SDEdit clean
    ) -> torch.Tensor:

        # Construct SDEdit init_y if enabled & y0_clean is provided
        if self.sdedit_enabled and (y0_clean is not None) and (init_y is None):
            if start_t is None:
                if self.sdedit_start_t is not None:
                    start_t = int(self.sdedit_start_t)
                elif self.sdedit_start_pct is not None:
                    start_t = int(max(0, min(1.0, float(self.sdedit_start_pct))) * (self.T - 1))
                else:
                    start_t = self.T // 2
            abar_t = self.alphas_cumprod[torch.tensor(start_t, device=x_cond.device)]
            noise = torch.randn_like(y0_clean)
            init_y = abar_t.sqrt() * y0_clean + (1 - abar_t).sqrt() * noise

        # If no explicit SDEdit start is requested, run full sampler
        if (init_y is None) or (start_t is None):
            if sampler == "ddpm":
                return self.ddpm_sample(x_cond, shape_target)
            else:
                return self.ddim_sample(x_cond, shape_target, eta=eta, steps=steps)

        # Run partial DDIM from start_t → 0 using provided init_y
        device = x_cond.device
        B = x_cond.shape[0]
        ts = torch.linspace(0, self.T - 1, steps, device=device).long()
        ts = ts[ts <= start_t]
        if len(ts) == 0:
            return init_y.to(device)

        y = init_y.to(device)
        for i in reversed(range(len(ts))):
            t = ts[i].expand(B)
            t_norm = (t.float() / float(self.T)).clamp(0, 1)
            abar_t = self.alphas_cumprod[t]
            pred = super().forward(y, x_cond, t_norm)
            if self.predict == "eps":
                eps = pred
                x0 = (y - (1 - abar_t).sqrt()[:, None, None, None] * eps) / abar_t.sqrt()[:, None, None, None]
            else:
                v = pred
                x0 = (abar_t.sqrt())[:, None, None, None] * y - ((1 - abar_t).sqrt())[:, None, None, None] * v

            if i == 0:
                y = x0
                break

            t_prev = ts[i - 1].expand(B)
            abar_prev = self.alphas_cumprod[t_prev]
            sigma = eta * (((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).clamp(min=0).sqrt())
            dir_xt = ((1 - abar_prev - sigma**2).clamp(min=0).sqrt())[:, None, None, None] * \
                     (y - abar_t.sqrt()[:, None, None, None] * x0) / (1 - abar_t).sqrt()[:, None, None, None]
            y = abar_prev.sqrt()[:, None, None, None] * x0 + dir_xt
            if eta > 0:
                y = y + sigma[:, None, None, None] * torch.randn_like(y)
        return y
