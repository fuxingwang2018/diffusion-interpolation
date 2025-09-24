# models/ddpm.py
from __future__ import annotations
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info

from .base import DiffusionBase


# ---------- Schedules ----------

def _cosine_beta_schedule(
    T: int,
    s: float = 0.02,                 # slightly larger s than the usual 0.008 -> more stable tails
    device=None,
    dtype=None
):
    t = torch.linspace(0, T, T + 1, device=device, dtype=dtype)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-6, 0.999)


def _linear_beta_schedule(
    T: int,
    beta_start=1e-4,
    beta_end=0.02,
    device=None,
    dtype=None
):
    return torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)


# ---------- Model ----------

class DDPMInterpolator(DiffusionBase):
    """
    Conditional diffusion with discrete timesteps (DDPM / DDIM).
    - Training target: ε (noise) or v (progressive distillation target)
    - Sampling: DDPM or DDIM
    - Stability features:
        * clamp ā_t and (1 - ā_t)
        * optional min-SNR loss weighting
        * safer cosine schedule (configurable s)
        * optional x0 clamp during sampling (debug)
        * optional strict float32
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

        # attention passthrough (if UNet2D supports it)
        use_attention: bool = False,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        attn_levels: Optional[Tuple[str, ...]] = ("mid",),

        # logging / saving
        sample_every_val: int = 1,
        sample_save_npz: bool = False,
        figures_cfg: Optional[dict] = None,
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,

        # DDPM specific
        T: int = 1000,
        schedule: str = "cosine",      # 'cosine' or 'linear'
        cosine_s: float = 0.02,         # safer tail for cosine schedule
        predict: str = "eps",           # 'eps' or 'v'
        loss_weighting: str = "none",   # 'none' or 'snr'
        snr_gamma: float = 5.0,         # cap for min-SNR loss weighting

        # numeric guards
        clamp_eps: float = 1e-5,        # clamp for abar and (1-abar)
        x0_clip: Optional[float] = None,# e.g., 3.0 to debug explosions; None disables
        force_float32: bool = False,    # set to True to force FP32 end-to-end

        # SDEdit knobs
        sdedit_enabled: bool = False,
        sdedit_start_t: Optional[int] = None,
        sdedit_start_pct: Optional[float] = None,

        # misc
        debug_sampling: bool = False,   # print per-step stats during sampling
        init_with_ones: bool = False,   # forwarded to base (tests)
    ):
        # ---- build base (UNet, etc.) ----
        super().__init__(
            cond_channels=cond_channels,
            target_channels=target_channels,
            extra_coord_channels=extra_coord_channels,
            extra_phys_time_scalar=extra_phys_time_scalar,
            unet_base=unet_base,
            time_embed_dim=time_embed_dim,
            use_attention=use_attention,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_levels=attn_levels,
            sample_every_val=sample_every_val,
            sample_save_npz=sample_save_npz,
            figures_cfg=figures_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            init_with_ones=init_with_ones,
        )

        # ensure val cache exists even if base didn't set it
        if not hasattr(self, "_val_cache"):
            self._val_cache = []  # type: ignore[attr-defined]

        # hyperparams
        self.T = int(T)
        self.predict = str(predict)
        self.loss_weighting = str(loss_weighting)
        self.snr_gamma = float(snr_gamma)
        self.clamp_eps = float(clamp_eps)
        self.x0_clip = float(x0_clip) if x0_clip is not None else None
        self.force_float32 = bool(force_float32)
        self.debug_sampling = bool(debug_sampling)

        # schedule buffers
        device = torch.device("cpu")
        dtype = torch.float32

        if schedule == "cosine":
            betas = _cosine_beta_schedule(self.T, s=float(cosine_s), device=device, dtype=dtype)
        elif schedule == "linear":
            betas = _linear_beta_schedule(self.T, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        one = torch.tensor([1.0], device=betas.device, dtype=betas.dtype)
        alphas_cumprod_prev = torch.cat([one, alphas_cumprod[:-1]], dim=0)

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
        try:
             rank_zero_info(f"XXXXXXXXX Stage: {stage}")
             rank_zero_info(x_cond.shape)
             rank_zero_info(y_clean.shape)
             rank_zero_info(meta)
        except Exception as e:
            pass

        meta = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x_cond, y_clean, meta = batch
        else:
            x_cond, y_clean = batch

        # enforce float32 if requested
        if self.force_float32:
            x_cond = x_cond.float()
            y_clean = y_clean.float()

        B = y_clean.shape[0]
        device = y_clean.device
        t = torch.randint(0, self.T, (B,), device=device)

        eps = self.clamp_eps
        abar_t = self.alphas_cumprod[t].clamp(min=eps, max=1 - eps)    # (B,)
        one_m_abar = (1 - abar_t).clamp(min=eps)

        sqrt_abar   = abar_t.sqrt()[:, None, None, None]
        sqrt_1mabar = one_m_abar.sqrt()[:, None, None, None]

        noise = torch.randn_like(y_clean)
        y_noisy = sqrt_abar * y_clean + sqrt_1mabar * noise

        # network expects a scalar time embedding per sample; here normalized t in [0,1]
        t_norm = (t.float() / float(self.T)).clamp(0, 1)
        pred = super().forward(y_noisy, x_cond, t_norm)

        # targets
        if self.predict == "eps":
            target = noise
        elif self.predict == "v":
            target = sqrt_abar * y_noisy - sqrt_1mabar * noise
        else:
            raise ValueError(f"Unknown predict='{self.predict}'")

        # loss
        if self.loss_weighting == "snr":
            snr = (abar_t / one_m_abar)
            snr_clamped = snr.clamp(max=self.snr_gamma)
            if self.predict == "eps":
                w = snr_clamped / (snr + 1e-8)
            else:  # 'v'
                w = snr_clamped / (snr + 1.0)

            loss = F.mse_loss(pred, target, reduction="none")
            loss = (loss * w[:, None, None, None]).mean()
        else:
            loss = F.mse_loss(pred, target, reduction="mean")

        self.log(
            f"{stage}_loss", loss, prog_bar=True,
            on_step=(stage == "train"), on_epoch=True, sync_dist=True, batch_size=B
        )
        if stage == "val":
            if not hasattr(self, "_val_cache"):
                self._val_cache = []  # type: ignore[attr-defined]
            self._cache_val_batch(x_cond, y_clean, meta)
        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):  # noqa: ARG002
            return self._shared_step(batch, "test")

    # ----- helpers -----
    def _maybe_float32(self, *tensors):
        if not self.force_float32:
            return tensors if len(tensors) > 1 else tensors[0]
        outs = tuple(t.float() for t in tensors)
        return outs if len(outs) > 1 else outs[0]

    # ----- samplers -----
    @torch.no_grad()
    def ddpm_sample(self, x_cond: torch.Tensor, shape_target: Tuple[int, int, int]) -> torch.Tensor:
        x_cond = self._maybe_float32(x_cond)
        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]
        y = torch.randn((B, Cy, H, W), device=x_cond.device, dtype=x_cond.dtype)

        eps = self.clamp_eps

        for t_int in reversed(range(self.T)):
            t = torch.full((B,), t_int, device=x_cond.device, dtype=torch.long)
            t_norm = (t.float() / float(self.T)).clamp(0, 1)

            abar_t = self.alphas_cumprod[t].clamp(min=eps, max=1 - eps)
            one_m_abar = (1 - abar_t).clamp(min=eps)
            sqrt_abar   = abar_t.sqrt()[:, None, None, None]
            sqrt_1mabar = one_m_abar.sqrt()[:, None, None, None]

            pred = super().forward(y, x_cond, t_norm)
            if self.predict == "eps":
                eps_hat = pred
            else:  # 'v'
                # eps = (sqrt(abar) * y - v) / sqrt(1 - abar)
                eps_hat = (sqrt_abar * y - pred) / sqrt_1mabar

            # x0 estimate
            x0 = (y - sqrt_1mabar * eps_hat) / sqrt_abar
            if self.x0_clip is not None:
                x0 = x0.clamp_(-self.x0_clip, self.x0_clip)

            # DDPM posterior mean
            den = one_m_abar
            coef1 = (self.alphas_cumprod_prev[t].sqrt() * self.betas[t] / den)
            coef2 = ((1 - self.alphas_cumprod_prev[t]) * self.alphas[t].sqrt() / den)
            mean = coef1[:, None, None, None] * x0 + coef2[:, None, None, None] * y

            if t_int > 0:
                # posterior variance β̃_t
                posterior_var = ((1 - self.alphas_cumprod_prev[t]) / den) * self.betas[t]
                y = mean + posterior_var.sqrt()[:, None, None, None] * torch.randn_like(y)
            else:
                y = mean

            if self.debug_sampling and (B == 1):
                print({
                    "t": t_int,
                    "abar_t": float(abar_t.mean()),
                    "y_mean": float(y.mean()),
                    "y_std": float(y.std()),
                    "pred_mean": float(pred.mean()),
                    "pred_std": float(pred.std()),
                })

        return y

    @torch.no_grad()
    def ddim_sample(
        self,
        x_cond: torch.Tensor,
        shape_target: Tuple[int, int, int],
        eta: float = 0.0,
        steps: int = 50
    ) -> torch.Tensor:
        x_cond = self._maybe_float32(x_cond)
        device = x_cond.device
        ts = torch.linspace(0, self.T - 1, steps, device=device).long()

        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]
        y = torch.randn((B, Cy, H, W), device=device, dtype=x_cond.dtype)

        eps = self.clamp_eps

        for i in reversed(range(len(ts))):
            t = ts[i].expand(B)
            t_norm = (t.float() / float(self.T)).clamp(0, 1)

            abar_t = self.alphas_cumprod[t].clamp(min=eps, max=1 - eps)
            one_m_abar = (1 - abar_t).clamp(min=eps)
            sqrt_abar   = abar_t.sqrt()[:, None, None, None]
            sqrt_1mabar = one_m_abar.sqrt()[:, None, None, None]

            pred = super().forward(y, x_cond, t_norm)
            if self.predict == "eps":
                eps_hat = pred
                x0 = (y - sqrt_1mabar * eps_hat) / sqrt_abar
            else:  # 'v'
                v = pred
                x0 = sqrt_abar * y - sqrt_1mabar * v

            if self.x0_clip is not None:
                x0 = x0.clamp_(-self.x0_clip, self.x0_clip)

            if i == 0:
                y = x0
                break

            t_prev = ts[i - 1].expand(B)
            abar_prev = self.alphas_cumprod[t_prev].clamp(min=eps, max=1 - eps)

            # DDIM update
            sigma = eta * (((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).clamp(min=0).sqrt())
            dir_xt = ((1 - abar_prev - sigma**2).clamp(min=0).sqrt())[:, None, None, None] * \
                     (y - sqrt_abar * x0) / sqrt_1mabar
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

        x_cond = self._maybe_float32(x_cond)
        eps = self.clamp_eps

        # SDEdit: make init_y from y0_clean
        if self.sdedit_enabled and (y0_clean is not None) and (init_y is None):
            if start_t is None:
                if self.sdedit_start_t is not None:
                    start_t = int(self.sdedit_start_t)
                elif self.sdedit_start_pct is not None:
                    start_t = int(max(0, min(1.0, float(self.sdedit_start_pct))) * (self.T - 1))
                else:
                    start_t = self.T // 2
            abar_t = self.alphas_cumprod[torch.tensor(start_t, device=x_cond.device, dtype=torch.long)]
            abar_t = abar_t.clamp(min=eps, max=1 - eps)
            sqrt_abar   = abar_t.sqrt()
            sqrt_1mabar = (1 - abar_t).clamp(min=eps).sqrt()
            noise = torch.randn_like(y0_clean)
            init_y = sqrt_abar * y0_clean + sqrt_1mabar * noise

        # No partial start -> run full sampler
        if (init_y is None) or (start_t is None):
            return self.ddpm_sample(x_cond, shape_target) if sampler == "ddpm" else \
                   self.ddim_sample(x_cond, shape_target, eta=eta, steps=steps)

        # Partial DDIM from start_t → 0
        device = x_cond.device
        B = x_cond.shape[0]
        ts = torch.linspace(0, self.T - 1, steps, device=device).long()
        ts = ts[ts <= start_t]
        if len(ts) == 0:
            return init_y.to(device, dtype=x_cond.dtype)

        y = init_y.to(device, dtype=x_cond.dtype)

        for i in reversed(range(len(ts))):
            t = ts[i].expand(B)
            t_norm = (t.float() / float(self.T)).clamp(0, 1)

            abar_t = self.alphas_cumprod[t].clamp(min=eps, max=1 - eps)
            one_m_abar = (1 - abar_t).clamp(min=eps)
            sqrt_abar   = abar_t.sqrt()[:, None, None, None]
            sqrt_1mabar = one_m_abar.sqrt()[:, None, None, None]

            pred = super().forward(y, x_cond, t_norm)
            if self.predict == "eps":
                eps_hat = pred
                x0 = (y - sqrt_1mabar * eps_hat) / sqrt_abar
            else:
                v = pred
                x0 = sqrt_abar * y - sqrt_1mabar * v

            if self.x0_clip is not None:
                x0 = x0.clamp_(-self.x0_clip, self.x0_clip)

            if i == 0:
                y = x0
                break

            t_prev = ts[i - 1].expand(B)
            abar_prev = self.alphas_cumprod[t_prev].clamp(min=eps, max=1 - eps)

            sigma = eta * (((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).clamp(min=0).sqrt())
            dir_xt = ((1 - abar_prev - sigma**2).clamp(min=0).sqrt())[:, None, None, None] * \
                     (y - sqrt_abar * x0) / sqrt_1mabar
            y = abar_prev.sqrt()[:, None, None, None] * x0 + dir_xt
            if eta > 0:
                y = y + sigma[:, None, None, None] * torch.randn_like(y)

        return y
