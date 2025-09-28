# diffusion/edm.py
import torch
from typing import Tuple, Set, Optional

from .base import DiffusionBase, fourier_embed
from samplers import SamplerBase


def _edm_precond(sigma: torch.Tensor, sigma_data: float):
    """
    EDM preconditioning coefficients (Karras et al.).
    Returns c_skip, c_out, c_in, c_noise with broadcasting-friendly shapes.
    """
    s2 = sigma * sigma
    sd2 = sigma_data * sigma_data
    c_skip = sd2 / (s2 + sd2)
    c_out = sigma * (sd2 ** 0.5) / (s2 + sd2).sqrt()
    c_in = 1.0 / (s2 + sd2).sqrt()
    c_noise = 0.25 * torch.log(torch.clamp(sigma, min=1e-12))
    return c_skip, c_out, c_in, c_noise


class EDM(DiffusionBase):
    """
    EDM-style diffusion module (no VAE).
    Uses DiffusionBase's UNet backbone and time/noise embedding MLP.
    """

    def __init__(
        self,
        data_shape: Tuple[int, int, int, int],     # (T, C, H, W)
        network_cfg: Optional[dict],                   # UNet hyperparameters
        optimizer_cfg: Optional[dict],
        scheduler_cfg: Optional[dict],
        sampler_cfg: Optional[dict],
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        super().__init__(
            data_shape=data_shape,
            network_cfg=network_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            sampler_cfg=sampler_cfg,
        )
 


        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    # ---- Base API ----
    def allowed_samplers(self) -> Set[str]:
        return {"heun_edm"}

    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # log-normal sigma sampling as in EDM
        return torch.exp(torch.randn(batch_size, device=device) * self.p_std + self.p_mean)

    def _denoise_target(
        self,
        cond_clean: torch.Tensor,
        target_noisy: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = _edm_precond(sigma, self.sigma_data)
        emb = self.emb_mlp(fourier_embed(c_noise))
        net_in = torch.cat([cond_clean, c_in.view(-1, 1, 1, 1) * target_noisy], dim=1)
        fx = self.network(net_in, emb)
        x_hat = c_skip.view(-1, 1, 1, 1) * target_noisy + c_out.view(-1, 1, 1, 1) * fx
        return x_hat

    def _compute_loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, device = target.size(0), target.device
        sigma = self._sample_sigma(B, device)
        noise = torch.randn_like(target) * sigma.view(-1, 1, 1, 1)
        target_noisy = target + noise

        x_hat = self._denoise_target(cond, target_noisy, sigma)

        # EDM weighting
        sd = self.sigma_data
        w = (sigma * sigma + sd * sd) / ((sigma * sd) ** 2 + 1e-12)
        loss = (w.view(-1, 1, 1, 1) * (x_hat - target) ** 2).mean()
        return loss
