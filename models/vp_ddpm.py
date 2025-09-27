# diffusion/vp_ddpm.py
import torch, math, torch.nn.functional as F
from typing import Set
from .base import DiffusionBase, fourier_embed

def _cosine_alpha_bar(tau: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    f = lambda u: torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    return torch.clamp(f(tau) / f(torch.zeros_like(tau)), 1e-8, 1.0)

class VPDiffuser(DiffusionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.T = int(cfg.sde.vp.T)
        self.cosine_s = float(cfg.sde.vp.cosine_s)

    def allowed_samplers(self) -> Set[str]:
        return {"ddim", "iddpm"}  # VP-compatible samplers

    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t = torch.randint(1, self.T + 1, (batch_size,), device=device)
        tau = t.to(torch.float32) / float(self.T)
        alpha_bar = _cosine_alpha_bar(tau, s=self.cosine_s)
        return torch.sqrt((1.0 - alpha_bar) / alpha_bar)

    def _denoise_target(self, cond_clean, target_noisy, sigma):
        # For VP training we predict ε, so the "denoise_target" returns x0_hat used in samplers
        emb = self.emb_mlp(fourier_embed(0.25 * torch.log(torch.clamp(sigma, min=1e-12))))
        eps_hat = self.net(torch.cat([cond_clean, target_noisy], dim=1), emb)
        x0_hat = target_noisy - sigma.view(-1,1,1,1) * eps_hat
        return x0_hat

    def _compute_loss(self, cond, target):
        B, device = target.size(0), target.device
        sigma = self._sample_sigma(B, device)
        eps = torch.randn_like(target)
        x_noisy = target + sigma.view(-1,1,1,1) * eps
        emb = self.emb_mlp(fourier_embed(0.25 * torch.log(torch.clamp(sigma, min=1e-12))))
        eps_hat = self.net(torch.cat([cond, x_noisy], dim=1), emb)
        return F.mse_loss(eps_hat, eps)
