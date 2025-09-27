# samplers/heun_edm.py
import torch
from .base import SamplerBase

def _karras_sigma_schedule(steps, sigma_min, sigma_max, rho, device):
    i = torch.linspace(0, steps-1, steps, device=device)
    inv = 1.0 / rho
    s = sigma_max**inv + (sigma_min**inv - sigma_max**inv) * i / max(steps-1, 1)
    return s**rho

class HeunEDMSampler(SamplerBase):
    def __init__(self, steps: int = 40, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0):
        self.steps, self.sigma_min, self.sigma_max, self.rho = int(steps), float(sigma_min), float(sigma_max), float(rho)

    @property
    def name(self) -> str:
        return "heun_edm"

    @torch.no_grad()
    def sample(self, module, cond, target_shape, device, **kwargs):
        B, C_t, H, W = target_shape
        x = torch.randn(B, C_t, H, W, device=device) * self.sigma_max
        sigmas = _karras_sigma_schedule(self.steps, self.sigma_min, self.sigma_max, self.rho, device)
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
        for i in range(self.steps):
            s_i, s_j = sigmas[i], sigmas[i+1]
            sigma_i = torch.full((B,), s_i, device=device)
            d_i = (module._denoise_target(cond, x, sigma_i) - x) / (s_i + 1e-12)
            x_e = x + (s_j - s_i) * d_i
            if s_j == 0:
                x = x_e; break
            d_j = (module._denoise_target(cond, x_e, torch.full((B,), s_j, device=device)) - x_e) / (s_j + 1e-12)
            x = x + (s_j - s_i) * 0.5 * (d_i + d_j)
        return x.clamp(-1, 1)
