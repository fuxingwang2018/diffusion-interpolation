# samplers/heun_edm.py
import math
import torch
from .base import SamplerBase

def _karras_sigma_schedule(steps: int, sigma_min: float, sigma_max: float, rho: float, device) -> torch.Tensor:
    """Karras et al. (EDM) noise schedule."""
    i = torch.linspace(0, steps - 1, steps, device=device, dtype=torch.float64)
    inv = 1.0 / rho
    sigmas = (sigma_max**inv + (sigma_min**inv - sigma_max**inv) * i / max(steps - 1, 1)).pow(rho)
    return sigmas  # float64 for stability; we’ll cast as needed

class HeunEDMSampler(SamplerBase):
    def __init__(
        self,
        steps: int = 40,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
    ):
        self.steps     = int(steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho       = float(rho)
        self.S_churn   = float(S_churn)
        self.S_min     = float(S_min)
        self.S_max     = float(S_max)
        self.S_noise   = float(S_noise)

    @property
    def name(self) -> str:
        return "heun_edm"

    @torch.no_grad()
    def sample(self, module, cond, target_shape, device, **kwargs):
        """
        module: your LightningModule (EDM variant) exposing _denoise_target(cond, x, sigma) -> x0_hat
        cond:   (B, cond_ch, H, W)
        target_shape: (B, C_t, H, W)
        """
        B, C_t, H, W = target_shape
        dtype = cond.dtype

        # Respect model sigma bounds/rounding if present (as in EDM code).
        net = getattr(module, "network", module)  # try to find the underlying net if wrapped
        sigma_min = max(self.sigma_min, float(getattr(net, "sigma_min", self.sigma_min)))
        sigma_max = min(self.sigma_max, float(getattr(net, "sigma_max", self.sigma_max)))
        round_sigma = getattr(net, "round_sigma", lambda s: s)  # identity if not provided

        # Karras schedule + terminal 0
        sigmas = _karras_sigma_schedule(self.steps, sigma_min, sigma_max, self.rho, device=device)  # float64
        sigmas = round_sigma(sigmas).to(dtype)  # optional rounding, then cast to model dtype
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # append 0

        # Init state: x ~ N(0, I) scaled by first sigma
        x = torch.randn_like(cond.new_empty(B, C_t, H, W)) * sigmas[0]

        # Main loop (Heun with optional churn)
        for i in range(self.steps):
            s_i = sigmas[i]     # current sigma (scalar tensor)
            s_j = sigmas[i + 1] # next sigma
            # Per-batch sigma tensors
            sigma_i = s_i.expand(B)
            sigma_j = s_j.expand(B)

            # ---- Churn (stochasticity boost) ----
            # gamma ∈ [0, sqrt(2)-1] but only when s_i within [S_min, S_max]
            if self.S_min <= float(s_i) <= self.S_max and self.S_churn > 0:
                gamma = min(self.S_churn / self.steps, math.sqrt(2) - 1.0)
            else:
                gamma = 0.0

            if gamma > 0.0:
                t_hat = round_sigma(s_i * (1.0 + gamma)).to(dtype)
                # add extra noise with std = sqrt(t_hat^2 - s_i^2) * S_noise
                noise = torch.randn_like(x)
                x_hat = x + (t_hat.square() - s_i.square()).clamp(min=0).sqrt() * self.S_noise * noise
            else:
                t_hat = s_i
                x_hat = x

            # ---- Euler step ----
            # EDM uses d = (x - x0_hat) / sigma   (note the sign!)
            x0_hat = module._denoise_target(cond, x_hat, t_hat.expand(B))
            d_i = (x_hat - x0_hat) / (t_hat + 1e-12)  # (B, C, H, W)
            x_e = x_hat + (s_j - t_hat) * d_i

            # ---- Heun correction (2nd order) ----
            if s_j == 0:
                x = x_e
                break
            x0_hat_j = module._denoise_target(cond, x_e, sigma_j)
            d_j = (x_e - x0_hat_j) / (s_j + 1e-12)
            x = x_hat + (s_j - t_hat) * 0.5 * (d_i + d_j)

        return x #.clamp(-1, 1)
