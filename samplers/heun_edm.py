# diffusion/samplers/heun_edm.py
import torch, math

def karras_sigma_schedule(steps, sigma_min, sigma_max, rho):
    i = torch.linspace(0, steps-1, steps)
    inv = 1.0/rho
    s = sigma_max**inv + (sigma_min**inv - sigma_max**inv)*i/max(steps-1,1)
    return s**rho

@torch.no_grad()
def heun_sample(module, cond, target_shape, device, cfg):
    """
    module: DiffusionLightning
    cond: (B, 2*C or 2*Clat, H, W)  -- clean conditioning channels
    target_shape: (B, K*C or K*Clat, H, W)
    """
    B, C_t, H, W = target_shape
    x = torch.randn(B, C_t, H, W, device=device) * cfg.edm.sigma_max
    sigmas = karras_sigma_schedule(cfg.steps, cfg.edm.sigma_min, cfg.edm.sigma_max, cfg.edm.rho).to(device)
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
    for i in range(cfg.steps):
        s_i, s_j = sigmas[i], sigmas[i+1]
        sigma_i = torch.full((B,), s_i, device=device)
        d_i = (module._denoise_target(cond, x, sigma_i) - x) / (s_i + 1e-12)
        x_e = x + (s_j - s_i) * d_i
        if s_j == 0: x = x_e; break
        d_j = (module._denoise_target(cond, x_e, torch.full((B,), s_j, device=device)) - x_e) / (s_j + 1e-12)
        x = x + (s_j - s_i) * 0.5*(d_i + d_j)
    return x.clamp(-1,1)
