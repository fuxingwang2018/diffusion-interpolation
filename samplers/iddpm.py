# diffusion/samplers/iddpm.py
import torch, math

def betas_for_cosine(T, s=0.008, device="cpu"):
    t = torch.arange(0, T+1, device=device)
    f = lambda u: torch.cos((u + s)/(1+s) * math.pi/2)**2
    ab = f(t/T) / f(torch.tensor(0.0, device=device))
    betas = (1 - (ab[1:]/ab[:-1])).clamp(1e-8, 0.999)
    return betas

@torch.no_grad()
def iddpm_sample(module, cond, target_shape, device, cfg):
    B, C_t, H, W = target_shape
    T = cfg.iddpm.T
    betas = betas_for_cosine(T, s=cfg.iddpm.cosine_s, device=device)
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)
    x = torch.randn(B, C_t, H, W, device=device)
    for t in range(T-1, -1, -1):
        alpha_bar_t = ab[t]
        sigma_t = torch.sqrt((1 - alpha_bar_t)/alpha_bar_t)
        sigma = torch.full((B,), sigma_t.item(), device=device)
        x0_hat = module._denoise_target(cond, x, sigma)
        eps_hat = (x - alpha_bar_t.sqrt()*x0_hat) / (1 - alpha_bar_t).sqrt()
        if t > 0:
            beta_t = betas[t]
            alpha_t = alphas[t]
            var = beta_t * (1 - ab[t-1]) / (1 - ab[t])
            noise = torch.randn_like(x)
            mean = (1/alpha_t.sqrt())*(x - beta_t/((1 - alpha_bar_t).sqrt())*eps_hat)
            x = mean + var.sqrt() * noise
        else:
            x = x0_hat
    return x.clamp(-1,1)
