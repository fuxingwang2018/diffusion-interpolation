# diffusion/samplers/ddim.py
import torch, math

def make_alpha_bar(T, s=0.008, device="cpu"):
    t = torch.arange(0, T+1, device=device)
    f = lambda u: torch.cos((u + s)/(1+s) * math.pi/2)**2
    ab = f(t/T) / f(torch.tensor(0.0, device=device))
    return ab.clamp(1e-8, 1.0)

@torch.no_grad()
def ddim_sample(module, cond, target_shape, device, cfg):
    B, C_t, H, W = target_shape
    T = cfg.ddim.T
    eta = cfg.ddim.eta
    ab = make_alpha_bar(T, s=cfg.ddim.cosine_s, device=device)
    x_t = torch.randn(B, C_t, H, W, device=device)
    for t in range(T, 0, -1):
        alpha_bar_t = ab[t]; alpha_bar_prev = ab[t-1]
        sigma_t = torch.sqrt((1 - alpha_bar_t)/alpha_bar_t)
        sigma = torch.full((B,), sigma_t.item(), device=device)
        x0_hat = module._denoise_target(cond, x_t, sigma)
        eps_hat = (x_t - x0_hat * alpha_bar_t.sqrt()) / (1 - alpha_bar_t).sqrt()
        dir_xt = (1 - alpha_bar_prev).sqrt() * eps_hat
        x0_scale = alpha_bar_prev.sqrt() * x0_hat
        if eta == 0:
            x_t = x0_scale + dir_xt
        else:
            sigma_ddim = eta * ((1 - alpha_bar_prev)/(1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))**0.5
            z = torch.randn_like(x_t)
            x_t = x0_scale + dir_xt + sigma_ddim * z
    return x_t.clamp(-1,1)
