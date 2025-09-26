# samplers/ddim.py
import torch, math

def _make_alpha_bar(T, s=0.008, device="cpu"):
    t = torch.arange(0, T+1, device=device)
    f = lambda u: torch.cos((u + s)/(1+s) * math.pi/2)**2
    ab = f(t/T) / f(torch.tensor(0.0, device=device))
    return ab.clamp(1e-8, 1.0)

class DDIMSampler:
    def __init__(self, T: int = 50, eta: float = 0.0, cosine_s: float = 0.008):
        self.T = T
        self.eta = eta
        self.cosine_s = cosine_s

    @torch.no_grad()
    def sample(self, module, cond, target_shape, device):
        B, C_t, H, W = target_shape
        ab = _make_alpha_bar(self.T, s=self.cosine_s, device=device)
        x_t = torch.randn(B, C_t, H, W, device=device)
        for t in range(self.T, 0, -1):
            alpha_bar_t = ab[t]; alpha_bar_prev = ab[t-1]
            sigma_t = torch.sqrt((1 - alpha_bar_t)/alpha_bar_t)
            sigma = torch.full((B,), sigma_t.item(), device=device)
            x0_hat = module._denoise_target(cond, x_t, sigma)
            eps_hat = (x_t - x0_hat * alpha_bar_t.sqrt()) / (1 - alpha_bar_t).sqrt()
            dir_xt = (1 - alpha_bar_prev).sqrt() * eps_hat
            x0_scale = alpha_bar_prev.sqrt() * x0_hat
            if self.eta == 0.0:
                x_t = x0_scale + dir_xt
            else:
                sigma_ddim = self.eta * ((1 - alpha_bar_prev)/(1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev))**0.5
                z = torch.randn_like(x_t)
                x_t = x0_scale + dir_xt + sigma_ddim * z
        return x_t.clamp(-1, 1)
