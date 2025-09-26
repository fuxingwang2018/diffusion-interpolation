# diffusion/diffusion_module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple
from omegaconf import DictConfig

from networks.unet import UNet2D
from networks.autoencoder import OptionalVAE
from samplers.heun_edm import heun_sample
from samplers.ddim import ddim_sample
from samplers.iddpm import iddpm_sample

# ---------- EDM preconditioning ----------
def edm_precond(sigma: torch.Tensor, sigma_data: float):
    s2, sd2 = sigma**2, sigma_data**2
    c_skip = sd2 / (s2 + sd2)
    c_out  = sigma * math.sqrt(sd2) / torch.sqrt(s2 + sd2)
    c_in   = 1.0 / torch.sqrt(s2 + sd2)
    c_noise = 0.25 * torch.log(sigma.clamp_min(1e-12))
    return c_skip, c_out, c_in, c_noise

def fourier_embed(x: torch.Tensor, dim=64):
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=x.device, dtype=x.dtype))
    x = x.view(-1,1)*freqs.view(1,-1)
    emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    if dim % 2: emb = F.pad(emb, (0,1))
    return emb

# ---------- Lightning Module ----------
class DiffusionLightning(L.LightningModule):
    """
    Conditional diffusion (EDM/VP/VE) that:
      - concatenates cond (2*C) channels (clean) with target (K*C) channels (noised),
      - denoises ONLY the target channels,
      - supports latent diffusion via optional frozen VAE.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.vae = OptionalVAE(cfg.latent)   # encodes/decodes cond & target if enabled

        # Channel math
        self.C = cfg.data.C
        self.T = cfg.data.T
        self.cond_ch   = 2 * self.C
        self.target_ch = (self.T - 2) * self.C
        in_model_ch = self.cond_ch + self.target_ch
        out_model_ch = self.target_ch

        if self.vae.enabled:
            # In latent mode, we encode cond and target separately then pack;
            # the per-stream channels change to cfg.latent.latent_channels per frame.
            self.cond_ch   = 2 * self.vae.latent_channels
            self.target_ch = (self.T - 2) * self.vae.latent_channels
            in_model_ch  = self.cond_ch + self.target_ch
            out_model_ch = self.target_ch

        self.raw_net = UNet2D(
            in_ch=in_model_ch, out_ch=out_model_ch,
            base_ch=cfg.model.base_ch,
            ch_mults=tuple(cfg.model.ch_mults),
            num_res_blocks=cfg.model.num_res_blocks,
            emb_dim=cfg.model.emb_dim,
            attn_resolutions=tuple(cfg.model.attn_resolutions),
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            use_gn=cfg.model.use_gn,
        )
        self.sigma_data = cfg.loss.sigma_data
        self.embed = nn.Sequential(nn.Linear(64, cfg.model.emb_dim), nn.SiLU(),
                                   nn.Linear(cfg.model.emb_dim, cfg.model.emb_dim))

    # --------- Noise level selection (EDM / VP / VE) ---------
    def _sample_sigma(self, B, device):
        sde = self.cfg.sde.name
        if sde in ("edm", "ve"):
            Pm, Ps = self.cfg.loss.P_mean, self.cfg.loss.P_std
            return torch.exp(torch.randn(B, device=device) * Ps + Pm)  # σ
        elif sde == "vp":
            T = self.cfg.sde.vp.T
            t = torch.randint(low=1, high=T+1, size=(B,), device=device)
            s = self.cfg.sde.vp.cosine_s
            f = lambda u: torch.cos((u + s)/(1+s) * math.pi/2)**2
            alpha_bar = (f(t/T)/f(torch.tensor(0.0, device=device))).clamp(1e-8, 1.0)
            sigma = torch.sqrt((1.0 - alpha_bar)/alpha_bar)
            return sigma
        else:
            raise ValueError("sde.name ∈ {edm, vp, ve}")

    # --------- Pack (x,y) into cond/target tensors ---------
    def _pack_xy(self, x, y):
        # x: (B, 2, C, H, W) or (2, C, H, W) → ensure batch
        if x.dim() == 4:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        B, two, C, H, W = x.shape
        _, K, C2, H2, W2 = y.shape
        assert two == 2 and C2 == C and H2 == H and W2 == W
        cond = x.reshape(B, 2*C, H, W)          # (B, 2C, H, W)
        target = y.reshape(B, K*C, H, W)        # (B, (T-2)C, H, W)
        return cond, target

    # --------- Optional VAE encode/decode ---------
    def _maybe_encode(self, cond, target):
        if not self.vae.enabled: return cond, target
        with torch.no_grad() if self.vae.frozen else torch.enable_grad():
            cond_z = self.vae.encode_cond(cond)       # (B, 2*Clat, H', W')
            targ_z = self.vae.encode_target(target)   # (B, K*Clat, H', W')
        return cond_z, targ_z

    def _maybe_decode_target(self, target):
        if not self.vae.enabled: return target
        with torch.no_grad():
            return self.vae.decode_target(target)     # (B, K*C, H, W) in [-1,1]

    # --------- Denoiser: target-only EDM preconditioning, cond as clean input ---------
    def _denoise_target(self, cond_clean, target_noisy, sigma):
        # sigma: (B,)
        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
        emb = self.embed(fourier_embed(c_noise))
        # scale only the target part
        target_in = c_in.view(-1,1,1,1) * target_noisy
        net_in = torch.cat([cond_clean, target_in], dim=1)
        fx = self.raw_net(net_in, emb)  # predicts clean target residual
        x_hat = c_skip.view(-1,1,1,1) * target_noisy + c_out.view(-1,1,1,1) * fx
        return x_hat  # same shape as target channels

    # --------- Loss ---------
    def _compute_loss(self, cond, target):
        B = target.size(0); device = target.device
        sigma = self._sample_sigma(B, device)
        noise = torch.randn_like(target) * sigma.view(-1,1,1,1)
        target_noisy = target + noise

        pred = self._denoise_target(cond, target_noisy, sigma)

        mode = self.cfg.loss.target  # 'x'|'eps'|'v'|'score'
        if mode == "x":
            tgt = target
        elif mode == "eps":
            tgt = noise / (sigma.view(-1,1,1,1)+1e-12)
            pred = (target_noisy - pred) / (sigma.view(-1,1,1,1)+1e-12)
        elif mode == "v":
            alpha = 1.0 / torch.sqrt(1.0 + sigma**2)
            alpha = alpha.view(-1,1,1,1); sigma_ = sigma.view(-1,1,1,1)
            x0_hat = pred
            eps_hat = (target_noisy - x0_hat) / (sigma_+1e-12)
            pred = alpha * eps_hat + sigma_ * x0_hat
            x0 = target
            eps = noise / (sigma_+1e-12)
            tgt = alpha * eps + sigma_ * x0
        elif mode == "score":
            tgt = (target - target_noisy) / (sigma.view(-1,1,1,1)**2 + 1e-12)
            pred = (pred   - target_noisy) / (sigma.view(-1,1,1,1)**2 + 1e-12)
        else:
            raise ValueError("loss.target ∈ {x, eps, v, score}")

        if self.cfg.loss.weight == "edm":
            w = (sigma**2 + self.sigma_data**2) / ((sigma*self.sigma_data)**2 + 1e-12)
            w = w.view(-1,1,1,1)
        else:
            w = 1.0

        return (w*(pred - tgt)**2).mean()

    # --------- Lightning hooks ---------
    def training_step(self, batch, _):
        x, y = batch  # x: (B,2,C,H,W), y: (B,T-2,C,H,W)
        cond, target = self._pack_xy(x, y)
        cond, target = self._maybe_encode(cond, target)
        loss = self._compute_loss(cond, target)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        cond, target = self._pack_xy(x, y)
        cond, target = self._maybe_encode(cond, target)
        loss = self._compute_loss(cond, target)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.lr,
                                betas=tuple(self.cfg.optim.betas), weight_decay=self.cfg.optim.weight_decay)
        return opt

# --------- EMA fallback ---------
class SimpleEMAFallback(L.Callback):
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay=decay; self.shadow=None
    def on_train_start(self, trainer, pl_module):
        self.shadow = {k: v.detach().clone() for k,v in pl_module.state_dict().items()}
    def on_after_backward(self, trainer, pl_module):
        with torch.no_grad():
            for k,v in pl_module.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def on_validation_start(self, trainer, pl_module):
        self._backup = {k: v.detach().clone() for k,v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.shadow, strict=False)
    def on_validation_end(self, trainer, pl_module):
        pl_module.load_state_dict(self._backup, strict=False)

# --------- Sampling callback (conditioned generation of target frames) ---------
class SamplerCallback(L.Callback):
    def __init__(self, every_n_epochs=1, n=4, sampler_cfg=None, out_key="samples"):
        super().__init__()
        self.every = every_n_epochs; self.n = n; self.sampler_cfg = sampler_cfg; self.key = out_key

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch+1) % self.every != 0: return
        B = self.n
        C = pl_module.C if not pl_module.vae.enabled else pl_module.vae.latent_channels
        H = W = pl_module.cfg.samples.size
        # Build random conditioning (demo); in your use, take real x
        cond = torch.randn(B, 2*C, H, W, device=pl_module.device) * 0.0  # zeros demo
        name = pl_module.cfg.sampler.name
        if name == "heun":
            targ = heun_sample(pl_module, cond, (B, pl_module.target_ch if not pl_module.vae.enabled else (pl_module.T-2)*C, H, W),
                               pl_module.device, pl_module.cfg.sampler)
        elif name == "ddim":
            targ = ddim_sample(pl_module, cond, (B, pl_module.target_ch if not pl_module.vae.enabled else (pl_module.T-2)*C, H, W),
                               pl_module.device, pl_module.cfg.sampler)
        elif name == "iddpm":
            targ = iddpm_sample(pl_module, cond, (B, pl_module.target_ch if not pl_module.vae.enabled else (pl_module.T-2)*C, H, W),
                                pl_module.device, pl_module.cfg.sampler)
        else:
            return

        if pl_module.vae.enabled:
            targ = pl_module._maybe_decode_target(targ)

        # For visualization, split first internal frame only (optional)
        try:
            import torchvision.utils as vutils
            # If K>1, visualize first C channels:
            K = (pl_module.T - 2)
            show = targ[:, :C, :, :].clamp(-1,1)
            grid = vutils.make_grid(show, nrow=int(self.n**0.5), normalize=True, value_range=(-1,1))
            if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_image"):
                trainer.logger.experiment.add_image(self.key, grid, global_step=trainer.global_step)
        except Exception:
            pass
