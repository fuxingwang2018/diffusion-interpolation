# diffusion/diffusion_module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

from networks.unet import UNet2D
from networks.autoencoder import OptionalVAE

### Classes


# configs/diffusion_config.py
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional

# -----------------------------
# Latent Autoencoder Config
# -----------------------------
@dataclass
class LatentConfig:
    """Optional VAE-style encoder/decoder for latent diffusion."""
    enable: bool = False          # Whether to use latent autoencoding
    frozen: bool = True           # Freeze encoder/decoder (no training)
    latent_channels: int = 4      # Channels in latent space
    in_channels: int = 3          # Input channels (for first conv in encoder)
    cond_channels: int = 6        # Channels for cond frames (2 * C)
    target_channels: int = 12     # Channels for target frames ((T-2) * C)

# -----------------------------
# Model (UNet backbone) Config
# -----------------------------
@dataclass
class ModelConfig:
    """UNet backbone hyperparameters."""
    base_ch: int = 128                  # Base number of channels
    ch_mults: Tuple[int, ...] = (1, 2, 2, 2)  # Multipliers for each UNet level
    num_res_blocks: int = 2             # Number of ResBlocks per level
    emb_dim: int = 256                  # Embedding dimension for timestep/σ embedding
    attn_resolutions: Tuple[int, ...] = (8, 16)  # Resolutions (downsample factors) where attention is used
    num_heads: int = 4                  # Number of attention heads
    dropout: float = 0.0                # Dropout rate
    use_gn: bool = True                 # Use GroupNorm instead of BatchNorm

# -----------------------------
# SDE / Noise Schedule Config
# -----------------------------
@dataclass
class SDEConfig:
    """Noise schedule configuration."""
    name: Literal["edm", "vp", "ve"] = "edm"  # Which family of SDE / noise schedule
    # VP-specific
    class VP:
        T: int = 1000
        cosine_s: float = 0.008
    vp: VP = field(default_factory=VP)
    # EDM-specific
    class EDM:
        sigma_min: float = 0.002
        sigma_max: float = 80.0
        rho: float = 7.0
    edm: EDM = field(default_factory=EDM)

# -----------------------------
# Loss Config
# -----------------------------
@dataclass
class LossConfig:
    """Loss and preconditioning options."""
    P_mean: float = -1.2              # Log-σ mean for EDM sampling
    P_std: float = 1.2                # Log-σ std for EDM sampling
    sigma_data: float = 0.5           # σ_data constant
    target: Literal["x", "eps", "v", "score"] = "x"  # Loss target type. "x" clean data
    weight: Literal["edm", "none"] = "edm"           # Loss weighting scheme

 

# -----------------------------
# Sampler Config
# -----------------------------
@dataclass
class SamplerConfig:
    """Sampler selection + parameters."""
    name: Literal["heun", "ddim", "iddpm"] = "heun"
    steps: int = 50                      # Number of denoising steps
    # Heun/EDM
    edm: SDEConfig.EDM = field(default_factory=SDEConfig.EDM)
    # DDIM
    class DDIM:
        T: int = 50
        eta: float = 0.0
        cosine_s: float = 0.008
    ddim: DDIM = field(default_factory=DDIM)
    # iDDPM
    class IDDPM:
        T: int = 1000
        cosine_s: float = 0.008
    iddpm: IDDPM = field(default_factory=IDDPM)

# -----------------------------
# Data Config
# -----------------------------
@dataclass
class DataConfig:
    """Shape/config info for dataset items (used to size model)."""
    C: int = 3      # Channels per frame
    T: int = 4      # Number of timesteps in sequence

# -----------------------------
# Samples (logging/vis) Config
# -----------------------------
@dataclass
class SamplesConfig:
    """Settings for sample callback during validation."""
    size: int = 256  # Resolution (H=W=size) for random sample grid

# -----------------------------
# Full Diffusion Module Config
# -----------------------------
@dataclass
class DiffusionConfig:
    """Top-level config for DiffusionLightning."""
    seed: int = 42
    latent: LatentConfig = field(default_factory=LatentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer_cfg: Optional[dict] = None,
    scheduler_cfg: Optional[dict] = None,    
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    samples: SamplesConfig = field(default_factory=SamplesConfig)

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
class DiffusionModule(L.LightningModule):
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

        self.vae = OptionalVAE(cfg.latent)

        # Channel math
        self.C = cfg.data.C
        self.T = cfg.data.T
        self.cond_ch   = 2 * self.C
        self.target_ch = (self.T - 2) * self.C
        in_model_ch = self.cond_ch + self.target_ch
        out_model_ch = self.target_ch

        if self.vae.enabled:
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
            return torch.exp(torch.randn(B, device=device) * Ps + Pm)
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

    # --------- Pack (x,y) -> cond/target ---------
    def _pack_xy(self, x, y):
        if x.dim() == 4:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        B, two, C, H, W = x.shape
        _, K, C2, H2, W2 = y.shape
        assert two == 2 and C2 == C and H2 == H and W2 == W
        cond = x.reshape(B, 2*C, H, W)
        target = y.reshape(B, K*C, H, W)
        return cond, target

    # --------- Optional VAE encode/decode ---------
    def _maybe_encode(self, cond, target):
        if not self.vae.enabled: return cond, target
        with torch.no_grad() if self.vae.frozen else torch.enable_grad():
            cond_z = self.vae.encode_cond(cond)
            targ_z = self.vae.encode_target(target)
        return cond_z, targ_z

    def _maybe_decode_target(self, target):
        if not self.vae.enabled: return target
        with torch.no_grad():
            return self.vae.decode_target(target)

    # --------- Denoiser: target-only preconditioning ---------
    def _denoise_target(self, cond_clean, target_noisy, sigma):
        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
        emb = self.embed(fourier_embed(c_noise))
        target_in = c_in.view(-1,1,1,1) * target_noisy
        net_in = torch.cat([cond_clean, target_in], dim=1)
        fx = self.raw_net(net_in, emb)
        x_hat = c_skip.view(-1,1,1,1) * target_noisy + c_out.view(-1,1,1,1) * fx
        return x_hat

    # --------- Loss ---------
    def _compute_loss(self, cond, target):
        B = target.size(0); device = target.device
        sigma = self._sample_sigma(B, device)
        noise = torch.randn_like(target) * sigma.view(-1,1,1,1)
        target_noisy = target + noise

        pred = self._denoise_target(cond, target_noisy, sigma)

        mode = self.cfg.loss.target
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
        x, y = batch
        cond, target = self._pack_xy(x, y)
        cond, target = self._maybe_encode(cond, target)
        loss = self._compute_loss(cond, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        cond, target = self._pack_xy(x, y)
        cond, target = self._maybe_encode(cond, target)
        loss = self._compute_loss(cond, target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        from hydra.utils import instantiate
        if isinstance(self.cfg.optimizer_cfg, dict) and "_target_" in self.cfg.optimizer_cfg:
            opt = instantiate(self.cfg.optimizer_cfg, params=self.parameters())
        else:
            raise ValueError("no hydra optimizer target")
        if self.cfg.scheduler_cfg:
            try:
                sch = instantiate(self.cfg.scheduler_cfg, optimizer=opt)
                return {"optimizer": opt, "lr_scheduler": sch}
            except Exception:
                return opt
        return opt

# --------- Sampling callback: Hydra-instantiated sampler class ---------
class SamplerCallback(L.Callback):
    def __init__(self, every_n_epochs=1, n=4, sampler_cfg=None, out_key="samples"):
        super().__init__()
        self.every = every_n_epochs
        self.n = n
        self.sampler_cfg = sampler_cfg
        self.key = out_key
        self._sampler = None  # created lazily via Hydra

    def on_fit_start(self, trainer, pl_module):
        # Instantiate sampler class once, using cfg.sampler (must have _target_)
        if self.sampler_cfg is not None and "_target_" in self.sampler_cfg:
            self._sampler = instantiate(self.sampler_cfg)
        else:
            self._sampler = None

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.every != 0:
            return
        if self._sampler is None:
            return

        B = self.n
        C = pl_module.C if not pl_module.vae.enabled else pl_module.vae.latent_channels
        H = W = pl_module.cfg.samples.size

        # Build dummy zero cond just for visual sanity; users should pass real cond for eval
        cond = torch.zeros(B, 2*C, H, W, device=pl_module.device)

        target_ch = pl_module.target_ch if not pl_module.vae.enabled else (pl_module.T - 2) * C
        targ = self._sampler.sample(pl_module, cond, (B, target_ch, H, W), pl_module.device)

        if pl_module.vae.enabled:
            targ = pl_module._maybe_decode_target(targ)

        try:
            import torchvision.utils as vutils
            show = targ[:, :C, :, :].clamp(-1, 1)
            grid = vutils.make_grid(show, nrow=int(self.n**0.5), normalize=True, value_range=(-1,1))
            if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_image"):
                trainer.logger.experiment.add_image(self.key, grid, global_step=trainer.global_step)
        except Exception:
            pass
