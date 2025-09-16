from __future__ import annotations
from typing import Optional, Tuple, List, Any

import io
import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

# Optional: MLflow figure/file logging
try:
    import mlflow
except Exception:
    mlflow = None


# -------------------------
# small helpers
# -------------------------

def exists(x): return x is not None

def default(val, dflt):
    return val if exists(val) else (dflt() if callable(dflt) else dflt)

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal time embedding (for noise level σ or any continuous scalar).
    t: (B,) tensor
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / max(half - 1, 1)))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def make_coord_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return (2, H, W) normalized coords in [-1,1]."""
    ys = torch.linspace(-1., 1., steps=h, device=device)
    xs = torch.linspace(-1., 1., steps=w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


# -------------------------
# very compact U-Net with FiLM from time/noise embedding
# -------------------------

class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, c_out * 2)  # gamma, beta for FiLM
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        gamma, beta = self.emb(emb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(self.act(h))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.pool = nn.Conv2d(c, c, 3, stride=2, padding=1)
    def forward(self, x): return self.pool(x)

class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class UNet2D(nn.Module):
    """
    Tiny U-Net with FiLM from time/noise embedding.
    in_ch: channels of [noisy_target || conditioning || optional extras]
    out_ch: channels of target only (predict noise on targets)
    """
    def __init__(self, in_ch: int, out_ch: int, base: int = 64, emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # Enc
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, base, emb_dim)
        self.down1 = Down(base)           # /2
        self.rb2 = ResBlock(base, base*2, emb_dim)
        self.down2 = Down(base*2)         # /4
        self.rb3 = ResBlock(base*2, base*4, emb_dim)

        # Dec
        self.up2 = Up(base*4, base*2)
        self.rb4 = ResBlock(base*4, base*2, emb_dim)
        self.up1 = Up(base*2, base)
        self.rb5 = ResBlock(base*2, base, emb_dim)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        
        temb = self.time_mlp(t_emb)  # (B, emb_dim)
        h0 = self.in_conv(x)
        h1 = self.rb1(h0, temb)
        h2 = self.rb2(self.down1(h1), temb)
        h3 = self.rb3(self.down2(h2), temb)

        u2 = self.up2(h3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb4(u2, temb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb5(u1, temb)

        return self.out(u1)


# -------------------------
# EDM-style Lightning module for interpolation
# -------------------------

class EDMInterpolator(L.LightningModule):
    """
    Conditional diffusion (EDM-style) that learns to map:
      endpoints (first+last)  --->  internal lead-times

    Batch from your DataModule:
      x_cond  = endpoints channels  (B, Cx, H, W)
      y_clean = internal channels   (B, Cy, H, W)

    We add noise to y_clean only, then predict noise ε on y.
    """

    def __init__(
        self,
        # channels
        cond_channels: int,
        target_channels: int,
        extra_coord_channels: bool = False,             # add (x,y) coords as two channels
        extra_phys_time_scalar: Optional[float] = None, # add a constant scalar channel

        # U-Net
        unet_base: int = 64,
        time_embed_dim: int = 256,

        # EDM noise / training
        sigma_data: float = 0.5,      # (kept for completeness; not used directly here)
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        p_mean: float = -1.2,         # log-normal sampling of sigma
        p_std: float = 1.2,
        loss_weighting: str = "edm",  # 'edm' (sigma^2) or 'none'

        # optimization
        lr: float = 3e-4,             # fallback LR if no optimizer_cfg is given
        optimizer_cfg: Optional[dict] = None,
        scheduler_cfg: Optional[dict] = None,

        # sampling
        sample_steps: int = 20,       # number of EDM steps for inference
        rho: float = 7.0,             # EDM schedule exponent
        sample_every_val: int = 1,    # generate+log samples every N val epochs
        sample_save_npz: bool = False,

        # figure logging
        figures: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer_cfg", "scheduler_cfg", "figures"])
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg
        self.fig_cfg = figures or {}

        self.cond_channels = int(cond_channels)
        self.target_channels = int(target_channels)
        self.fallback_lr = float(lr)

        # extras
        self.add_coords = bool(extra_coord_channels)
        self.phys_time_scalar = extra_phys_time_scalar

        # build UNet
        in_ch = self.target_channels + self.cond_channels
        if self.add_coords:
            in_ch += 2
        if exists(self.phys_time_scalar):
            in_ch += 1

        self.unet = UNet2D(in_ch=in_ch, out_ch=self.target_channels,
                           base=unet_base, emb_dim=time_embed_dim)

        # cache for val sampling
        self._val_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # ------------- EDM core utilities -------------

    @torch.no_grad()
    def get_sigma_schedule(self, steps: int, rho: float) -> torch.Tensor:
        """Karras scheduler (EDM). Returns (steps,) sigmas decreasing from sigma_max -> sigma_min."""
        s0, s1 = self.hparams.sigma_max, self.hparams.sigma_min
        ramp = torch.linspace(0, 1, steps, device=self.device)
        sigmas = (s0 ** (1 / rho) + ramp * (s1 ** (1 / rho) - s0 ** (1 / rho))) ** rho
        return torch.flip(sigmas, dims=[0])  # descending

    def sample_sigmas_train(self, b: int) -> torch.Tensor:
        """Draw σ ~ LogNormal(p_mean, p_std), clamp to [sigma_min, sigma_max]."""
        p_mean, p_std = self.hparams.p_mean, self.hparams.p_std
        sigma = torch.exp(torch.randn(b, device=self.device) * p_std + p_mean)
        sigma = sigma.clamp(self.hparams.sigma_min, self.hparams.sigma_max)
        return sigma

    def add_extras(self, x_like: torch.Tensor) -> torch.Tensor:
        """
        Build optional extra channels (coords, phys_time_scalar) to concat with inputs.
        Returns (B, C_extra, H, W) or zeros if none.
        """
        B, _, H, W = x_like.shape
        extras: List[torch.Tensor] = []
        if self.add_coords:
            grid = make_coord_grid(H, W, x_like.device)              # (2,H,W)
            extras.append(grid.unsqueeze(0).expand(B, -1, -1, -1))   # (B,2,H,W)
        if exists(self.phys_time_scalar):
            tchan = torch.full((B, 1, H, W), float(self.phys_time_scalar), device=x_like.device)
            extras.append(tchan)
        if not extras:
            return torch.zeros((B, 0, H, W), device=x_like.device, dtype=x_like.dtype)
        return torch.cat(extras, dim=1)

    # ------------- forward = one denoising pass -------------
    def forward(self, y_noisy: torch.Tensor, x_cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        y_noisy: (B, Cy, H, W), x_cond: (B, Cx, H, W), sigma: (B,)
        returns: eps_hat (B, Cy, H, W)
        """
        # time embedding uses log sigma for scale invariance
        # use the first layer's input dim from the UNet time MLP
        time_embed_dim = self.unet.time_mlp[0].in_features
        t_emb = sinusoidal_embedding(sigma.log(), time_embed_dim)  # (B, time_embed_dim)

        extras = self.add_extras(y_noisy)                          # (B, Cextra, H, W)
        inp = torch.cat([y_noisy, x_cond, extras], dim=1)          # concat along channels
        return self.unet(inp, t_emb)

    # ------------- training / validation -------------
    def _shared_step(self, batch: Any, stage: str):
        """
        batch = (x_cond, y_clean)
          x_cond: endpoints (B, Cx, H, W)
          y_clean: internals (B, Cy, H, W)
        """
        x_cond, y_clean = batch
        B = y_clean.shape[0]

        sigma = self.sample_sigmas_train(B)                             # (B,)
        noise = torch.randn_like(y_clean)
        y_noisy = y_clean + noise * sigma[:, None, None, None]          # add EDM noise

        # predict noise
        eps_hat = self(y_noisy, x_cond, sigma)

        # loss (EDM weighting ~ sigma^2; set 'none' for plain MSE)
        if self.hparams.loss_weighting == "edm":
            w = (sigma ** 2)[:, None, None, None]
            loss = F.mse_loss(eps_hat * w.sqrt(), noise * w.sqrt(), reduction="mean")
        else:
            loss = F.mse_loss(eps_hat, noise, reduction="mean")

        # DDP-safe log
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=True)

        # cache small batch for sampling viz at epoch end (validation only)
        if stage == "val" and len(self._val_cache) < 1:
            self._val_cache.append((x_cond.detach(), y_clean.detach()))

        return loss

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        self._shared_step(batch, "val")

    # ------------- sampling (EDM steps, simple Euler) -------------
    @torch.no_grad()
    def sample_from_cond(self, x_cond: torch.Tensor, shape_target: Tuple[int, int, int], steps: Optional[int] = None) -> torch.Tensor:
        """
        Given endpoints x_cond (B, Cx, H, W), generate internals (B, Cy, H, W)
        using an EDM sigma schedule and Euler updates:
            x0_hat = y - sigma * eps_hat
            y <- x0_hat + sigma_next * eps_hat
        """
        B, H, W = x_cond.shape[0], x_cond.shape[2], x_cond.shape[3]
        Cy = shape_target[0]
        steps = default(steps, self.hparams.sample_steps)

        sigmas = self.get_sigma_schedule(steps=steps, rho=self.hparams.rho)  # (S,)
        y = torch.randn((B, Cy, H, W), device=x_cond.device) * sigmas[0]     # init at highest sigma

        for i, sigma in enumerate(sigmas):
            sigma_b = torch.full((B,), float(sigma), device=x_cond.device)
            eps = self(y, x_cond, sigma_b)
            x0_hat = y - sigma * eps
            if i == len(sigmas) - 1:
                y = x0_hat
            else:
                y = x0_hat + sigmas[i + 1] * eps
        return y

    # ------------- end-of-epoch logging -------------

    def on_validation_epoch_end(self) -> None:
        print("Validation Epoch ended")
        # run every N epochs on rank 0
        if (self.current_epoch + 1) % int(self.hparams.sample_every_val or 1) != 0:
            self._val_cache.clear()
            return
        if not self.trainer.is_global_zero:
            self._val_cache.clear()
            return
        if not self._val_cache:
            return

        x_cond, y_gt = self._val_cache[0]
        x_cond = x_cond.to(self.device)
        y_gt = y_gt.to(self.device)
        y_pred = self.sample_from_cond(x_cond, shape_target=y_gt.shape[1:4])

        # --- build figure grid (first few channels) ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            warnings.warn(f"Matplotlib not available for logging: {e}")
            self._val_cache.clear()
            return

        def _grid(t: torch.Tensor, title: str, max_ch=3):
            t = t.detach().float().cpu()
            b, c, h, w = t.shape
            ch = min(c, max_ch)
            fig, axes = plt.subplots(1, ch, figsize=(ch * 2.6, 2.6))
            if ch == 1:
                axes = [axes]
            for i in range(ch):
                axes[i].imshow(t[0, i].numpy(), cmap="viridis")
                axes[i].axis("off")
                axes[i].set_title(f"{title} ch#{i}", fontsize=8)
            fig.tight_layout()
            return fig

        figs = [
            ("figures/val_cond_endpoints.png", _grid(x_cond, "cond endpoints")),
            ("figures/val_pred_internals.png", _grid(y_pred, "pred internals")),
            ("figures/val_gt_internals.png", _grid(y_gt, "gt internals")),
        ]

        # Log to MLflow if configured via Lightning logger
        from lightning.pytorch.loggers import MLFlowLogger
        if isinstance(self.logger, MLFlowLogger) and mlflow is not None:
            run_id = self.logger.run_id
            try:
                with mlflow.start_run(run_id=run_id):
                    for path, fig in figs:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        mlflow.log_figure(fig, path)
                        import matplotlib.pyplot as plt
                        plt.close(fig)

                    if self.hparams.sample_save_npz:
                        import numpy as np
                        npz_bytes = io.BytesIO()
                        np.savez_compressed(
                            npz_bytes,
                            cond=x_cond[0].detach().cpu().numpy(),
                            pred=y_pred[0].detach().cpu().numpy(),
                            gt=y_gt[0].detach().cpu().numpy(),
                        )
                        npz_bytes.seek(0)
                        # write BytesIO to a temp file and log
                        tmp_path = self._bytes_to_tempfile(npz_bytes, f"samples_epoch_{self.current_epoch:04d}.npz")
                        mlflow.log_artifact(tmp_path, artifact_path="samples")
            except Exception as e:
                warnings.warn(f"MLflow logging failed: {e}")

        self._val_cache.clear()

    def _bytes_to_tempfile(self, b: io.BytesIO, name: str) -> str:
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as f:
            f.write(b.read())
        return path

    # ------------- optimizers -------------
    def configure_optimizers(self):
        """Hydra-friendly optimizer creation with a safe fallback."""
        try:
            from hydra.utils import instantiate
            if isinstance(self.optimizer_cfg, dict) and "_target_" in self.optimizer_cfg:
                opt = instantiate(self.optimizer_cfg, params=self.parameters())
            else:
                raise ValueError("no hydra optimizer target")
        except Exception:
            # Safe fallback
            opt = torch.optim.Adam(self.parameters(), lr=self.fallback_lr)

        if self.scheduler_cfg:
            try:
                from hydra.utils import instantiate as _inst
                sch = _inst(self.scheduler_cfg, optimizer=opt)
                return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
            except Exception:
                return opt
        return opt
