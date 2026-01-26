import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple, Optional, Dict
# Rombach, Robin; Blattmann, Andreas; Lorenz, Dominik; Esser, Patrick; Ommer, Björn (CVPR 2022).
# "High-Resolution Image Synthesis with Latent Diffusion Models", This paper introduced the two-stage approach


# ==========================================
# Helper Modules (ResNets, Norms, etc.)
# ==========================================

class GroupNorm32(nn.GroupNorm):
    """Fixed GroupNorm with 32 groups (standard in diffusion/VAEs)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__(num_groups=32, num_channels=num_channels, eps=eps)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm32(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = Swish()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.conv2(self.dropout(self.activation(self.norm2(h))))
        return h + self.shortcut(x)

class Downsample(nn.Module):
    """Downsamples by factor of 2 using strided convolution."""
    def __init__(self, in_channels: int):
        super().__init__()
        # Kernel 3, stride 2, padding 1 = exact 2x downsample
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """Upsamples by factor of 2 using Nearest Neighbor + Conv."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

# ==========================================
# Core Components
# ==========================================

class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        z_channels: int, 
        base_channels: int = 64, 
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        blocks = []
        now_ch = base_channels
        
        # We need exactly 2 downsampling steps to go 256 -> 64 (f=4)
        # ch_mult usually dictates depth. For f=4, we process levels 0, 1, then bottleneck.
        
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(now_ch, out_ch))
                now_ch = out_ch
            
            # Add downsample if not the last level
            if i != len(ch_mult) - 1:
                blocks.append(Downsample(now_ch))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Middle / Bottleneck
        self.mid_block1 = ResnetBlock(now_ch, now_ch)
        self.mid_block2 = ResnetBlock(now_ch, now_ch)
        
        # Out
        self.norm_out = GroupNorm32(now_ch)
        self.conv_out = nn.Conv2d(now_ch, 2 * z_channels, kernel_size=3, padding=1) # 2*z for Mean + Logvar
        self.act = Swish()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, 
        out_channels: int, 
        z_channels: int, 
        base_channels: int = 64, 
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2
    ):
        super().__init__()
        
        # Reverse multiplier
        ch_mult = ch_mult[::-1]
        now_ch = base_channels * ch_mult[0]
        
        self.conv_in = nn.Conv2d(z_channels, now_ch, kernel_size=3, padding=1)
        
        self.mid_block1 = ResnetBlock(now_ch, now_ch)
        self.mid_block2 = ResnetBlock(now_ch, now_ch)
        
        blocks = []
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(now_ch, out_ch))
                now_ch = out_ch
            
            # Upsample if not the last level
            if i != len(ch_mult) - 1:
                blocks.append(Upsample(now_ch))
                
        self.blocks = nn.Sequential(*blocks)
        
        self.norm_out = GroupNorm32(now_ch)
        self.conv_out = nn.Conv2d(now_ch, out_channels, kernel_size=3, padding=1)
        self.act = Swish()

    def forward(self, z):
        z = self.conv_in(z)
        z = self.mid_block1(z)
        z = self.mid_block2(z)
        z = self.blocks(z)
        z = self.act(self.norm_out(z))
        z = self.conv_out(z)
        return z

# ==========================================
# Distribution Helper
# ==========================================

class DiagonalGaussianDistribution:
    """Helper to handle the reparameterization trick."""
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.])
        if other is None:
            # KL against standard normal N(0, 1)
            return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=[1, 2, 3])

# ==========================================
# Main VAE Lightning Module
# ==========================================

class AutoencoderKL(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        z_channels: int = 4,
        base_channels: int = 64,
        # To get 256->64, we need 2 downsamples. 
        # ch_mult=[1, 2, 4] with 3 levels implies 2 downsampling interfaces (1->2, 2->4).
        ch_mult: Tuple[int, ...] = (1, 2, 4), 
        kl_weight: float = 1e-6,
        optimizer_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.kl_weight = kl_weight
        self.optimizer_cfg = optimizer_cfg

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=z_channels,
            base_channels=base_channels,
            ch_mult=ch_mult
        )
        
        self.decoder = Decoder(
            out_channels=out_channels,
            z_channels=z_channels,
            base_channels=base_channels,
            ch_mult=ch_mult
        )
        
        # Often helpful to start with small weights for the "mean/logvar" projection
        with torch.no_grad():
            self.encoder.conv_out.weight.mul_(0.1)
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        x: (B, C, H, W) -> Returns distribution P(z|x)
        """
        h = self.encoder(x)
        moments = h
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
        
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, z_channels, H_lat, W_lat) -> Returns reconstruction
        """
        x_rec = self.decoder(z)
        return x_rec

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean
        rec = self.decode(z)
        return rec, posterior

    def training_step(self, batch, batch_idx):
        # Assuming batch is (x, ...) or just x
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        reconstruction, posterior = self(x)
        
        # 1. Reconstruction Loss (L1 or MSE)
        rec_loss = F.mse_loss(reconstruction, x, reduction='mean') # or L1
        
        # 2. KL Divergence Loss
        kl_loss = posterior.kl()
        kl_loss = torch.mean(kl_loss)
        
        total_loss = rec_loss + self.kl_weight * kl_loss
        
        self.log("vae/rec_loss", rec_loss, prog_bar=True)
        self.log("vae/kl_loss", kl_loss, prog_bar=True)
        self.log("vae/total_loss", total_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        reconstruction, posterior = self(x, sample_posterior=False)
        rec_loss = F.mse_loss(reconstruction, x)
        
        self.log("val/rec_loss", rec_loss, sync_dist=True)

    def configure_optimizers(self):
        from hydra.utils import instantiate
        if self.optimizer_cfg:
            return instantiate(self.optimizer_cfg, params=self.parameters())
        # Default fallback
        return torch.optim.Adam(self.parameters(), lr=1e-4)