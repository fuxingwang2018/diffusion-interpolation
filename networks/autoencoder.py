# diffusion/models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptionalVAE(nn.Module):
    """
    Placeholder VAE-like encoder/decoder for latent diffusion.
    Encodes 'cond' (2*C channels) and 'target' (K*C channels) separately.
    In practice, plug a real VAE (e.g., SD's autoencoder) here.
    """
    def __init__(self, cfg):
        super().__init__()
        self.enabled = cfg.enable
        self.frozen  = cfg.frozen
        self.latent_channels = cfg.latent_channels
        self.in_channels = cfg.in_channels
        if not self.enabled:
            self.encoder_cond = self.encoder_targ = self.decoder_targ = None
            return

        # Simple enc/dec stubs (downsample x4)
        def enc_block(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1), nn.SiLU(),
                nn.Conv2d(64, self.latent_channels, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(self.latent_channels, self.latent_channels, 3, stride=2, padding=1), nn.SiLU(),
            )
        def dec_block(out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(self.latent_channels, self.latent_channels, 4, stride=2, padding=1), nn.SiLU(),
                nn.ConvTranspose2d(self.latent_channels, 64, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(64, out_ch, 3, padding=1), nn.Tanh(),
            )

        # cond has 2*C channels, target has K*C channels
        self.encoder_cond = enc_block(cfg.cond_channels)
        self.encoder_targ = enc_block(cfg.target_channels)
        self.decoder_targ = dec_block(cfg.target_channels)

        if self.frozen:
            for p in self.parameters(): p.requires_grad_(False)

    def encode_cond(self, cond):   # (B, 2*C, H, W) -> (B, 2*Clat?, H', W')
        if not self.enabled: return cond
        return self.encoder_cond(cond)

    def encode_target(self, target):  # (B, K*C, H, W) -> (B, K*Clat?, H', W')
        if not self.enabled: return target
        return self.encoder_targ(target)

    def decode_target(self, z_target):
        if not self.enabled: return z_target
        return self.decoder_targ(z_target)
