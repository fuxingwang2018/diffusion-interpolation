# diffusion/base.py
from __future__ import annotations
import torch 
from .base import AutoEncoderBase

class IdentityAutoEncoder(AutoEncoderBase):
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z   
    