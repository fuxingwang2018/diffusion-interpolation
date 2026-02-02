# diffusion/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch 
 
class AutoEncoderBase(ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #  takes B,C,H,W  and returns B,LC,LH,LW
        raise NotImplementedError


    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        #  takes B,LC,LH,LW  and returns B,C,H,W
        raise NotImplementedError


