# diffusion/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set, Tuple
from abc import ABC, abstractmethod
from hydra.utils import instantiate

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from .base import AutoEncoderBase

class IdentityAutoEncoder(AutoEncoderBase):
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z   
    