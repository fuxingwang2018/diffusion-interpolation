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

class AutoEncoderBase(L.LightningModule, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #  takes B,C,H,W  and returns B,LC,LH,LW
        raise NotImplementedError


    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        #  takes B,LC,LH,LW  and returns B,C,H,W
        raise NotImplementedError


