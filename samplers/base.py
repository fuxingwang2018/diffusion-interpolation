# samplers/sampler_base.py
from abc import ABC, abstractmethod
import lightning as L

class SamplerBase(ABC):
    """
    All samplers must expose:
      - .name : short identifier, e.g. "heun_edm", "ddim", "iddpm"
      - .sample(module, cond, target_shape, device, **kwargs) -> Tensor
    """
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def sample(self, module, cond, target_shape, device, **kwargs):
        ...
