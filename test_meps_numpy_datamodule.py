from __future__ import annotations
import os
import sys
from pathlib import Path
import warnings

import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra import main as hydra_main
from hydra.utils import instantiate

from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info
import torch.distributed as dist
import logging


LOGGER = logging.getLogger("trainer")
 

rank=-1
world_size=-1



#warnings.filterwarnings("ignore", ".*does not have many workers.*")

@hydra_main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Reproducibility (DDP-friendly)
    L.seed_everything(cfg.seed, workers=True)

    # Matmul kernel selection
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


    dm = instantiate(cfg.datamodule, _recursive_=False)  
    
    dm.prepare_data()   # safe no-op here
    dm.setup("fit")     # build train/val/test datasets

    # Train dataset
    train_set = dm.train_set
    print("Train samples:", len(train_set))
    
    # One item (without DataLoader collation)
    x, y, meta = train_set[0]

    print("Single sample:")
    print("  length:", len(train_set)  )
    print("  x:", x.shape, x.dtype)
    print("  y:", y.shape, y.dtype)
    print("  meta:", meta)

    print(torch.min(x), torch.max(x), torch.mean(x), torch.std(x))

if __name__ == "__main__":
    # Common env for NCCL stability in some multi-GPU clusters
    #os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
 
    main()
