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
    train_ds = dm.train_ds
    print("Train samples:", len(train_ds))
    
    # One item (without DataLoader collation)
    x, y, meta = train_ds[0]

    print("Single sample:")
    print("  length:", len(train_ds)  )
    print("  x:", x.shape, x.dtype)
    print("  y:", y.shape, y.dtype)
    print("  meta:", meta)

    # Global stats
    print("\nGlobal Stats:")
    print(f"Min: {torch.min(x):.4f}, Max: {torch.max(x):.4f}, Mean: {torch.mean(x.float()):.4f}, Std: {torch.std(x.float()):.4f}")

    # Per-channel stats
    # Assuming x is (C, ...) channel-first, which is standard for PyTorch datasets
    print("\nPer-Channel Statistics:")
    print(f"{'Ch':<4} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 55)

    num_channels = x.shape[0]
    for c in range(num_channels):
        # Select channel and cast to float for accurate mean/std calculation
        c_data = x[c].float() 
        
        c_min = torch.min(c_data).item()
        c_max = torch.max(c_data).item()
        c_mean = torch.mean(c_data).item()
        c_std = torch.std(c_data).item()
        
        print(f"{c:<4} | {c_min:<10.4f} | {c_max:<10.4f} | {c_mean:<10.4f} | {c_std:<10.4f}")

if __name__ == "__main__":
    # Common env for NCCL stability in some multi-GPU clusters
    #os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
 
    main()