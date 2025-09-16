from __future__ import annotations
import os, time
from datetime import timedelta
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
# add project root to sys.path dynamically
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


rank=-1
world_size=-1

if dist.is_available() and dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

rank_zero_info("Cuda is enabled.")

LOGGER.info(f"Hello from rank {rank}")
 


warnings.filterwarnings("ignore", ".*does not have many workers.*")

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
    
    # ####
    # dm.prepare_data()   # safe no-op here
    # dm.setup("fit")     # build train/val/test datasets
    # # Train dataset
    # train_set = dm.train_set
    # print("Train samples:", len(train_set))
    
    # # One item (without DataLoader collation)
    # x, y = train_set[0]

    # print("Single sample:")
    # print("  x:", x.shape, x.dtype)
    # print("  y:", y.shape, y.dtype)

    # exit() 
    # ####

    model = instantiate(cfg.model, _recursive_=False, _convert_="partial",
        optimizer_cfg=cfg.get("optimizer", None),
        scheduler_cfg=cfg.get("lr_scheduler", None))

    # --------- logger & callbacks ----------
    logger = instantiate(cfg.logger)
    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", [])]

    trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)

    # --------- train ----------
    rank_zero_info(OmegaConf.to_yaml(cfg, resolve=True))
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("ckpt_path", None))


if __name__ == "__main__":
    # Common env for NCCL stability in some multi-GPU clusters
    #os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    start_time = time.perf_counter() 
    main()
    end_time = time.perf_counter() 
    elapsed = end_time - start_time
    elapsed_td = timedelta(seconds=int(elapsed)) 
    rank_zero_info(f"Run completed in {elapsed_td} (hh:mm:ss)")
