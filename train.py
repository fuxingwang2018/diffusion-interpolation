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

from models.simple_cnn import SimpleCNN


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


    model = SimpleCNN(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        dropout=cfg.model.dropout,
        compile_cfg=cfg.model.compile,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.get("lr_scheduler"),
        figures=cfg.model.figures,
    )

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
 
    main()
