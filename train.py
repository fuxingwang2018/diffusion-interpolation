from __future__ import annotations
import os
import warnings

import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf
from hydra import main as hydra_main
from hydra.utils import instantiate

from models.simple_cnn import SimpleCNN
from data.mnist_datamodule import MNISTDataModule

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

    # --------- data & model ----------
    dm = MNISTDataModule(
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        val_split=cfg.data.val_split,
        download=cfg.data.download,
    )

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
    print(OmegaConf.to_yaml(cfg, resolve=True))
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    # Common env for NCCL stability in some multi-GPU clusters
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    main()
