# train.py
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
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import logging

from utils.log_hydra import log_cfg_to_mlflow

LOGGER = logging.getLogger("trainer")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("TORCH CUDA AVAILABLE:", torch.cuda.is_available())
print("TORCH DEVICE COUNT:", torch.cuda.device_count())

warnings.filterwarnings("ignore", ".*does not have many workers.*")

@hydra_main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # --------- data & model ----------
    dm = instantiate(cfg.datamodule, _recursive_=False)
    model = instantiate(cfg.model, _recursive_=False, _convert_="partial")

    # --------- logger & callbacks ----------
    logger = instantiate(cfg.logger)
    log_cfg_to_mlflow(logger, cfg)  # safe no-op if not MLflow

    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", [])]

    # --- profiler ---
    profiler = None
    if "profiler" in cfg.trainer and cfg.trainer.profiler is not None:
        profiler = instantiate(cfg.trainer.profiler)

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("profiler", None)

    trainer = L.Trainer(**trainer_cfg, profiler=profiler, logger=logger, callbacks=callbacks)

    # --------- train ----------
    start_time = time.perf_counter()

    ckpt_path = None
    resume_from = cfg.get("resume_from_ckpt", None)
    if resume_from in ("last", "best", "file"):
        for cb in callbacks:
            if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint):
                if resume_from == "last" and cb.last_model_path:
                    ckpt_path = cb.last_model_path
                elif resume_from == "best" and cb.best_model_path:
                    ckpt_path = cb.best_model_path
                elif resume_from == "file":
                    ckpt_path = cfg.get("ckpt_path", None)
                break

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    elapsed_td = timedelta(seconds=int(time.perf_counter() - start_time))
    rank_zero_info(f"Run completed in {elapsed_td} (hh:mm:ss)")

if __name__ == "__main__":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    main()
