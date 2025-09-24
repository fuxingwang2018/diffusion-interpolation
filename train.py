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
import logging
from lightning.pytorch.loggers import MLFlowLogger
from utils.log_hydra import log_cfg_to_mlflow


LOGGER = logging.getLogger("trainer")
# add project root to sys.path dynamically
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("TORCH CUDA AVAILABLE:", torch.cuda.is_available())
print("TORCH DEVICE COUNT:", torch.cuda.device_count())




 

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

    #rank_zero_info(OmegaConf.to_yaml(cfg, resolve=True))
    
    
    # --------- data & model ----------
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

    model = instantiate(cfg.model, _recursive_=False, _convert_="partial")

    # --------- logger & callbacks ----------
    logger = instantiate(cfg.logger)
    log_cfg_to_mlflow(logger, cfg)
    
    #if isinstance(logger, MLFlowLogger):
    #    exp_id = logger.experiment_id
    #    rank_zero_info(f"MLflow experiment_name={logger._experiment_name}, run_id={logger.run_id}")
    #    rank_zero_info(f"MLflow experiment_id={exp_id}, tracking_uri={logger.save_dir}")
    #
    #exit()

    callbacks = [instantiate(cb) for cb in cfg.get("callbacks", [])]

    # --- profiler ---
    profiler = None
    if "profiler" in cfg.trainer and cfg.trainer.profiler is not None:
        profiler = instantiate(cfg.trainer.profiler)
    
    # --- trainer args ---
    # Convert DictConfig to a plain dict (resolves interpolations too)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # Remove profiler key so it doesn't get passed twice
    trainer_cfg.pop("profiler", None)


    trainer = L.Trainer(**trainer_cfg, profiler=profiler, logger=logger, callbacks=callbacks)

    # --------- train ----------

    start_time = time.perf_counter() 

    ckpt_path = None
    resume_from = cfg.get("resume_from_ckpt", None)

    if resume_from in ("last", "best", "file"):
        # Look in the checkpoint callback if available
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

    end_time = time.perf_counter() 
    elapsed = end_time - start_time
    elapsed_td = timedelta(seconds=int(elapsed)) 
    rank_zero_info(f"Run completed in {elapsed_td} (hh:mm:ss)")

if __name__ == "__main__":
    # Common env for NCCL stability in some multi-GPU clusters
    #os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    main()
