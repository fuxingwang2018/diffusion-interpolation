#!/usr/bin/env python3
"""
sample.py — Generate samples from validation/test/train and save to NPZ.

- Uses Hydra config (same as train.py) via `def main(cfg: DictConfig)`.
- Instantiates datamodule + model.
- Optionally loads a Lightning checkpoint (path).
- Runs inference on N batches from the chosen split (train/val/test).
- Saves NPZ per batch containing:
    * data: predicted internals (first item in batch), shape (T,C,H,W)
    * meta_json: JSON string with dataset metadata + split
Filename pattern (sanitized):
    <split>__<start>__w<window>__m<member>.npz
"""

from __future__ import annotations
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Any, Dict

import torch
import lightning as L
from omegaconf import DictConfig
from hydra import main as hydra_main
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf




def _sanitize(s: Any) -> str:
    """Make a string filesystem-safe (letters, digits, _- only)."""
    if s is None:
        return "NA"
    s = str(s).strip()
    s = s.replace(" ", "_").replace(":", "-").replace("/", "-").replace("\\", "-")
    # keep only safe chars
    safe = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("-")
    out = "".join(safe)
    # collapse repeated dashes/underscores
    while "--" in out:
        out = out.replace("--", "-")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "NA"


def _get_meta_value(meta: Any, key: str, default: str = "NA") -> str:
    """Fetch a value from meta which may be dict-like; fallback to default."""
    try:
        if isinstance(meta, dict):
            return _sanitize(meta.get(key, default))
        # Some datasets carry attr-style objects; try getattr
        if hasattr(meta, key):
            return _sanitize(getattr(meta, key))
    except Exception:
        pass
    return _sanitize(default)


def _build_filename(meta: Any, split: str, sample_index ) -> str:
    """Build filename from start_date, window, member, and split."""
    split_s = _sanitize(split)
    start_s = _get_meta_value(meta, "date", "startNA")
    window_s = _get_meta_value(meta, "window", "wNA")
    member_s = _get_meta_value(meta, "member", "mNA")

    return f"{split_s}__{start_s}__w{window_s}__m{member_s}__s{sample_index}.npz"


# ---------------- Hydra entry ----------------

@hydra_main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:


    assert cfg.batches > 0 or cfg.batches == -1
    assert cfg.split in ("train", "val", "test")

    print("Config:\n", OmegaConf.to_yaml(cfg), flush=True)
    
    # Optional runtime flags (device)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--device", default="cuda:0")
    args, _ = ap.parse_known_args()

    L.seed_everything(cfg.seed, workers=True)

    number_of_samples = cfg.number_of_samples 

    # Instantiate datamodule & model from Hydra cfg
    dm = instantiate(cfg.datamodule, _recursive_=False)
    model = instantiate(cfg.model, _recursive_=False, _convert_="partial")

    # Optionally load checkpoint
    if getattr(cfg, "ckpt_file", None):
        ckpt_path = Path(cfg.ckpt_file)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded ckpt: {ckpt_path}\n  missing={missing}\n  unexpected={unexpected}", flush=True)
        else:
            print(f"[warn] ckpt not found: {ckpt_path}", flush=True)

    # Prepare dataloaders
    dm.prepare_data()
    dm.setup(cfg.split)
    if cfg.split == "test":
        loader = dm.test_dataloader()
    elif cfg.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.train_dataloader()

    
    
    # Device & eval mode
    device = torch.device(getattr(cfg, "device", args.device))
    model.eval().to(device)

    # Output dir
    outdir = cfg.outdir
    os.makedirs(outdir, exist_ok=True)

    num_done = 0
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            print (f"Processing Batch {bidx+1}", flush=True)
            # Unpack
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, meta = batch 
                B = x.size(0)
            else:
                raise RuntimeError(f"Unexpected batch format: {type(batch)}")

            x = x.to(device, non_blocking=True)
 
            # Pack to (cond, target)
            cond = model._pack_x(x)

            cond_shape = cond.shape
            _, C, H, W = cond_shape
            target_shape = (B, 5*int(cfg.number_of_channels), H, W )

  

            for sidx in range(number_of_samples):
                # Predict
                pred = model.sample(cond, target_shape)  # (B, T*C, H, W)
    
                # Shapes for saving (take item 0 of batch)
                pred_bt   = pred.view(B, 5, int(cfg.number_of_channels), H, W)    # (B,T,C,H,W)
                # Optionally also cond_bt if you ever want to save conditioning
                # cond_bt   = _ensure_btchw(cond, guide_y=x)
                for bi in range(B):
                    
                    # Build filename from meta
                     
    
    
                    # Collect metadata (ensure JSON-serializable)
                    meta_dict: Dict[str, Any] = {}
      
                    meta_dict["date"] = meta["date"][bi]
                    meta_dict["window"] = meta["window"][bi]
                    meta_dict["member"] = meta["member"][bi].tolist()
                    meta_dict["rec_index"] = meta["rec_index"][bi].tolist()
                    meta_dict["sample_index"] = meta["sample_index"][bi].tolist()
                    meta_dict["start_valid_time"] = meta["start_valid_time"][bi]
                    meta_dict["end_valid_time"] = meta["end_valid_time"][bi]
                    meta_dict["gt_path"] = meta["path"][bi] 
                    meta_dict["split"] = cfg.split
                    if bi == 0:
                        print("meta_dict", meta_dict)
                    meta_json = json.dumps(meta_dict, ensure_ascii=False)
                    fname = _build_filename(meta_dict, split=cfg.split, sample_index=sidx)
                    fpath = os.path.join(outdir, fname)
                    np.savez_compressed(
                    fpath,
                    data=pred_bt[bi].cpu().numpy(),   # (T,C,H,W)
                    meta_json=np.array(meta_json, dtype=object),
                    )
                 

            num_done += 1
            if num_done >= cfg.batches and not (cfg.batches == -1):
               break


if __name__ == "__main__":

    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "64")))
    torch.set_num_interop_threads(2)  # optional
    main()
