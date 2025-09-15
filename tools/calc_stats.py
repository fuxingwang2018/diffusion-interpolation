#!/usr/bin/env python3
"""
Compute per-channel global statistics (min, max, mean, std) from .npy files listed in a CSV.
Results are saved to a single .npz file.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

disable_tqdm = "SLURM_JOB_ID" in os.environ  # disable tqdm under SLURM


def file_stats(fn: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    arr = np.load(fn, mmap_mode="r")  # shape (C, H, W)
    fmin = arr.min(axis=(1, 2))
    fmax = arr.max(axis=(1, 2))
    fsum = arr.sum(axis=(1, 2))
    fsum_sq = (arr ** 2).sum(axis=(1, 2))
    pixel_count = int(arr.shape[1] * arr.shape[2])
    return fmin, fmax, fsum, fsum_sq, pixel_count


def main():
    ap = argparse.ArgumentParser(description="Compute global stats from file-list CSV.")
    ap.add_argument("--file-list", type=Path, required=True, help="CSV file with column 'file-list'")
    ap.add_argument("--root-dir", type=Path, required=True, help="Optional root path to prepend to file paths")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz file")
    args = ap.parse_args()

    df = pd.read_csv( args.file_list)
    if "file-list" not in df.columns:
        raise ValueError("CSV must have a column named 'file-list'")
    files = [Path(x) for x in df["file-list"].dropna().tolist()]

    total_files = len(files)
    if total_files == 0:
        logger.warning("No files found in CSV.")
        return
    root_dir = args.root_dir
    if root_dir is not None:
        files = [root_dir / f for f in files]
    # Load first file to get channel count
    sample_arr = np.load(files[0], mmap_mode="r")
    n_channels = sample_arr.shape[0]

    global_min    = np.full(n_channels,  np.inf, dtype=np.float64)
    global_max    = np.full(n_channels, -np.inf, dtype=np.float64)
    global_sum    = np.zeros(n_channels, dtype=np.float64)
    global_sum_sq = np.zeros(n_channels, dtype=np.float64)
    total_pixels  = 0

    processed = 0
    with ProcessPoolExecutor() as exe:
        for fmin, fmax, fsum, fsum_sq, pixel_count in tqdm(
            exe.map(file_stats, files, chunksize=32),
            total=total_files,
            disable=disable_tqdm,
            desc="Processing files"
        ):
            processed += 1
            global_min    = np.minimum(global_min, fmin)
            global_max    = np.maximum(global_max, fmax)
            global_sum    += fsum
            global_sum_sq += fsum_sq
            total_pixels  += pixel_count

            if processed % 1000 == 0 or processed == total_files:
                logger.info("Processed %d/%d files (%.1f%%)", processed, total_files, 100*processed/total_files)

    global_mean = global_sum / total_pixels
    global_var  = global_sum_sq / total_pixels - global_mean**2
    global_var  = np.maximum(global_var, 0)
    global_std  = np.sqrt(global_var)

    # Report
    for i in range(n_channels):
        logger.info("Channel %d: min=% .6g, max=% .6g, mean=% .6g, std=% .6g",
                    i, global_min[i], global_max[i], global_mean[i], global_std[i])

    # Save all stats into a single .npz file
    np.savez(args.out,
             global_min=global_min,
             global_max=global_max,
             global_mean=global_mean,
             global_std=global_std)

    logger.info("Saved global stats to %s", args.out)


if __name__ == "__main__":
    main()
