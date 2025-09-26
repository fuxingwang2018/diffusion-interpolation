#!/usr/bin/env python3
"""
Compute per-channel global statistics (min, max, mean, std) from .npy files listed in a CSV.
- Works with merged 4D arrays (e.g., (T, C, H, W)) or legacy 3D arrays (C, H, W).
- Reduces across all non-channel axes (time + space), leaving one value per channel.
- Saves results to a single .npz file.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

disable_tqdm = "SLURM_JOB_ID" in os.environ  # Often nicer to disable TQDM on SLURM


def _normalize_axes(ndim: int, axis: int) -> int:
    """Convert possibly-negative axis to a non-negative one and validate."""
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise ValueError(f"Axis {axis} out of range for array with ndim={ndim}")
    return axis


def file_stats(
    fn: Path,
    channel_axis_hint: int,
    time_axis_hint: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load a .npy and compute per-channel min/max/sum/sum_sq over all non-channel axes.

    Returns:
        fmin, fmax, fsum, fsum_sq, count_elements_per_channel
        where each array has shape (C,)
        and count_elements_per_channel is TOTAL elements per channel (same for all channels).
    """
    arr = np.load(fn, mmap_mode="r")  # supports big arrays without loading fully into RAM
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape} in {fn}")

    # Normalize axes and move channel axis to front
    if arr.ndim == 3:
        c_axis = _normalize_axes(3, channel_axis_hint)
        a = np.moveaxis(arr, c_axis, 0)  # (C, H, W)
        reduce_axes = (1, 2)
    else:
        c_axis = _normalize_axes(4, channel_axis_hint)
        if time_axis_hint is not None:
            t_axis = _normalize_axes(4, time_axis_hint)
            if t_axis == c_axis:
                raise ValueError(
                    f"time_axis ({time_axis_hint}) and channel_axis ({channel_axis_hint}) refer to same dim in {fn}"
                )
        a = np.moveaxis(arr, c_axis, 0)  # (C, ?, ?, ?)
        reduce_axes = tuple(range(1, a.ndim))  # reduce over everything except channel

    a64 = a.astype(np.float64, copy=False)
    fmin = np.min(a64, axis=reduce_axes)
    fmax = np.max(a64, axis=reduce_axes)
    fsum = np.sum(a64, axis=reduce_axes)
    fsum_sq = np.sum(a64 * a64, axis=reduce_axes)

    per_channel_elems = int(np.prod([a.shape[ax] for ax in reduce_axes], dtype=np.int64))
    return fmin, fmax, fsum, fsum_sq, per_channel_elems


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute global stats from merged .npy file-list CSV.")
    ap.add_argument("--file-list", type=Path, required=True, help="CSV file with a column named 'file-list'")
    ap.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Optional root path to prepend to relative file paths",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output .npz file path")
    ap.add_argument(
        "--channel-axis",
        type=int,
        default=1,
        help="Channel axis in arrays. For merged (T,C,H,W) use 1 (default). For legacy (C,H,W) use 0.",
    )
    ap.add_argument(
        "--time-axis",
        type=int,
        default=0,
        help="Time axis for merged arrays. Ignored for 3D inputs. Default 0 (T,C,H,W).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: CPU count)",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Chunksize for ProcessPoolExecutor.map",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.file_list)
    if "file-list" not in df.columns:
        raise ValueError("CSV must have a column named 'file-list'")

    files = [Path(x) for x in df["file-list"].dropna().astype(str).tolist()]
    if args.root_dir is not None:
        files = [args.root_dir / f for f in files]

    total_files = len(files)
    if total_files == 0:
        logger.warning("No files found in CSV.")
        np.savez(args.out, global_min=[], global_max=[], global_mean=[], global_std=[])
        logger.info("Saved empty stats to %s", args.out)
        return

    # Probe first file to determine number of channels after moving channel axis to front
    probe = np.load(files[0], mmap_mode="r")
    if probe.ndim == 3:
        c_axis = _normalize_axes(3, args.channel_axis)
        sample = np.moveaxis(probe, c_axis, 0)  # (C, H, W)
    elif probe.ndim == 4:
        c_axis = _normalize_axes(4, args.channel_axis)
        sample = np.moveaxis(probe, c_axis, 0)  # (C, T, H, W) or similar
    else:
        raise ValueError(f"Expected 3D or 4D arrays. First file has shape {probe.shape} -> {files[0]}")
    n_channels = int(sample.shape[0])

    global_min = np.full(n_channels, np.inf, dtype=np.float64)
    global_max = np.full(n_channels, -np.inf, dtype=np.float64)
    global_sum = np.zeros(n_channels, dtype=np.float64)
    global_sum_sq = np.zeros(n_channels, dtype=np.float64)
    total_elems = 0  # elements per channel aggregated over all files

    processed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        iterator = exe.map(
            file_stats,
            files,                      # fn
            repeat(args.channel_axis),  # channel_axis_hint
            repeat(args.time_axis),     # time_axis_hint
            chunksize=args.chunksize,
        )
        for fmin, fmax, fsum, fsum_sq, n_elems in tqdm(
            iterator, total=total_files, disable=disable_tqdm, desc="Processing files"
        ):
            if fmin.shape[0] != n_channels:
                raise ValueError("Channel count mismatch between files; check --channel-axis / shapes.")

            global_min = np.minimum(global_min, fmin)
            global_max = np.maximum(global_max, fmax)
            global_sum += fsum
            global_sum_sq += fsum_sq
            total_elems += n_elems
            processed += 1

            if processed % 1000 == 0 or processed == total_files:
                logger.info("Processed %d/%d files (%.1f%%)", processed, total_files, 100 * processed / total_files)

    if total_elems == 0:
        raise ValueError("Total reduced elements is zero; check array shapes/axes.")

    global_mean = global_sum / total_elems
    # Var = E[x^2] - (E[x])^2
    global_var = global_sum_sq / total_elems - global_mean ** 2
    global_var = np.maximum(global_var, 0.0)  # clamp tiny negatives
    global_std = np.sqrt(global_var)

    # Report
    for i in range(n_channels):
        logger.info(
            "Channel %d: min=% .6g, max=% .6g, mean=% .6g, std=% .6g",
            i, global_min[i], global_max[i], global_mean[i], global_std[i]
        )

    # Save all stats into a single .npz file
    np.savez(
        args.out,
        global_min=global_min,
        global_max=global_max,
        global_mean=global_mean,
        global_std=global_std,
    )
    logger.info("Saved global stats to %s", args.out)


if __name__ == "__main__":
    main()
