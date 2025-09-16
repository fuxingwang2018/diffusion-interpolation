from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import json
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader


# ----------------------------------
# CSV -> in-memory "ensemble windows"
# ----------------------------------

def _ensure_abs(p: Union[str, Path], root: Union[str, Path]) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else Path(root) / p)

def read_grouped_ensemble_windows(
    csv_path: Union[str, Path],
    root: Union[str, Path],
    members_filter: Optional[Iterable[int]] = None,
    leads_filter: Optional[Iterable[int]] = None,
    require_internal_targets: bool = True,
    require_all_selected_members: bool = True,
) -> List[Dict[str, Any]]:
    """
    Read 'grouped' sequences CSV (columns Members, LeadTimes, MemberPaths as JSON lists).
    Return one record per (Date, Window) that contains ALL selected members (if required),
    with lead-times sorted ascending and paths aligned as a 2D list: [member][lead].

    Record schema:
      {
        "Date": str | None,
        "Window": str | None,
        "members": List[int],                # in CSV order after filtering
        "lead_times": List[int],             # sorted ascending (after filtering)
        "member_paths": List[List[str]],     # shape (M, T), absolute paths
      }
    """
    df = pd.read_csv(csv_path)
    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        members = row["Members"]
        leads = row["LeadTimes"]
        mpaths = row["MemberPaths"]
        if isinstance(members, str): members = json.loads(members)
        if isinstance(leads, str): leads = json.loads(leads)
        if isinstance(mpaths, str): mpaths = json.loads(mpaths)

        # filter leads (preserve selection, then sort ascending)
        lt_sel = list(leads)
        if leads_filter is not None:
            keep = set(int(x) for x in leads_filter)
            lt_sel = [lt for lt in lt_sel if lt in keep]
        lt_sel = sorted(lt_sel)

        # need endpoints; optionally need internals
        if len(lt_sel) < 2:
            continue
        if require_internal_targets and len(lt_sel) < 3:
            continue

        # filter members (preserve original order)
        if members_filter is None:
            mem_sel = list(members)
        else:
            req = list(int(x) for x in members_filter)
            mem_sel = [m for m in members if m in set(req)]
            if require_all_selected_members and set(req) - set(mem_sel):
                # If any requested member missing in this window, drop it
                continue

        # build member_paths 2D (M x T) respecting selected orders
        lt2i = {lt: i for i, lt in enumerate(leads)}
        m2i = {m: i for i, m in enumerate(members)}
        member_paths: List[List[str]] = []
        for m in mem_sel:
            mi = m2i[m]
            row_paths = [mpaths[mi][lt2i[lt]] for lt in lt_sel]
            member_paths.append([_ensure_abs(p, root) for p in row_paths])

        out.append({
            "Date": row.get("Date"),
            "Window": row.get("Window"),
            "members": [int(m) for m in mem_sel],
            "lead_times": [int(lt) for lt in lt_sel],
            "member_paths": member_paths,  # (M, T)
        })

    return out


# --------------------
# Normalization helper
# --------------------

class Normalizer:
    """
    Apply normalization to arrays of shape (C, H, W). Supports broadcasting of stats
    (scalar, (C,), (H,W), or (C,H,W)).

    Modes:
      - "none"
      - "zscore":   (x - mean) / std
      - "symrange": norm_const * (x - average) / max(|global_min - average|, |global_max - average|)
    """

    def __init__(
        self,
        mode: Literal["none", "zscore", "symrange"] = "none",
        stats_npz: Optional[Union[str, Path]] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 1.0,
    ) -> None:
        self.mode = mode
        self.norm_const = float(norm_const)
        self.arr_mean = self.arr_std = None
        self.arr_avg = self.arr_gmin = self.arr_gmax = None

        if self.mode != "none":
            if stats_npz is None:
                raise ValueError(f"normalize='{self.mode}' requires stats_npz")
            stats = np.load(stats_npz, allow_pickle=False)
            if self.mode == "zscore":
                self.arr_mean = np.asarray(stats[mean_key])
                self.arr_std = np.asarray(stats[std_key])
            elif self.mode == "symrange":
                self.arr_avg  = np.asarray(stats[average_key])
                self.arr_gmin = np.asarray(stats[global_min_key])
                self.arr_gmax = np.asarray(stats[global_max_key])

    def _bcast(self, ref: np.ndarray, arr: np.ndarray) -> np.ndarray:
        # try broadcast as-is
        try:
            _ = ref + arr
            return arr
        except Exception:
            pass
        # reshape common cases
        if arr.ndim == 0:
            return arr
        if arr.ndim == 1 and arr.shape[0] == ref.shape[0]:   # (C,)
            return arr[:, None, None]
        if arr.ndim == 2 and arr.shape == ref.shape[1:]:     # (H, W)
            return arr[None, ...]
        if arr.shape == ref.shape:                            # (C, H, W)
            return arr
        raise ValueError(f"Stats shape {arr.shape} not broadcastable to {ref.shape}")

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x
        if self.mode == "zscore":
            mean = self._bcast(x, self.arr_mean)
            std  = self._bcast(x, self.arr_std)
            std  = np.where(std == 0, 1.0, std)
            return (x - mean) / std
        if self.mode == "symrange":
            avg  = self._bcast(x, self.arr_avg)
            gmin = self._bcast(x, self.arr_gmin)
            gmax = self._bcast(x, self.arr_gmax)
            denom = np.maximum(np.abs(gmin - avg), np.abs(gmax - avg))
            denom = np.where(denom == 0, 1.0, denom)
            return self.norm_const * (x - avg) / denom
        raise ValueError(f"Unknown normalize mode: {self.mode}")


# ---------------------------
# Dataset with two sample modes
# ---------------------------

class MEPSWindowDataset(Dataset):
    """
    Build samples from ensemble windows.

    sample_mode:
      - "ensemble": one sample per (Date, Window) with an explicit member axis
                    x: (M, 2*C, H, W)  or (M, 2, C, H, W) if stack_time_on_channel=False
                    y: (M, (T-2)*C, H, W) or (M, T-2, C, H, W)
      - "per_member": one sample per (Date, Window, Member)
                    x: (2*C, H, W)     or (2, C, H, W)
                    y: ((T-2)*C, H, W) or (T-2, C, H, W)

    If stack_member_on_channel=True (ensemble mode only), the member axis is also
    fused onto channels for CNNs that want (C,H,W) only.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        file_channel_indices: Sequence[int] = (0, 1, 2, 3),
        sample_mode: Literal["ensemble", "per_member"] = "ensemble",
        stack_time_on_channel: bool = True,
        stack_member_on_channel: bool = False,  # only used in ensemble mode
        dtype: Literal["float32", "float16", "bfloat16", "float64"] = "float32",
        mmap: bool = True,
        # normalization
        normalize: Literal["none", "zscore", "symrange"] = "none",
        stats_npz: Optional[Union[str, Path]] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 1.0,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("MEPSWindowDataset: received 0 records.")

        self.recs = records
        self.chan_idx = list(file_channel_indices)
        self.sample_mode = sample_mode
        self.stack_time_on_channel = bool(stack_time_on_channel)
        self.stack_member_on_channel = bool(stack_member_on_channel)
        self.mmap = mmap

        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        self.out_dtype = self.dtype_map[dtype]
        self.norm = Normalizer(
            mode=normalize, stats_npz=stats_npz,
            mean_key=mean_key, std_key=std_key,
            average_key=average_key, global_min_key=global_min_key,
            global_max_key=global_max_key, norm_const=norm_const,
        )

        # For "per_member", build an index mapping each dataset index to (rec_idx, member_idx)
        self._index: Optional[List[Tuple[int, int]]] = None
        if self.sample_mode == "per_member":
            idx: List[Tuple[int, int]] = []
            for r_i, rec in enumerate(self.recs):
                M = len(rec["members"])
                idx.extend([(r_i, m_i) for m_i in range(M)])
            self._index = idx

    def __len__(self) -> int:
        if self.sample_mode == "ensemble":
            return len(self.recs)
        else:
            return len(self._index)  # type: ignore[arg-type]

    def _load_sel(self, path: str) -> np.ndarray:
        arr = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)  # (4,H,W)
        if arr.ndim != 3 or arr.shape[0] < max(self.chan_idx) + 1:
            raise ValueError(f"Expected (>=C,H,W), got {arr.shape} @ {path}")
        x = arr[np.array(self.chan_idx, dtype=int), ...]  # (C,H,W)
        x = self.norm.apply(x)
        return x

    def _stack_time(self, arr_list: List[np.ndarray]) -> np.ndarray:
        # list of (C,H,W) -> (T,C,H,W)
        return np.stack(arr_list, axis=0)

    def __getitem__(self, i: int):
        if self.sample_mode == "ensemble":
            rec = self.recs[i]
            paths_2d: List[List[str]] = rec["member_paths"]  # (M, T)
            M = len(paths_2d)
            # Build x=(M,2,C,H,W), y=(M,T-2,C,H,W)
            x_m = []
            y_m = []
            for m_paths in paths_2d:
                first = self._load_sel(m_paths[0])
                last  = self._load_sel(m_paths[-1])
                internals = [self._load_sel(p) for p in m_paths[1:-1]]
                x_tchw = self._stack_time([first, last])          # (2,C,H,W)
                y_tchw = self._stack_time(internals)              # (T-2,C,H,W)
                x_m.append(x_tchw); y_m.append(y_tchw)

            x = np.stack(x_m, axis=0)  # (M,2,C,H,W)
            y = np.stack(y_m, axis=0)  # (M,T-2,C,H,W)

            # fuse time->channel
            if self.stack_time_on_channel:
                M, T2, C, H, W = x.shape
                x = x.reshape(M, T2 * C, H, W)            # (M, 2*C, H, W)
                M, Ti, C, H, W = y.shape
                y = y.reshape(M, Ti * C, H, W)            # (M, (T-2)*C, H, W)

            # optionally fuse member->channel (only if you want pure (C,H,W))
            if self.stack_member_on_channel:
                if x.ndim == 4:    # (M, C', H, W)
                    M, Cx, H, W = x.shape; x = x.reshape(M * Cx, H, W)
                    M, Cy, H, W = y.shape; y = y.reshape(M * Cy, H, W)
                else:              # (M, T, C, H, W)
                    M, T2, C, H, W = x.shape; x = x.reshape(M * T2 * C, H, W)
                    M, Ti, C, H, W = y.shape; y = y.reshape(M * Ti * C, H, W)

            return (torch.as_tensor(x, dtype=self.out_dtype),
                    torch.as_tensor(y, dtype=self.out_dtype))

        # sample_mode == "per_member"
        rec_idx, m_idx = self._index[i]  # type: ignore[index]
        rec = self.recs[rec_idx]
        m_paths: List[str] = rec["member_paths"][m_idx]  # length T

        first = self._load_sel(m_paths[0])
        last  = self._load_sel(m_paths[-1])
        internals = [self._load_sel(p) for p in m_paths[1:-1]]

        x_tchw = self._stack_time([first, last])            # (2,C,H,W)
        y_tchw = self._stack_time(internals)                # (T-2,C,H,W)

        if self.stack_time_on_channel:
            T2, C, H, W = x_tchw.shape; x = x_tchw.reshape(T2 * C, H, W)          # (2*C,H,W)
            Ti, C, H, W = y_tchw.shape; y = y_tchw.reshape(Ti * C, H, W)          # ((T-2)*C,H,W)
        else:
            x = x_tchw; y = y_tchw

        return (torch.as_tensor(x, dtype=self.out_dtype),
                torch.as_tensor(y, dtype=self.out_dtype))


# -------------
# DataModule
# -------------

@dataclass
class SplitConfig:
    type: Literal["ratio", "count", "none"] = "ratio"
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    shuffle_before_split: bool = True

class MEPSNPYDataModule(L.LightningDataModule):
    """
    DataModule for MEPS-like npy stacks with grouped CSVs.

    Each .npy is (4,H,W). You pick 'file_channel_indices' (C of them).
    For each member inside a (Date,Window):
      - x := endpoints (first & last leads)
      - y := internals (all leads between first and last)

    sample_mode:
      - "ensemble": sample keeps ALL selected members as a member dimension
      - "per_member": sample is a single member

    Shapes (assuming stack_time_on_channel=True):
      - ensemble:   x = (M, 2*C, H, W),    y = (M, (T-2)*C, H, W)
      - per_member: x = (2*C, H, W),       y = ((T-2)*C, H, W)
    """

    def __init__(
        self,
        # Sources
        root: str = ".",
        sequences_csv: Optional[str] = None,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,

        # Filtering / inclusion
        members: Optional[Sequence[int]] = None,
        leads: Optional[Sequence[int]] = None,
        require_internal_targets: bool = True,
        require_all_selected_members: bool = True,

        # Dataset behavior
        sample_mode: Literal["ensemble", "per_member"] = "ensemble",
        file_channel_indices: Sequence[int] = (0, 1, 2, 3),
        stack_time_on_channel: bool = True,
        stack_member_on_channel: bool = False,  # only used in ensemble mode

        # Normalization
        normalize: Literal["none", "zscore", "symrange"] = "none",
        stats_npz: Optional[str] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 1.0,

        # Loader
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        shuffle_train: bool = True,
        dtype: Literal["float32", "float16", "bfloat16", "float64"] = "float32",
        mmap: bool = True,

        # Split
        split: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.sequences_csv = sequences_csv
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        self.members = None if members is None else list(members)
        self.leads = None if leads is None else list(leads)
        self.require_internal_targets = bool(require_internal_targets)
        self.require_all_selected_members = bool(require_all_selected_members)

        self.sample_mode = sample_mode
        self.file_channel_indices = list(file_channel_indices)
        self.stack_time_on_channel = bool(stack_time_on_channel)
        self.stack_member_on_channel = bool(stack_member_on_channel)

        self.normalize = normalize
        self.stats_npz = stats_npz
        self.mean_key = mean_key
        self.std_key = std_key
        self.average_key = average_key
        self.global_min_key = global_min_key
        self.global_max_key = global_max_key
        self.norm_const = norm_const

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if persistent_workers is not None else (num_workers > 0)
        self.shuffle_train = shuffle_train
        self.dtype = dtype
        self.mmap = mmap

        self.split_cfg = SplitConfig(**split) if split is not None else SplitConfig()

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    # Lightning hooks
    def prepare_data(self) -> None:
        pass

    def _read(self, csv_path: str) -> List[Dict[str, Any]]:
        return read_grouped_ensemble_windows(
            csv_path=csv_path,
            root=self.root,
            members_filter=self.members,
            leads_filter=self.leads,
            require_internal_targets=self.require_internal_targets,
            require_all_selected_members=self.require_all_selected_members,
        )

    def _split(self, records: List[Dict[str, Any]]):
        cfg = self.split_cfg
        n = len(records)
        idx = list(range(n))
        if cfg.shuffle_before_split:
            g = torch.Generator().manual_seed(cfg.seed)
            idx = torch.randperm(n, generator=g).tolist()

        if cfg.type == "none":
            return records, [], []

        if cfg.type == "ratio":
            t = max(0.0, min(1.0, float(cfg.train)))
            v = max(0.0, min(1.0, float(cfg.val)))
            n_train = int(round(t * n))
            n_val = int(round(v * n))
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)
            n_test = n - n_train - n_val
        elif cfg.type == "count":
            n_train, n_val, n_test = int(cfg.train), int(cfg.val), int(cfg.test)
            if n_train + n_val + n_test > n:
                raise ValueError(f"Split counts exceed dataset size ({n}).")
        else:
            raise ValueError(f"Unknown split.type='{cfg.type}'")

        take = lambda ids: [records[i] for i in ids]
        i_tr = idx[:n_train]
        i_va = idx[n_train:n_train + n_val]
        i_te = idx[n_train + n_val:n_train + n_val + n_test]
        return take(i_tr), take(i_va), take(i_te)

    def setup(self, stage: Optional[str] = None) -> None:
        # choose sources
        if self.train_csv or self.val_csv or self.test_csv:
            rec_train = self._read(self.train_csv) if self.train_csv else []
            rec_val   = self._read(self.val_csv) if self.val_csv else []
            rec_test  = self._read(self.test_csv) if self.test_csv else []
        else:
            if not self.sequences_csv:
                raise ValueError("Provide sequences_csv or train/val/test CSVs.")
            all_rec = self._read(self.sequences_csv)
            rec_train, rec_val, rec_test = self._split(all_rec)

        mk = lambda recs: MEPSWindowDataset(
            records=recs,
            file_channel_indices=self.file_channel_indices,
            sample_mode=self.sample_mode,
            stack_time_on_channel=self.stack_time_on_channel,
            stack_member_on_channel=self.stack_member_on_channel,
            dtype=self.dtype,
            mmap=self.mmap,
            normalize=self.normalize,
            stats_npz=self.stats_npz,
            mean_key=self.mean_key,
            std_key=self.std_key,
            average_key=self.average_key,
            global_min_key=self.global_min_key,
            global_max_key=self.global_max_key,
            norm_const=self.norm_const,
        ) if recs else None

        self.train_set = mk(rec_train)
        self.val_set   = mk(rec_val)
        self.test_set  = mk(rec_test)

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            return DataLoader([], batch_size=self.batch_size)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_set is None:
            return DataLoader([], batch_size=self.batch_size)
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_set is None:
            return DataLoader([], batch_size=self.batch_size)
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
