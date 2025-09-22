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


# ============================================================
# Utilities
# ============================================================

def _ensure_abs(p: Union[str, Path], root: Union[str, Path]) -> str:
    """
    Return an absolute path string for `p`, resolved under `root` if `p` is relative.
    """
    p = Path(p)
    return str(p if p.is_absolute() else Path(root) / p)


# ============================================================
# CSV → in-memory grouped ensemble windows
# ============================================================

def read_grouped_ensemble_windows(
    csv_path: Union[str, Path],
    root: Union[str, Path],
    members_filter: Optional[Iterable[int]] = None,
    leads_filter: Optional[Iterable[int]] = None,
    require_internal_targets: bool = True,
    require_all_selected_members: bool = True,
) -> List[Dict[str, Any]]:
    """
    Read a *grouped* sequences CSV where the following columns contain JSON arrays:
      - Members:      list[int]
      - LeadTimes:    list[int]    (ascending is not required; we sort)
      - MemberPaths:  list[list[str]] with shape (M, T)

    Other useful columns (kept as-is if present): Date, Window, StartValidTime, EndValidTime.

    Returns a list of normalized records (dict) per (Date, Window) like:
      {
        "date": str | None,
        "window": str | None,
        "members": List[int],                  # after filtering, original order preserved
        "lead_times": List[int],               # after filtering, sorted ascending
        "member_paths": List[List[str]],       # shape (M, T), absolute paths
        "start_valid_time": Any | None,
        "end_valid_time": Any | None,
      }

    Filtering:
      - If `members_filter` is given, only those members are kept.
        If `require_all_selected_members=True`, the record is dropped if any requested member is missing.
      - If `leads_filter` is given, only those lead times are kept (and then sorted).
      - If fewer than 2 leads remain, the record is dropped (no endpoints).
      - If `require_internal_targets=True`, at least 3 leads must remain (need internals).
    """
    df = pd.read_csv(csv_path)
    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        members = row.get("Members")
        leads = row.get("LeadTimes")
        mpaths = row.get("MemberPaths")

        # Optional columns (pass-through; may be NaN)
        date = row.get("Date")
        window = row.get("Window")
        start_valid_time = row.get("StartValidTime")
        end_valid_time = row.get("EndValidTime")

        # Parse JSON columns if needed
        if isinstance(members, str):
            members = json.loads(members)
        if isinstance(leads, str):
            leads = json.loads(leads)
        if isinstance(mpaths, str):
            mpaths = json.loads(mpaths)

        if not isinstance(members, (list, tuple)) or not isinstance(leads, (list, tuple)) or not isinstance(mpaths, (list, tuple)):
            continue  # malformed row

        # Select and sort lead times
        lt_sel = list(int(x) for x in leads)
        if leads_filter is not None:
            keep = set(int(x) for x in leads_filter)
            lt_sel = [lt for lt in lt_sel if lt in keep]
        lt_sel = sorted(lt_sel)

        # Need endpoints; optionally internals
        if len(lt_sel) < 2:
            continue
        if require_internal_targets and len(lt_sel) < 3:
            continue

        # Filter members (preserve CSV order)
        if members_filter is None:
            mem_sel = [int(m) for m in members]
        else:
            req = [int(x) for x in members_filter]
            mem_sel = [int(m) for m in members if int(m) in set(req)]
            if require_all_selected_members and (set(req) - set(mem_sel)):
                continue  # some requested member missing

        # Align paths to selected leads and members
        lt2i = {int(lt): i for i, lt in enumerate(leads)}
        m2i = {int(m): i for i, m in enumerate(members)}

        member_paths: List[List[str]] = []
        for m in mem_sel:
            mi = m2i[m]
            row_paths = [mpaths[mi][lt2i[lt]] for lt in lt_sel]
            member_paths.append([_ensure_abs(p, root) for p in row_paths])

        out.append({
            "date": date if pd.notna(date) else None,
            "window": window if pd.notna(window) else None,
            "members": mem_sel,
            "lead_times": lt_sel,
            "member_paths": member_paths,     # (M, T)
            "start_valid_time": start_valid_time if pd.notna(start_valid_time) else None,
            "end_valid_time": end_valid_time if pd.notna(end_valid_time) else None,
        })

    return out


# ============================================================
# Normalization
# ============================================================

class Normalizer:
    """
    Apply normalization to channel-first (C, H, W) arrays.

    Modes:
      - "none":     return x unchanged
      - "zscore":   (x - mean) / std
      - "symrange": norm_const * (x - average) / max(|global_min - average|, |global_max - average|)

    Stats are loaded from `stats_npz` and are allowed to be scalar, (C,), (H, W), or (C, H, W).
    If `channel_indices` is provided, stats are sliced along the leading channel axis when possible.
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
        channel_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.mode = mode
        self.norm_const = float(norm_const)
        self.chan_idx = list(channel_indices) if channel_indices is not None else None

        # Arrays (possibly None if mode == "none")
        self.arr_mean = self.arr_std = None
        self.arr_avg = self.arr_gmin = self.arr_gmax = None

        if self.mode != "none":
            if stats_npz is None:
                raise ValueError(f"normalize='{self.mode}' requires stats_npz")
            stats = np.load(stats_npz, allow_pickle=False)

            def _maybe_slice(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr)
                if self.chan_idx is None:
                    return arr
                # Slice along channel axis if shape matches
                if arr.ndim == 1 and arr.shape[0] >= max(self.chan_idx) + 1:
                    return arr[self.chan_idx]
                if arr.ndim == 3 and arr.shape[0] >= max(self.chan_idx) + 1:
                    return arr[self.chan_idx, ...]
                return arr

            if self.mode == "zscore":
                self.arr_mean = _maybe_slice(stats[mean_key])
                self.arr_std = _maybe_slice(stats[std_key])
            elif self.mode == "symrange":
                self.arr_avg = _maybe_slice(stats[average_key])
                self.arr_gmin = _maybe_slice(stats[global_min_key])
                self.arr_gmax = _maybe_slice(stats[global_max_key])

    @staticmethod
    def _bcast(ref: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """
        Broadcast `arr` to the shape of `ref` (C,H,W) for common stat shapes:
        scalar, (C,), (H,W), (C,H,W).
        """
        # try direct broadcast
        try:
            _ = ref + arr
            return arr
        except Exception:
            pass

        if arr.ndim == 0:  # scalar
            return arr
        if arr.ndim == 1 and arr.shape[0] == ref.shape[0]:       # (C,)
            return arr[:, None, None]
        if arr.ndim == 2 and arr.shape == ref.shape[1:]:         # (H, W)
            return arr[None, ...]
        if arr.shape == ref.shape:                                # (C, H, W)
            return arr

        raise ValueError(f"Stats shape {arr.shape} not broadcastable to {ref.shape}")

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return x
        if self.mode == "zscore":
            mean = self._bcast(x, self.arr_mean)
            std = self._bcast(x, self.arr_std)
            std = np.where(std == 0, 1.0, std)
            return (x - mean) / std
        if self.mode == "symrange":
            avg = self._bcast(x, self.arr_avg)
            gmin = self._bcast(x, self.arr_gmin)
            gmax = self._bcast(x, self.arr_gmax)
            denom = np.maximum(np.abs(gmin - avg), np.abs(gmax - avg))
            denom = np.where(denom == 0, 1.0, denom)
            return self.norm_const * (x - avg) / denom
        raise ValueError(f"Unknown normalize mode: {self.mode}")


# ============================================================
# Dataset
# ============================================================

class MEPSWindowDataset(Dataset):
    """
    Create (x, y, meta) samples from grouped ensemble windows.

    Two sample modes:

    1) sample_mode="ensemble": one sample per (date, window) keeping all members.
       If stack_time_on_channel=True:
           x shape: (M, 2*C, H, W)     # first+last leads per member, time fused into channels
           y shape: (M, (T-2)*C, H, W) # internal leads per member, time fused into channels
       Else (time kept separate):
           x shape: (M, 2, C, H, W)
           y shape: (M, T-2, C, H, W)
       If stack_member_on_channel=True (only when time is fused), the member axis is also
       folded into channels:
           x shape: (M*2*C, H, W)
           y shape: (M*(T-2)*C, H, W)

    2) sample_mode="per_member": one sample per (date, window, member).
       If stack_time_on_channel=True:
           x shape: (2*C, H, W)
           y shape: ((T-2)*C, H, W)
       Else (time kept separate):
           x shape: (2, C, H, W)
           y shape: (T-2, C, H, W)

    Returns:
        (x: torch.Tensor, y: torch.Tensor, meta: dict)
        where `meta` includes: date, window, lead_times, members/member_index, etc.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        file_channel_indices: Sequence[int] = (0, 1, 2, 3),
        sample_mode: Literal["ensemble", "per_member"] = "ensemble",
        stack_time_on_channel: bool = True,
        stack_member_on_channel: bool = False,  # only used in ensemble mode when time is fused
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
            mode=normalize,
            stats_npz=stats_npz,
            mean_key=mean_key,
            std_key=std_key,
            average_key=average_key,
            global_min_key=global_min_key,
            global_max_key=global_max_key,
            norm_const=norm_const,
            channel_indices=self.chan_idx,
        )

        # Build per-member index for the "per_member" mode
        self._index: Optional[List[Tuple[int, int]]] = None
        if self.sample_mode == "per_member":
            idx: List[Tuple[int, int]] = []
            for r_i, rec in enumerate(self.recs):
                M = len(rec["members"])
                idx.extend((r_i, m_i) for m_i in range(M))
            self._index = idx

    def __len__(self) -> int:
        if self.sample_mode == "ensemble":
            return len(self.recs)
        assert self._index is not None
        return len(self._index)

    # ---------- internal helpers ----------

    def _load_sel(self, path: str) -> np.ndarray:
        """
        Load a .npy of shape (>=C, H, W), select channels, normalize, return (C, H, W).
        """
        arr = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)
        if arr.ndim != 3 or arr.shape[0] < max(self.chan_idx) + 1:
            raise ValueError(f"Expected (>=C,H,W), got {arr.shape} @ {path}")
        x = arr[np.asarray(self.chan_idx, dtype=int), ...]
        x = self.norm.apply(x)
        return x

    @staticmethod
    def _stack_time(arr_list: List[np.ndarray]) -> np.ndarray:
        """
        List of (C,H,W) → (T,C,H,W) by stacking along a new leading time axis.
        """
        return np.stack(arr_list, axis=0)

    # ---------- main sampling ----------

    def __getitem__(self, i: int):
        if self.sample_mode == "ensemble":
            rec = self.recs[i]
            date = rec["date"]
            window = rec["window"]
            members = rec["members"]               # List[int]
            lead_ts = rec["lead_times"]            # List[int]
            paths_2d: List[List[str]] = rec["member_paths"]  # (M, T)
            start_valid_time = rec["start_valid_time"]
            end_valid_time = rec["end_valid_time"]

            # Build x=(M,2,C,H,W), y=(M,T-2,C,H,W)
            x_m: List[np.ndarray] = []
            y_m: List[np.ndarray] = []
            for m_paths in paths_2d:
                first = self._load_sel(m_paths[0])
                last = self._load_sel(m_paths[-1])
                internals = [self._load_sel(p) for p in m_paths[1:-1]]
                x_tchw = self._stack_time([first, last])          # (2,C,H,W)
                y_tchw = self._stack_time(internals)              # (T-2,C,H,W)
                x_m.append(x_tchw)
                y_m.append(y_tchw)

            x = np.stack(x_m, axis=0)  # (M,2,C,H,W)
            y = np.stack(y_m, axis=0)  # (M,T-2,C,H,W)

            # Fuse time->channel if requested
            if self.stack_time_on_channel:
                M, T2, C, H, W = x.shape
                x = x.reshape(M, T2 * C, H, W)                    # (M, 2*C, H, W)
                M, Ti, C, H, W = y.shape
                y = y.reshape(M, Ti * C, H, W)                    # (M, (T-2)*C, H, W)

            # Optionally also fuse member->channel (only meaningful if time already fused)
            if self.stack_member_on_channel and x.ndim == 4:
                M, Cx, H, W = x.shape
                x = x.reshape(M * Cx, H, W)                       # (M*2*C, H, W)
                M, Cy, H, W = y.shape
                y = y.reshape(M * Cy, H, W)                       # (M*(T-2)*C, H, W)

            meta = {
                "sample_mode": "ensemble",
                "date": date,
                "window": window,
                "lead_times": lead_ts,
                "members": members,
                "start_valid_time": start_valid_time,
                "end_valid_time": end_valid_time,
                "rec_index": int(i),
            }

            return (
                torch.as_tensor(x, dtype=self.out_dtype),
                torch.as_tensor(y, dtype=self.out_dtype),
                meta,
            )

        # ---------- sample_mode == "per_member" ----------
        assert self._index is not None
        rec_idx, m_idx = self._index[i]
        rec = self.recs[rec_idx]

        date = rec["date"]
        window = rec["window"]
        members = rec["members"]                      # List[int]
        lead_ts = rec["lead_times"]                   # List[int]
        paths_2d: List[List[str]] = rec["member_paths"]
        m_paths: List[str] = paths_2d[m_idx]          # (T,)

        first_path, last_path = m_paths[0], m_paths[-1]
        internal_list = m_paths[1:-1]

        first = self._load_sel(first_path)
        last = self._load_sel(last_path)
        internals = [self._load_sel(p) for p in internal_list]

        x_tchw = self._stack_time([first, last])      # (2,C,H,W)
        y_tchw = self._stack_time(internals)          # (T-2,C,H,W)

        if self.stack_time_on_channel:
            T2, C, H, W = x_tchw.shape
            x = x_tchw.reshape(T2 * C, H, W)          # (2*C, H, W)
            Ti, C, H, W = y_tchw.shape
            y = y_tchw.reshape(Ti * C, H, W)          # ((T-2)*C, H, W)
        else:
            x, y = x_tchw, y_tchw

        meta = {
            "sample_mode": "per_member",
            "date": date,
            "window": window,
            "lead_times": lead_ts,
            "member_index": int(m_idx),
            "member": int(members[m_idx]) if (isinstance(members, (list, tuple)) and len(members) > m_idx) else int(m_idx),
            "n_members": int(len(members)),
            "start_valid_time": rec.get("start_valid_time"),
            "end_valid_time": rec.get("end_valid_time"),
            "rec_index": int(rec_idx),
        }

        return (
            torch.as_tensor(x, dtype=self.out_dtype),
            torch.as_tensor(y, dtype=self.out_dtype),
            meta,
        )


# ============================================================
# Lightning DataModule
# ============================================================

@dataclass
class SplitConfig:
    """
    Split strategy for a single CSV source:
      - ratio: use train/val/test fractions (must sum to ~1.0 with rounding)
      - count: use absolute counts for train/val/test
      - none:  don't split (everything → train; val/test empty)
    """
    type: Literal["ratio", "count", "none"] = "ratio"
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    shuffle_before_split: bool = True


class MEPSNPYDataModule(L.LightningDataModule):
    """
    DataModule for MEPS-like ensembles stored as .npy stacks with a grouped CSV.

    Each .npy file is (C_file, H, W). You select `file_channel_indices` (C of them).
    For each member inside a (date, window):
      - x := endpoints (first & last lead)
      - y := internals (all leads between first and last)

    sample_mode behavior matches `MEPSWindowDataset` (see its docstring).
    """

    def __init__(
        self,
        # Sources
        root: str = ".",
        sequences_csv: Optional[str] = None,   # single CSV; will be split by `split`
        train_csv: Optional[str] = None,       # OR provide explicit CSVs (no split)
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
        stack_member_on_channel: bool = False,  # ensemble mode only (when time fused)

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

        # Split (only when sequences_csv is provided)
        split: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # Sources
        self.root = root
        self.sequences_csv = sequences_csv
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        # Filters
        self.members = None if members is None else list(members)
        self.leads = None if leads is None else list(leads)
        self.require_internal_targets = bool(require_internal_targets)
        self.require_all_selected_members = bool(require_all_selected_members)

        # Dataset behavior
        self.sample_mode = sample_mode
        self.file_channel_indices = list(file_channel_indices)
        self.stack_time_on_channel = bool(stack_time_on_channel)
        self.stack_member_on_channel = bool(stack_member_on_channel)

        # Normalization
        self.normalize = normalize
        self.stats_npz = stats_npz
        self.mean_key = mean_key
        self.std_key = std_key
        self.average_key = average_key
        self.global_min_key = global_min_key
        self.global_max_key = global_max_key
        self.norm_const = norm_const

        # Loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if persistent_workers is not None else (num_workers > 0)
        self.shuffle_train = shuffle_train
        self.dtype = dtype
        self.mmap = mmap

        # Split
        self.split_cfg = SplitConfig(**split) if split is not None else SplitConfig()

        # Datasets
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    # ---------- Lightning hooks ----------

    def prepare_data(self) -> None:
        # No downloads or preprocessing here; CSV and npy files are assumed to exist.
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
        """
        Split `records` according to `self.split_cfg` (ratio | count | none).
        Returns (train_records, val_records, test_records).
        """
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

        def take(ids: List[int]) -> List[Dict[str, Any]]:
            return [records[i] for i in ids]

        i_tr = idx[:n_train]
        i_va = idx[n_train:n_train + n_val]
        i_te = idx[n_train + n_val:n_train + n_val + n_test]
        return take(i_tr), take(i_va), take(i_te)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate train/val/test datasets either from a single CSV (then split)
        or from explicit CSVs per split (no additional splitting).
        """
        if self.train_csv or self.val_csv or self.test_csv:
            rec_train = self._read(self.train_csv) if self.train_csv else []
            rec_val = self._read(self.val_csv) if self.val_csv else []
            rec_test = self._read(self.test_csv) if self.test_csv else []
        else:
            if not self.sequences_csv:
                raise ValueError("Provide `sequences_csv` or explicit `train_csv`/`val_csv`/`test_csv`.")
            all_rec = self._read(self.sequences_csv)
            rec_train, rec_val, rec_test = self._split(all_rec)

        def mk(recs: List[Dict[str, Any]]) -> Optional[MEPSWindowDataset]:
            if not recs:
                return None
            return MEPSWindowDataset(
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
            )

        self.train_set = mk(rec_train)
        self.val_set = mk(rec_val)
        self.test_set = mk(rec_test)

    # ---------- DataLoaders ----------

    def _loader(self, ds: Optional[Dataset], shuffle: bool) -> DataLoader:
        if ds is None:
            # Return a no-op loader; caller should handle empty sets gracefully.
            return DataLoader([], batch_size=self.batch_size)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)
