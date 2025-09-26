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
from lightning.pytorch.utilities.rank_zero import rank_zero_info

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
# CSV → in-memory grouped ensemble windows (MERGED variant)
# ============================================================

def read_grouped_ensemble_windows_merged(
    csv_path: Union[str, Path],
    root: Union[str, Path],
    members_filter: Optional[Iterable[int]] = None,
    leads_filter: Optional[Iterable[int]] = None,
    require_internal_targets: bool = True,
    require_all_selected_members: bool = True,
) -> List[Dict[str, Any]]:
    """
    Read a grouped sequences CSV where each member has ONE merged file with shape (T,C,H,W).

    Required JSON columns:
      - Members:       list[int]
      - LeadTimes:     list[int]    (order exactly matches the merged file time axis)
      - MemberMerged:  list[str]    (one path per member)

    Optional passthrough columns: Date, Window, StartValidTime, EndValidTime.
    """
    df = pd.read_csv(csv_path)
    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        members = row.get("Members")
        leads = row.get("LeadTimes")
        mmerged = row.get("MemberMerged")

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
        if isinstance(mmerged, str):
            mmerged = json.loads(mmerged)

        if not isinstance(members, (list, tuple)) or not isinstance(leads, (list, tuple)) or not isinstance(mmerged, (list, tuple)):
            # Malformed row, skip
            continue

        # Original lead order (as along T in merged file)
        leads_all = [int(x) for x in leads]

        # Optional lead filter (preserve original order)
        if leads_filter is not None:
            keep = set(int(x) for x in leads_filter)
            lead_times = [lt for lt in leads_all if lt in keep]
        else:
            lead_times = list(leads_all)

        # Need endpoints; optionally internals
        if len(lead_times) < 2:
            continue
        if require_internal_targets and len(lead_times) < 3:
            continue

        # Map lead -> index in merged file time axis
        lt2i = {lt: i for i, lt in enumerate(leads_all)}
        lead_indices = [lt2i[lt] for lt in lead_times]

        # Filter members (preserve CSV order)
        if members_filter is None:
            mem_sel = [int(m) for m in members]
        else:
            req = [int(x) for x in members_filter]
            mem_sel = [int(m) for m in members if int(m) in set(req)]
            if require_all_selected_members and (set(req) - set(mem_sel)):
                # some requested member is missing in this row -> skip
                continue

        # Align merged file paths to selected members
        m2i = {int(m): i for i, m in enumerate(members)}
        member_files = []
        for m in mem_sel:
            mi = m2i[m]
            member_files.append(_ensure_abs(mmerged[mi], root))

        out.append({
            "date": date if pd.notna(date) else None,
            "window": window if pd.notna(window) else None,
            "members": mem_sel,
            "lead_times": lead_times,
            "lead_indices": lead_indices,
            "member_files": member_files,
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
    Dataset for MEPS-like ensembles from MERGED files (one .npy per member with shape (T,C,H,W)).

    sample_mode="ensemble":
        x: (M, 2, C, H, W)
        y: (M, T-2, C, H, W)

    sample_mode="per_member":
        x: (2, C, H, W)
        y: (T-2, C, H, W)

    This class handles *all* user-facing config keys (root, sequences_csv, normalization, etc.).
    The DataModule only orchestrates splitting and DataLoaders.
    """

    def __init__(
        self,
        # Build records from CSV (preferred; matches Hydra YAML usage)
        root: Optional[str] = None,
        sequences_csv: Optional[str] = None,
        members: Optional[Sequence[int]] = None,
        leads: Optional[Sequence[int]] = None,
        require_internal_targets: bool = True,
        require_all_selected_members: bool = True,

        # Or pass records directly (advanced / internal use)
        records: Optional[List[Dict[str, Any]]] = None,

        # Dataset behavior
        sample_mode: Literal["ensemble", "per_member"] = "per_member",
        file_channel_indices: Sequence[int] = (0, 1, 2),

        # Normalization
        normalize: Literal["none", "zscore", "symrange"] = "none",
        stats_npz: Optional[str] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 1.0,

        # IO / dtype
        dtype: Literal["float32", "float16", "bfloat16", "float64"] = "float32",
        mmap: bool = True,

        # Loader-ish keys may exist in YAML; accept & ignore to avoid Hydra errors
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        shuffle_train: Optional[bool] = None,
        split: Optional[Dict[str, Any]] = None,
        **_
    ) -> None:
        super().__init__()

        # Build records if not provided
        if records is None:
            if not (root and sequences_csv):
                raise ValueError("MEPSWindowDataset: provide either `records` OR both `root` and `sequences_csv`.")
            records = read_grouped_ensemble_windows_merged(
                csv_path=sequences_csv,
                root=root,
                members_filter=members,
                leads_filter=leads,
                require_internal_targets=require_internal_targets,
                require_all_selected_members=require_all_selected_members,
            )
            if not records:
                raise ValueError(
                    "MEPSWindowDataset: 0 records after reading CSV/filters. "
                    "Check members/leads and that your CSV has Members, LeadTimes, MemberMerged."
                )
            if split is not None:
                rank_zero_info("MEPSWindowDataset: `split` provided in config is ignored by the Dataset. "
                               "Use the DataModule to perform splitting.")
        self.recs = records

        self.chan_idx = list(file_channel_indices)
        self.sample_mode = sample_mode
        self.mmap = mmap

        # Output dtype
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        self.out_dtype = self.dtype_map[dtype]

        # Normalizer configured for selected channels
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

        # Build per-member index for "per_member"
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

    def _load_merged_sel(self, path: str, time_indices: Sequence[int]) -> np.ndarray:
        """
        Load a merged .npy with shape (T,C,H,W), select time indices, then channels,
        and apply normalization per frame. Returns (T', C_sel, H, W).
        """
        arr = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)
        if arr.ndim != 4:
            raise ValueError(f"Expected merged (T,C,H,W), got {arr.shape} @ {path}")

        tsel = np.asarray(time_indices, dtype=int)
        a = arr[tsel, ...]  # (T', C, H, W)

        ci = np.asarray(self.chan_idx, dtype=int)
        if a.shape[1] <= int(ci.max()):
            raise ValueError(f"Channel index {int(ci.max())} out of range for shape {a.shape} @ {path}")
        a = a[:, ci, :, :]  # (T', C_sel, H, W)

        frames = [self.norm.apply(a[t]) for t in range(a.shape[0])]
        return np.stack(frames, axis=0)  # (T', C_sel, H, W)

    @staticmethod
    def _first_last_internals(tchw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        (T, C, H, W) -> (2, C, H, W), (T-2, C, H, W)
        """
        if tchw.shape[0] < 2:
            raise ValueError(f"Need at least 2 time steps, got {tchw.shape}")
        first = tchw[0]
        last = tchw[-1]
        internals = tchw[1:-1] if tchw.shape[0] > 2 else np.empty((0,) + tchw.shape[1:], dtype=tchw.dtype)
        x = np.stack([first, last], axis=0)
        y = internals
        return x, y

    # ---------- main sampling ----------

    def __getitem__(self, i: int):
        if self.sample_mode == "ensemble":
            rec = self.recs[i]
            members = rec["members"]
            lead_idx = rec["lead_indices"]
            files: List[str] = rec["member_files"]

            x_m: List[np.ndarray] = []
            y_m: List[np.ndarray] = []
            for f in files:
                tchw = self._load_merged_sel(f, lead_idx)
                x_tchw, y_tchw = self._first_last_internals(tchw)
                x_m.append(x_tchw)
                y_m.append(y_tchw)

            x = np.stack(x_m, axis=0)  # (M,2,C,H,W)
            y = np.stack(y_m, axis=0)  # (M,T-2,C,H,W)

            meta = {
                "sample_mode": "ensemble",
                "date": rec.get("date"),
                "window": rec.get("window"),
                "lead_times": rec.get("lead_times"),
                "members": members,
                "start_valid_time": rec.get("start_valid_time"),
                "end_valid_time": rec.get("end_valid_time"),
                "rec_index": int(i),
            }

            return (
                torch.as_tensor(x, dtype=self.out_dtype),
                torch.as_tensor(y, dtype=self.out_dtype),
                meta,
            )

        # per_member
        assert self._index is not None
        rec_idx, m_idx = self._index[i]
        rec = self.recs[rec_idx]

        lead_idx = rec["lead_indices"]
        files: List[str] = rec["member_files"]
        fpath = files[m_idx]

        tchw = self._load_merged_sel(fpath, lead_idx)
        x_tchw, y_tchw = self._first_last_internals(tchw)

        meta = {
            "sample_mode": "per_member",
            "date": rec.get("date"),
            "window": rec.get("window"),
            "lead_times": rec.get("lead_times"),
            "member_index": int(m_idx),
            "member": int(rec["members"][m_idx]) if isinstance(rec.get("members"), (list, tuple)) else int(m_idx),
            "n_members": int(len(rec["members"])) if isinstance(rec.get("members"), (list, tuple)) else None,
            "start_valid_time": rec.get("start_valid_time"),
            "end_valid_time": rec.get("end_valid_time"),
            "rec_index": int(rec_idx),
        }

        return (
            torch.as_tensor(x_tchw, dtype=self.out_dtype),
            torch.as_tensor(y_tchw, dtype=self.out_dtype),
            meta,
        )


# ============================================================
# Split config (ratio / count / none)
# ============================================================

@dataclass
class SplitConfig:
    """
    Split strategy when using the DataModule.

    Types:
      - "ratio": date-aware, BLOCKED split using ONLY 'test' ratio.
                 Remaining earliest days form the train+val pool, which is
                 allocated with alternating TRAIN/VAL blocks and skips.
      - "count": absolute counts (no date awareness)
      - "none":  no split (everything -> train)
    """
    type: Literal["ratio", "count", "none"] = "ratio"

    # --- Ratio split uses ONLY 'test' ---
    test: float = 0.1  # fraction of total unique days reserved at the END for test

    # --- Block parameters (in DAYS) for arranging train+val inside the remaining pool ---
    n_train_days_per_block: int = 25
    skip_after_train_days: int = 2
    n_val_days_per_block: int = 3
    skip_after_val_days: int = 2
    start_with: Literal["train", "val"] = "train"

    # --- Count split controls (unchanged) ---
    # For type="count", interpret train/val/test as absolute COUNTS
    train: int = 0
    val: int = 0
    seed: int = 42
    shuffle_before_split: bool = True


# ============================================================
# Lightning DataModule
# ============================================================

class MEPSNPYWindowDataModule(L.LightningDataModule):
    """
    DataModule wrapper that:
      1) Reads merged CSV -> records (Members, LeadTimes, MemberMerged),
      2) Applies the requested split (ratio/count/none),
      3) Instantiates `MEPSWindowDataset` for each split,
      4) Builds DataLoaders.

    NOTE: All dataset-specific knobs (root, sequences_csv, normalization, etc.) are
    handled by `MEPSWindowDataset`. This module takes the same keys and simply
    forwards them.
    """

    def __init__(
        self,
        # --- Dataset-facing keys (forwarded to MEPSWindowDataset) ---
        root: str = ".",
        sequences_csv: Optional[str] = None,   # single CSV; will be split by `split`
        # filtering
        members: Optional[Sequence[int]] = None,
        leads: Optional[Sequence[int]] = None,
        require_internal_targets: bool = True,
        require_all_selected_members: bool = True,
        # dataset behavior
        sample_mode: Literal["ensemble", "per_member"] = "per_member",
        file_channel_indices: Sequence[int] = (0, 1, 2),
        # normalization
        normalize: Literal["none", "zscore", "symrange"] = "symrange",
        stats_npz: Optional[str] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 0.95,
        # dtype / IO
        dtype: Literal["float32", "float16", "bfloat16", "float64"] = "float32",
        mmap: bool = True,

        # --- Loader ---
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        shuffle_train: bool = True,

        # --- Split ---
        split: Optional[Dict[str, Any]] = None,  # e.g. {type: ratio, test: 0.1, n_train_days_per_block: 25, ...}

    ) -> None:
        super().__init__()
        # Store
        self.root = root
        self.sequences_csv = sequences_csv

        self.members = None if members is None else list(members)
        self.leads = None if leads is None else list(leads)
        self.require_internal_targets = bool(require_internal_targets)
        self.require_all_selected_members = bool(require_all_selected_members)

        self.sample_mode = sample_mode
        self.file_channel_indices = list(file_channel_indices)

        self.normalize = normalize
        self.stats_npz = stats_npz
        self.mean_key = mean_key
        self.std_key = std_key
        self.average_key = average_key
        self.global_min_key = global_min_key
        self.global_max_key = global_max_key
        self.norm_const = norm_const

        self.dtype = dtype
        self.mmap = mmap

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if persistent_workers is not None else (num_workers > 0)
        self.shuffle_train = shuffle_train

        self.split_cfg = SplitConfig(**split) if split is not None else SplitConfig()

        # Datasets
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    # ---------- Lightning hooks ----------

    def prepare_data(self) -> None:
        # No downloads or preprocessing here; CSV and npy files are assumed to exist.
        pass

    # ---- Split helpers ----

    def _read_all_records(self) -> List[Dict[str, Any]]:
        if not self.sequences_csv:
            raise ValueError("Provide `sequences_csv` for MEPSNPYWindowDataModule.")
        return read_grouped_ensemble_windows_merged(
            csv_path=self.sequences_csv,
            root=self.root,
            members_filter=self.members,
            leads_filter=self.leads,
            require_internal_targets=self.require_internal_targets,
            require_all_selected_members=self.require_all_selected_members,
        )

    def _ratio_split_blocked(self, records: List[Dict[str, Any]]):
        """
        Date-aware BLOCKED ratio split using ONLY 'test' ratio.

        Steps:
          1) Map each record to a day (YYYY-MM-DD). If any records have dates,
             drop undated ones from splitting (cannot be ordered).
          2) Sort unique days ascending.
          3) test_days_target = floor(test * n_days_total)
          4) tv_days = earliest (n_days_total - test_days_target) days (train+val pool)
             test_days = the remaining tail days
          5) Inside tv_days, allocate alternating TRAIN and VAL blocks with configured
             sizes and skips until tv_days is exhausted.

        If no valid dates exist at all, we fall back to **index-based** blocked
        allocation: treat contiguous record indices as "pseudo-days" and apply
        the same block cadence; the last test_count = floor(test * N) records
        are assigned to test.
        """
        cfg = self.split_cfg

        # Collect normalized dates per record
        rec_dates: List[Optional[pd.Timestamp]] = []
        for r in records:
            d = r.get("date", None)
            ts = pd.to_datetime(d, errors="coerce") if d is not None else pd.NaT
            if pd.notna(ts):
                ts = pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)
            else:
                ts = pd.NaT
            rec_dates.append(ts)

        valid_idx = [i for i, ts in enumerate(rec_dates) if pd.notna(ts)]
        have_dates = len(valid_idx) > 0

        # Helper to assign records by day sets
        def assign_by_days(tv_days: List[pd.Timestamp], test_days: List[pd.Timestamp]):
            # Blocked allocation over tv_days
            p = 0
            tv_train_days: List[pd.Timestamp] = []
            tv_val_days: List[pd.Timestamp] = []
            turn = cfg.start_with

            n_tr = max(0, int(cfg.n_train_days_per_block))
            n_va = max(0, int(cfg.n_val_days_per_block))
            s_tr = max(0, int(cfg.skip_after_train_days))
            s_va = max(0, int(cfg.skip_after_val_days))

            if n_tr == 0 and n_va == 0:
                # No block sizes -> everything in tv_days goes to train (no val)
                tv_train_days = list(tv_days)
            else:
                while p < len(tv_days):
                    if turn == "train":
                        if n_tr > 0:
                            take = min(n_tr, len(tv_days) - p)
                            tv_train_days.extend(tv_days[p:p + take])
                            p += take
                            skip = min(s_tr, len(tv_days) - p)
                            p += skip
                        turn = "val"
                    else:  # val
                        if n_va > 0:
                            take = min(n_va, len(tv_days) - p)
                            tv_val_days.extend(tv_days[p:p + take])
                            p += take
                            skip = min(s_va, len(tv_days) - p)
                            p += skip
                        turn = "train"

            # Build assignment map
            day_to_split: Dict[pd.Timestamp, str] = {}
            for d in tv_train_days:
                day_to_split[d] = "train"
            for d in tv_val_days:
                if d not in day_to_split:
                    day_to_split[d] = "val"
            for d in test_days:
                day_to_split[d] = "test"

            # Collect records
            rec_train: List[Dict[str, Any]] = []
            rec_val: List[Dict[str, Any]] = []
            rec_test: List[Dict[str, Any]] = []

            # Iterate only over dated records
            for i in valid_idx:
                day = rec_dates[i]
                tag = day_to_split.get(day, None)
                if tag == "train":
                    rec_train.append(records[i])
                elif tag == "val":
                    rec_val.append(records[i])
                elif tag == "test":
                    rec_test.append(records[i])
            return rec_train, rec_val, rec_test

        if have_dates:
            # Unique sorted days
            days = sorted({rec_dates[i] for i in valid_idx})
            n_days_total = len(days)
            if n_days_total == 0:
                return records, [], []

            test_days_target = int(np.floor(float(cfg.test) * n_days_total))
            test_days_target = max(0, min(test_days_target, n_days_total))

            # Split days
            tv_days = days[: n_days_total - test_days_target]
            test_days = days[n_days_total - test_days_target + cfg.skip_after_val_days :]

            rec_train, rec_val, rec_test = assign_by_days(tv_days, test_days)
            return rec_train, rec_val, rec_test

        # ---- Fallback: no valid dates -> index-based blocked allocation ----
        N = len(records)
        test_count = int(np.floor(float(cfg.test) * N))
        test_count = max(0, min(test_count, N))

        tv_end = N - test_count  # exclusive
        tv_indices = list(range(0, tv_end))
        test_indices = list(range(tv_end, N))

        # Apply the same cadence over indices
        p = 0
        i_train: List[int] = []
        i_val: List[int] = []
        turn = cfg.start_with

        n_tr = max(0, int(cfg.n_train_days_per_block))
        n_va = max(0, int(cfg.n_val_days_per_block))
        s_tr = max(0, int(cfg.skip_after_train_days))
        s_va = max(0, int(cfg.skip_after_val_days))

        if n_tr == 0 and n_va == 0:
            i_train = tv_indices[:]  # everything to train
        else:
            while p < len(tv_indices):
                if turn == "train":
                    if n_tr > 0:
                        take = min(n_tr, len(tv_indices) - p)
                        i_train.extend(tv_indices[p:p + take])
                        p += take
                        skip = min(s_tr, len(tv_indices) - p)
                        p += skip
                    turn = "val"
                else:
                    if n_va > 0:
                        take = min(n_va, len(tv_indices) - p)
                        i_val.extend(tv_indices[p:p + take])
                        p += take
                        skip = min(s_va, len(tv_indices) - p)
                        p += skip
                    turn = "train"

        def take(ids: List[int]) -> List[Dict[str, Any]]:
            return [records[i] for i in ids]

        rec_train = take(i_train)
        rec_val   = take(i_val)
        rec_test  = take(test_indices)
        return rec_train, rec_val, rec_test

    def _count_split(self, records: List[Dict[str, Any]]):
        cfg = self.split_cfg
        n = len(records)
        idx = list(range(n))

        if cfg.shuffle_before_split:
            g = torch.Generator().manual_seed(cfg.seed)
            idx = torch.randperm(n, generator=g).tolist()

        if cfg.type == "none":
            return records, [], []

        # Interpret train/val as absolute counts
        n_train, n_val = int(cfg.train), int(cfg.val)
        if n_train + n_val > n:
            raise ValueError(f"Split counts exceed dataset size ({n}).")
        n_test = n - n_train - n_val

        def take(ids: List[int]) -> List[Dict[str, Any]]:
            return [records[i] for i in ids]

        i_tr = idx[:n_train]
        i_va = idx[n_train:n_train + n_val]
        i_te = idx[n_train + n_val:n_train + n_val + n_test]
        return take(i_tr), take(i_va), take(i_te)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate train/val/test datasets from a single CSV using the requested split.
        """
        all_rec = self._read_all_records()

        if self.split_cfg.type == "ratio":
            rec_train, rec_val, rec_test = self._ratio_split_blocked(all_rec)
        elif self.split_cfg.type == "count":
            rec_train, rec_val, rec_test = self._count_split(all_rec)
        elif self.split_cfg.type == "none":
            rec_train, rec_val, rec_test = all_rec, [], []
        else:
            raise ValueError(f"Unknown split.type='{self.split_cfg.type}'")

        def mk(recs: List[Dict[str, Any]]) -> Optional[MEPSWindowDataset]:
            if not recs:
                return None
            return MEPSWindowDataset(
                # records path (dataset handles config keys)
                records=recs,
                sample_mode=self.sample_mode,
                file_channel_indices=self.file_channel_indices,
                normalize=self.normalize,
                stats_npz=self.stats_npz,
                mean_key=self.mean_key,
                std_key=self.std_key,
                average_key=self.average_key,
                global_min_key=self.global_min_key,
                global_max_key=self.global_max_key,
                norm_const=self.norm_const,
                dtype=self.dtype,
                mmap=self.mmap,
            )

        self.train_set = mk(rec_train)
        self.val_set = mk(rec_val)
        self.test_set = mk(rec_test)

        train_size = len(self.train_set) if self.train_set else 0
        val_size = len(self.val_set) if self.val_set else 0
        test_size = len(self.test_set) if self.test_set else 0
        rank_zero_info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # ---------- DataLoaders ----------

    def _loader(self, ds: Optional[Dataset], shuffle: bool) -> DataLoader:
        if ds is None:
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
