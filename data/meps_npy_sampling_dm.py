from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """Return absolute path string for `p`, resolved under `root` if `p` is relative."""
    p = Path(p)
    return str(p if p.is_absolute() else Path(root) / p)

def _parse_json_maybe(x):
    """Parse JSON if `x` is a string; otherwise return as-is."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def _to_isostr(v) -> Optional[str]:
    """Convert pandas/NumPy/py datetime-like to ISO 8601 string, else None or original str."""
    if v is None:
        return None
    # Already a string
    if isinstance(v, str):
        return v
    # Pandas Timestamp / NaT
    try:
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        # keep timezone info; Pandas uses UTC by default here
        return ts.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(v)


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
        mode: str = "none",                       # "none" | "zscore" | "symrange"
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
                # allow 1D (C,), 3D (C,H,W)
                if arr.ndim == 1 and arr.shape[0] >= max(self.chan_idx) + 1:
                    return arr[self.chan_idx]
                if arr.ndim == 3 and arr.shape[0] >= max(self.chan_idx) + 1:
                    return arr[self.chan_idx, ...]
                return arr  # fallback
            if self.mode == "zscore":
                self.arr_mean = _maybe_slice(stats[mean_key])
                self.arr_std = _maybe_slice(stats[std_key])
            elif self.mode == "symrange":
                self.arr_avg  = _maybe_slice(stats[average_key])
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
        """
        x: (C, H, W) numpy array
        """
        if self.mode == "none":
            return x
        if self.mode == "zscore":
            mean = self._bcast(x, np.asarray(self.arr_mean))
            std  = self._bcast(x, np.asarray(self.arr_std))
            std  = np.where(std == 0, 1.0, std)
            return (x - mean) / std
        if self.mode == "symrange":
            avg  = self._bcast(x, np.asarray(self.arr_avg))
            gmin = self._bcast(x, np.asarray(self.arr_gmin))
            gmax = self._bcast(x, np.asarray(self.arr_gmax))
            denom = np.maximum(np.abs(gmin - avg), np.abs(gmax - avg))
            denom = np.where(denom == 0, 1.0, denom)
            res = self.norm_const * (x - avg) / denom
            res_t = torch.as_tensor(res, dtype=torch.float32, device="cpu")
            return res_t.clamp(-self.norm_const, self.norm_const)
        raise ValueError(f"Unknown normalize mode: {self.mode}")


# ============================================================
# CSV → records (sorted & date-filtered)
# ============================================================

def read_grouped_ensemble_windows_merged(
    csv_path: Union[str, Path],
    root: Union[str, Path],
    members_filter: Optional[Iterable[int]] = None,
    leads_filter: Optional[Iterable[int]] = None,
    require_two_endpoints: bool = True,
    date_range: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Read a grouped sequences CSV where each member has ONE merged .npy with shape (T,C,H,W).
    """
    df = pd.read_csv(csv_path)

    # Ensure there's a sortable datetime column; prefer "Date"
    time_col = None
    for cand in ("Date", "StartValidTime"):
        if cand in df.columns:
            time_col = cand
            break

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.sort_values(by=time_col, ascending=True, kind="mergesort").reset_index(drop=True)

    # Date filtering
    if date_range is not None:
        if time_col is None:
            raise ValueError("CSV has no 'Date' or 'StartValidTime' column to filter by date_range.")
        start_ts = pd.to_datetime(date_range[0], utc=True)
        end_ts   = pd.to_datetime(date_range[1], utc=True)
        mask = (df[time_col].notna()) & (df[time_col] >= start_ts) & (df[time_col] <= end_ts)
        df = df.loc[mask].reset_index(drop=True)

    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        members = _parse_json_maybe(row.get("Members"))
        leads    = _parse_json_maybe(row.get("LeadTimes"))
        merged   = _parse_json_maybe(row.get("MemberMerged"))

        if not isinstance(members, (list, tuple)) or not isinstance(leads, (list, tuple)) or not isinstance(merged, (list, tuple)):
            continue

        if len(members) != len(merged):
            continue

        # Lead-time filtering (preserve order)
        leads_all = [int(x) for x in leads]
        if leads_filter is not None:
            keep = set(int(x) for x in leads_filter)
            lead_times = [lt for lt in leads_all if lt in keep]
        else:
            lead_times = list(leads_all)

        if require_two_endpoints and len(lead_times) < 2:
            continue

        # Indices of first/last *within the merged array's time axis*
        lt2i = {lt: i for i, lt in enumerate(leads_all)}
        t_idx_first = lt2i[lead_times[0]]
        t_idx_last  = lt2i[lead_times[-1]]

        # Members filtering
        if members_filter is None:
            mem_sel = [int(m) for m in members]
        else:
            req = {int(x) for x in members_filter}
            mem_sel = [int(m) for m in members if int(m) in req]
            if not mem_sel:
                continue

        # Paths aligned to selected members
        m2i = {int(m): i for i, m in enumerate(members)}
        member_files = [_ensure_abs(merged[m2i[m]], root) for m in mem_sel]

        # Convert any times to ISO strings to keep collate happy
        date_s            = _to_isostr(row.get("Date", None))
        start_valid_time_s = _to_isostr(row.get("StartValidTime", None))
        end_valid_time_s   = _to_isostr(row.get("EndValidTime", None))

        out.append({
            "date": date_s,
            "window": row.get("Window", None),
            "start_valid_time": start_valid_time_s,
            "end_valid_time": end_valid_time_s,
            "members": mem_sel,
            "member_files": member_files,
            "lead_times": lead_times,
            "t_idx_first": int(t_idx_first),
            "t_idx_last": int(t_idx_last),
        })

    return out


# ============================================================
# Dataset (per-member; endpoints only; returns x, meta) + normalization
# ============================================================

class MEPSSamplingWindowDataset(Dataset):
    """
    Per-member dataset that loads ONLY the endpoints (first & last) for each record.
    Returns (x, meta) with x = (2, C, H, W).
    """

    def __init__(
        self,
        root: str,
        sequences_csv: str,
        members: Optional[Sequence[int]] = None,
        leads: Optional[Sequence[int]] = None,
        file_channel_indices: Sequence[int] = (0, 1, 2, 3),
        dtype: str = "float32",
        mmap: bool = True,
        date_range: Optional[Tuple[str, str]] = None,

        # Normalization
        normalize: str = "none",                 # "none" | "zscore" | "symrange"
        stats_npz: Optional[Union[str, Path]] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 1.0,
    ) -> None:
        super().__init__()

        self.recs = read_grouped_ensemble_windows_merged(
            csv_path=sequences_csv,
            root=root,
            members_filter=members,
            leads_filter=leads,
            require_two_endpoints=True,
            date_range=date_range,
        )
        if not self.recs:
            raise ValueError("MEPSSamplingWindowDataset: 0 records after reading CSV/filters/date_range.")

        # Flat per-member index
        self.index: List[Tuple[int, int]] = []
        for r_i, rec in enumerate(self.recs):
            M = len(rec["members"])
            self.index.extend((r_i, m_i) for m_i in range(M))

        # IO / dtype / channels
        self.chan_idx = np.asarray(list(file_channel_indices), dtype=int)
        self.mmap = bool(mmap)
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        self.out_dtype = self.dtype_map[dtype]

        # Normalizer for selected channels
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

    def __len__(self) -> int:
        return len(self.index)

    def _load_endpoints(self, path: str, t_first: int, t_last: int) -> np.ndarray:
        """
        Load ONLY two frames (first & last) from (T,C,H,W), select channels,
        and apply normalization per frame. Returns (2, C_sel, H, W).
        """
        arr = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)
        if arr.ndim != 4:
            raise ValueError(f"Expected (T,C,H,W), got {arr.shape} @ {path}")

        two = arr[[t_first, t_last], :, :, :]        # (2, C, H, W)
        if two.shape[1] <= int(self.chan_idx.max()):
            raise ValueError(f"Channel index {int(self.chan_idx.max())} out of range for shape {two.shape} @ {path}")
        two = two[:, self.chan_idx, :, :]           # (2, C_sel, H, W)

        if self.norm.mode != "none":
            frames = [self.norm.apply(two[t]) for t in range(two.shape[0])]
            two = np.stack(frames, axis=0)

        return two

    def __getitem__(self, i: int):
        rec_idx, m_idx = self.index[i]
        rec = self.recs[rec_idx]

        fpath = rec["member_files"][m_idx]
        x_2chw = self._load_endpoints(
            path=fpath,
            t_first=rec["t_idx_first"],
            t_last=rec["t_idx_last"],
        )

        meta = {
            "date": rec.get("date"),                       # already ISO string or None
            "window": rec.get("window"),
            "lead_times": rec.get("lead_times"),
            "member_index": int(m_idx),
            "member": int(rec["members"][m_idx]),
            "n_members": int(len(rec["members"])),
            "start_valid_time": rec.get("start_valid_time"),  # ISO string or None
            "end_valid_time": rec.get("end_valid_time"),      # ISO string or None
            "rec_index": int(rec_idx),
            "sample_index": int(i),
            "path": fpath,
        }

        x = torch.as_tensor(x_2chw, dtype=self.out_dtype)  # (2, C, H, W)
        return x, meta


# ============================================================
# Lightning DataModule (single stream; no split; no shuffle)
# ============================================================

class MEPSNPYSamplingWindowDataModule(L.LightningDataModule):
    """
    DataModule that exposes a single dataset/dataloader for train/val/test hooks.
    - Control the time interval via `date_range=(start, end)` (inclusive).
    - No shuffle.
    - Each batch yields (x, meta) with x: (B, 2, C, H, W).
    """

    def __init__(
        self,
        root: str,
        sequences_csv: str,
        members: Optional[Sequence[int]] = None,
        leads: Optional[Sequence[int]] = None,
        file_channel_indices: Sequence[int] = (0, 1, 2, 3),
        dtype: str = "float32",
        mmap: bool = True,
        date_range: Optional[Tuple[str, str]] = None,

        # Normalization
        normalize: str = "none",                 # "none" | "zscore" | "symrange"
        stats_npz: Optional[Union[str, Path]] = None,
        mean_key: str = "mean",
        std_key: str = "std",
        average_key: str = "average",
        global_min_key: str = "global_min",
        global_max_key: str = "global_max",
        norm_const: float = 1.0,

        # Loader
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.sequences_csv = sequences_csv
        self.members = members
        self.leads = leads
        self.file_channel_indices = file_channel_indices
        self.dtype = dtype
        self.mmap = mmap
        self.date_range = date_range

        # Normalization config
        self.normalize = normalize
        self.stats_npz = stats_npz
        self.mean_key = mean_key
        self.std_key = std_key
        self.average_key = average_key
        self.global_min_key = global_min_key
        self.global_max_key = global_max_key
        self.norm_const = norm_const

        # Loader
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass  # nothing to download

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = MEPSSamplingWindowDataset(
            root=self.root,
            sequences_csv=self.sequences_csv,
            members=self.members,
            leads=self.leads,
            file_channel_indices=self.file_channel_indices,
            dtype=self.dtype,
            mmap=self.mmap,
            date_range=self.date_range,
            normalize=self.normalize,
            stats_npz=self.stats_npz,
            mean_key=self.mean_key,
            std_key=self.std_key,
            average_key=self.average_key,
            global_min_key=self.global_min_key,
            global_max_key=self.global_max_key,
            norm_const=self.norm_const,
        )
        rank_zero_info(f"Dataset size = {len(self.dataset)} samples")

    def _loader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader.")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,                # ← NO SHUFFLE
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
        )

    # Same loader for all hooks (you control the range via date_range in config)
    def train_dataloader(self) -> DataLoader: return self._loader()
    def val_dataloader(self) -> DataLoader:   return self._loader()
    def test_dataloader(self) -> DataLoader:  return self._loader()
