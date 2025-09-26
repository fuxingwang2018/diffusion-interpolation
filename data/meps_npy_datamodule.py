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
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info

 
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
    Read a grouped sequences CSV produced by the *merged* pipeline.

    Required JSON columns:
      - Members:       list[int]
      - LeadTimes:     list[int]     (the full set stored in the merged file, in order)
      - MemberMerged:  list[str]     (one merged path per member; shape (M,))

    Optional passthrough columns: Date, Window, StartValidTime, EndValidTime.

    Returns a list of records per (Date, Window):
      {
        "date": str | None,
        "window": str | None,
        "members": List[int],             # possibly filtered, CSV order preserved
        "lead_times": List[int],          # filtered & sorted ascending
        "lead_indices": List[int],        # indices into time axis of merged array
        "member_files": List[str],        # one merged file path per selected member
        "start_valid_time": Any | None,
        "end_valid_time": Any | None,
      }
    """
    df = pd.read_csv(csv_path)
    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        members = row.get("Members")
        leads = row.get("LeadTimes")
        mmerged = row.get("MemberMerged")

        # Optional columns (pass-through)
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
            continue  # malformed row

        # Original lead order (as in merged file time axis)
        leads_all = [int(x) for x in leads]

        # Apply optional lead filter -> indices into leads_all
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

        # Build index mapping: lead -> position in merged file time axis
        lt2i = {lt: i for i, lt in enumerate(leads_all)}
        lead_indices = [lt2i[lt] for lt in lead_times]  # positions we’ll slice

        # Filter members (preserve CSV order)
        if members_filter is None:
            mem_sel = [int(m) for m in members]
        else:
            req = [int(x) for x in members_filter]
            mem_sel = [int(m) for m in members if int(m) in set(req)]
            if require_all_selected_members and (set(req) - set(mem_sel)):
                continue  # some requested member missing

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
            "lead_times": lead_times,         # filtered set (ascending b/c we kept order)
            "lead_indices": lead_indices,     # indices into time axis of merged array
            "member_files": member_files,     # one merged file per selected member
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
    Create (x, y, meta) samples from grouped ensemble windows, loading from *merged* files.

    sample_mode="ensemble":
        x: (M, 2, C, H, W)   -> first & last frames in the selected lead_times
        y: (M, T-2, C, H, W) -> internal frames

    sample_mode="per_member":
        x: (2, C, H, W)
        y: (T-2, C, H, W)
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        file_channel_indices: Sequence[int] = (0, 1, 2, 3),
        sample_mode: Literal["ensemble", "per_member"] = "ensemble",
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

    def _load_merged_sel(self, path: str, time_indices: Sequence[int]) -> np.ndarray:
        """
        Load a merged .npy and return (T', C, H, W) for selected time indices,
        then select channels -> (T', C_sel, H, W).

        Assumes merged shape is (T, C, H, W). If you used a different merge axis,
        adjust the indexing/moveaxis accordingly.
        """
        arr = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)
        if arr.ndim != 4:
            raise ValueError(f"Expected merged (T,C,H,W), got {arr.shape} @ {path}")

        # subset time
        tsel = np.asarray(time_indices, dtype=int)
        a = arr[tsel, ...]  # (T', C, H, W)

        # select channels
        ci = np.asarray(self.chan_idx, dtype=int)
        if a.shape[1] <= int(ci.max()):
            raise ValueError(f"Channel index out of range for shape {a.shape} @ {path}")
        a = a[:, ci, :, :]  # (T', C_sel, H, W)

        # normalize per frame (Normalizer expects (C,H,W))
        frames = []
        for t in range(a.shape[0]):
            frames.append(self.norm.apply(a[t]))
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
            date = rec["date"]
            window = rec["window"]
            members = rec["members"]                       # List[int]
            lead_ts = rec["lead_times"]                    # List[int] (filtered)
            lead_idx = rec["lead_indices"]                 # List[int] (positions in merged)
            files: List[str] = rec["member_files"]         # List[str], one per member
            start_valid_time = rec["start_valid_time"]
            end_valid_time = rec["end_valid_time"]

            # Build x=(M,2,C,H,W), y=(M,T-2,C,H,W)
            x_m: List[np.ndarray] = []
            y_m: List[np.ndarray] = []
            for f in files:
                tchw = self._load_merged_sel(f, lead_idx)  # (T',C,H,W)
                x_tchw, y_tchw = self._first_last_internals(tchw)
                x_m.append(x_tchw)
                y_m.append(y_tchw)

            x = np.stack(x_m, axis=0)  # (M,2,C,H,W)
            y = np.stack(y_m, axis=0)  # (M,T-2,C,H,W)

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
        lead_idx = rec["lead_indices"]                # List[int]
        files: List[str] = rec["member_files"]
        fpath = files[m_idx]

        tchw = self._load_merged_sel(fpath, lead_idx)     # (T',C,H,W)
        x_tchw, y_tchw = self._first_last_internals(tchw) # -> (2,C,H,W), (T-2,C,H,W)
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
