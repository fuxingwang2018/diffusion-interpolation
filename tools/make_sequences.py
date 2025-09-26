#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any, Tuple

import os
os.environ["TZ"] = "UTC"

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    start: int
    end: int  # inclusive

    @classmethod
    def parse_many(cls, spec: str) -> List["Window"]:
        out: List[Window] = []
        for chunk in (s.strip() for s in spec.split(",")):
            if not chunk:
                continue
            a, b = chunk.split("-", 1)
            s, e = int(a), int(b)
            if e < s:
                raise ValueError(f"Invalid window '{chunk}': end < start")
            out.append(Window(s, e))
        return out

    def required_leads(self) -> List[int]:
        return list(range(self.start, self.end + 1))

    def label(self) -> str:
        return f"{self.start}-{self.end}"


def _resolve_path(name: str, root: Path) -> Path:
    p = Path(name)
    return p if p.is_absolute() else (root / p)


def _parse_iso_utc(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s.strip(), utc=True, errors="raise")


def load_labels(
    labels_csv: Path,
    root: Path,
    verify_fs: bool,
    members: Optional[Iterable[int]] = None,
    date_min: Optional[pd.Timestamp] = None,
    date_max: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Load labels and normalize:
      - Accepts 'Date' as the base-time column.
      - Produces columns: Date (UTC tz-aware), LeadTime[int], Member[int], AbsPath[str], Exists[bool]
      - Filters by date range and member selection.
    """
    df = pd.read_csv(labels_csv)

    date_col = "Date"
    req = {"Name", date_col, "LeadTime", "Member"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if df["Date"].isna().any():
        bad = df[df["Date"].isna()]
        raise ValueError(f"Found unparseable {date_col} values:\n{bad[date_col]}")

    df["LeadTime"] = df["LeadTime"].astype(int)
    df["Member"] = df["Member"].astype(int)

    if members is not None:
        member_set = set(int(m) for m in members)
        df = df[df["Member"].isin(member_set)]

    if date_min is not None:
        df = df[df["Date"] >= date_min]
    if date_max is not None:
        df = df[df["Date"] <= date_max]

    # Resolve paths + existence
    abs_paths: List[Path] = []
    exists: List[bool] = []
    for name in df["Name"]:
        p = _resolve_path(str(name), root)
        abs_paths.append(p)
        exists.append(p.exists() if verify_fs else True)
    df["AbsPath"] = [str(p) for p in abs_paths]
    df["Exists"] = exists

    # Only keep existing if verifying
    if verify_fs:
        df = df[df["Exists"]].copy()

    # Deduplicate within a cycle
    df.sort_values(["Date", "Member", "LeadTime"], inplace=True)
    df.drop_duplicates(subset=["Date", "Member", "LeadTime"], keep="first", inplace=True)

    return df


def build_cycle_index(df: pd.DataFrame) -> Dict[pd.Timestamp, Dict[int, Dict[int, str]]]:
    """
    Index: cycle -> member -> leadtime -> path
    """
    index: Dict[pd.Timestamp, Dict[int, Dict[int, str]]] = {}
    for (date, member), grp in df.groupby(["Date", "Member"], sort=True):
        lead_to_path = dict(zip(grp["LeadTime"].astype(int), grp["AbsPath"]))
        index.setdefault(date, {})[int(member)] = lead_to_path
    return index


def find_sequences_all_members(
    cycle_index: Dict[pd.Timestamp, Dict[int, Dict[int, str]]],
    windows: List[Window],
    selected_members: List[int],
    dates_sorted: List[pd.Timestamp],
) -> pd.DataFrame:
    """
    For each (Date, Window), require every member to have all required leads.
    Emit one row with aligned lists of member->leadtime paths.
    """
    records: List[Dict[str, Any]] = []
    for date in dates_sorted:
        member_map = cycle_index.get(date, {})
        if not all(m in member_map for m in selected_members):
            continue

        for w in windows:
            req = w.required_leads()
            ok = True
            per_member_paths: List[List[str]] = []
            for m in selected_members:
                lt2p = member_map[m]
                if not all(lt in lt2p for lt in req):
                    ok = False
                    break
                per_member_paths.append([lt2p[lt] for lt in req])
            if not ok:
                continue

            start_valid = date + pd.to_timedelta(min(req), unit="h")
            end_valid = date + pd.to_timedelta(max(req), unit="h")
            records.append(
                {
                    "Date": date,  # UTC tz-aware
                    "Window": w.label(),
                    "LeadTimes": req,                 # list[int]
                    "Members": selected_members,      # list[int]
                    "MemberPaths": per_member_paths,  # list[list[str]] aligned with Members
                    "NMembers": len(selected_members),
                    "NFilesPerMember": len(req),
                    "TotalFiles": len(req) * len(selected_members),
                    "StartValidTime": start_valid,
                    "EndValidTime": end_valid,
                }
            )
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out.sort_values(["Date", "Window"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


# ---------------------- NEW: materialization helpers ----------------------
def _cycle_str(dt: pd.Timestamp) -> str:
    # YYYYMMDDHH based on UTC
    dt = pd.to_datetime(dt, utc=True)
    return dt.strftime("%Y%m%d%H")


def _merge_target_path(merge_root: Path, dt: pd.Timestamp, window: str, member: int) -> Path:
    # organize per cycle under merge_root/YYYY/MM/DD/HH/
    yyyy = dt.strftime("%Y")
    mm = dt.strftime("%m")
    dd = dt.strftime("%d")
    hh = dt.strftime("%H")
    base = f"{_cycle_str(dt)}_w{window}_mem{member:03d}.npy"
    return merge_root / yyyy / mm / dd / hh / base


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(path, arr)

def _format_window_2d(win: str) -> str:
    """Convert 'a-b' to two-digit 'aa-bb' (e.g., '0-6' -> '00-06')."""
    a, b = (s.strip() for s in win.split("-", 1))
    return f"{int(a):02d}-{int(b):02d}"

def merge_member_leads_for_row(
    row: pd.Series,
    merge_root: Path,
    merge_axis: int = 0,
    overwrite: bool = False,
) -> List[str]:
    """
    For one (Date, Window) row, load each member's per-lead files, stack along `merge_axis`,
    validate shapes, and write one .npy per member. Return list of merged file paths
    aligned with `row["Members"]`.
    """
    members: List[int] = row["Members"]
    member_paths: List[List[str]] = row["MemberPaths"]
    leadtimes: List[int] = row["LeadTimes"]
    date_ts: pd.Timestamp = row["Date"]
    window_lbl: str = _format_window_2d(row["Window"])

    merged_paths: List[str] = []
    for m, files in zip(members, member_paths):
        target = _merge_target_path(merge_root, date_ts, window_lbl, m)
        if target.exists() and not overwrite:
            merged_paths.append(str(target))
            continue

        # Load all lead arrays
        arrays: List[np.ndarray] = []
        ref_shape: Optional[Tuple[int, ...]] = None
        ref_dtype: Optional[np.dtype] = None
        for fp in files:
            a = np.load(fp, mmap_mode=None)
            if ref_shape is None:
                ref_shape = a.shape
                ref_dtype = a.dtype
            else:
                if a.shape != ref_shape:
                    raise ValueError(
                        f"Shape mismatch while merging Date={date_ts} Window={window_lbl} Member={m} "
                        f"LeadTimes={leadtimes}: expected {ref_shape}, got {a.shape} for {fp}"
                    )
                if a.dtype != ref_dtype:
                    # allow safe upcast
                    ref_dtype = np.result_type(ref_dtype, a.dtype)
            arrays.append(a.astype(ref_dtype, copy=False))

        # Stack along merge_axis (time dimension)
        merged = np.stack(arrays, axis=merge_axis)
        _atomic_save_npy(target, merged)
        merged_paths.append(str(target))

    return merged_paths


def materialize_merged_files(
    seq_df: pd.DataFrame,
    merge_root: Optional[Path],
    merge_axis: int,
    overwrite: bool,
) -> pd.DataFrame:
    """
    If merge_root is provided, create merged files and return a copy of seq_df
    with 'MemberMerged' instead of 'MemberPaths', and with counts updated.
    If merge_root is None, returns seq_df unchanged.
    """
    if merge_root is None:
        return seq_df

    if seq_df.empty:
        return seq_df

    out_rows: List[Dict[str, Any]] = []
    for _, row in seq_df.iterrows():
        merged = merge_member_leads_for_row(
            row=row,
            merge_root=merge_root,
            merge_axis=merge_axis,
            overwrite=overwrite,
        )
        new_row = row.to_dict()
        new_row["MemberMerged"] = merged  # list[str], one per member
        new_row["NFilesPerMember"] = 1
        new_row["TotalFiles"] = new_row["NMembers"]  # 1 per member
        # Keep old MemberPaths as reference if you like, or drop it:
        del new_row["MemberPaths"]
        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    out_df.sort_values(["Date", "Window"], inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    return out_df
# -------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Find lead-time windows present for ALL selected members, cycle-first.")
    ap.add_argument("--labels", type=Path, required=True, help="Path to labels.csv")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root to prepend to relative Name paths")
    ap.add_argument("--windows", type=str, default="0-6,6-12,12-18", help="Comma-separated inclusive windows")
    ap.add_argument("--members", type=str, default=None, help="Comma-separated member IDs to require (e.g. 0,1,2)")
    ap.add_argument("--no-verify-fs", action="store_true", help="Skip filesystem existence checks")
    ap.add_argument("--start-date", type=str, default=None, help="ISO-8601 lower bound (UTC)")
    ap.add_argument("--end-date", type=str, default=None, help="ISO-8601 upper bound (UTC)")
    ap.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    ap.add_argument("--extra-out", type=Path, default=None,
                    help="Optional path to write a flat file list CSV (column: file-list)")

    # NEW: merging options
    ap.add_argument("--merge-root", type=Path, default=None,
                    help="If set, write one merged .npy per member (per Date×Window) under this root.")
    ap.add_argument("--merge-axis", type=int, default=0,
                    help="Axis to stack lead times on in the merged array (default: 0, i.e., time-first).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing merged .npy files if they already exist.")

    args = ap.parse_args()

    windows = Window.parse_many(args.windows)
    windows = sorted(windows, key=lambda w: (w.start, w.end))

    members_list: Optional[List[int]] = None
    if args.members:
        members_list = [int(x) for x in args.members.split(",") if x.strip() != ""]

    date_min = _parse_iso_utc(args.start_date)
    date_max = _parse_iso_utc(args.end_date)

    df = load_labels(
        labels_csv=args.labels,
        root=args.root,
        verify_fs=not args.no_verify_fs,
        members=members_list,
        date_min=date_min,
        date_max=date_max,
    )

    selected_members = sorted(df["Member"].unique().tolist()) if members_list is None else sorted(members_list)

    cycle_index = build_cycle_index(df)
    dates_sorted = sorted(cycle_index.keys())

    seq_df = find_sequences_all_members(
        cycle_index=cycle_index,
        windows=windows,
        selected_members=selected_members,
        dates_sorted=dates_sorted,
    )

    # --- NEW: materialize merged files if requested ---
    merged_df = materialize_merged_files(
        seq_df=seq_df,
        merge_root=args.merge_root,
        merge_axis=args.merge_axis,
        overwrite=args.overwrite,
    )

    # Summary
    if merged_df.empty:
        print("No complete (Date, Window) sequences found for ALL selected members.")
    else:
        ndates = merged_df["Date"].nunique()
        print(
            f"Found {len(merged_df)} (Date, Window) sequences across {ndates} cycles "
            f"for {len(selected_members)} members."
        )
        summary_cols = ["Date", "Window", "NMembers", "NFilesPerMember", "StartValidTime", "EndValidTime"]
        with pd.option_context("display.max_colwidth", 80):
            print(merged_df.head(10)[summary_cols])

    # CSV out (robust to empty)
    if args.out:
        if merged_df.empty:
            # include new schema with MemberMerged instead of MemberPaths
            cols = [
                "Date", "Window", "LeadTimes", "Members", "MemberMerged",
                "NMembers", "NFilesPerMember", "TotalFiles", "StartValidTime", "EndValidTime"
            ]
            pd.DataFrame(columns=cols).to_csv(args.out, index=False)
            print(f"Wrote 0 sequences (header only) to {args.out}")
        else:
            out_df = merged_df.copy()
            # ISO-8601 for timestamps
            for col in ["Date", "StartValidTime", "EndValidTime"]:
                out_df[col] = pd.to_datetime(out_df[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            # JSON-encode structured columns
            out_df["LeadTimes"] = out_df["LeadTimes"].apply(lambda x: json.dumps(x))
            out_df["Members"] = out_df["Members"].apply(lambda x: json.dumps(x))
            out_df["MemberMerged"] = out_df["MemberMerged"].apply(lambda x: json.dumps(x))
            # ensure consistent column order
            out_cols = [
                "Date", "Window", "LeadTimes", "Members", "MemberMerged",
                "NMembers", "NFilesPerMember", "TotalFiles", "StartValidTime", "EndValidTime"
            ]
            out_df = out_df[out_cols]
            out_df.to_csv(args.out, index=False)
            print(f"Wrote {len(out_df)} sequences to {args.out}")

    # Extra-out (flat list) → now writes the merged file list
    if args.extra_out:
        if merged_df.empty:
            pd.DataFrame(columns=["file-list"]).to_csv(args.extra_out, index=False)
            print(f"Wrote 0 files (header only) to {args.extra_out}")
        else:
            all_files: List[str] = []
            for merged_per_member in merged_df["MemberMerged"]:
                all_files.extend(merged_per_member)
            pd.DataFrame({"file-list": all_files}).to_csv(args.extra_out, index=False)
            print(f"Wrote {len(all_files)} files to {args.extra_out}")


if __name__ == "__main__":
    main()
