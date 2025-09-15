#!/usr/bin/env python3
"""
Cycle-first sequence discovery:
- Take unique forecast base times (from column 'Date').
- Filter to [--start-date, --end-date].
- For each cycle and each window (e.g. 0-6, 6-12, 12-18), verify that *all selected members*
  have all required lead times and that files exist (unless --no-verify-fs).
- Emit one row per (Date, Window) with aligned paths across members.

Example:
  python find_sequences_all_members.py \
      --labels labels.csv --root /data \
      --windows 0-6,6-12,12-18 \
      --members 0,1,2 \
      --start-date 2023-01-01T00:00:00Z \
      --end-date 2023-03-01T00:00:00Z \
      --out sequences_grouped.csv

Sample input labels.csv:
Name,Importance,PosX,PosY,Date,LeadTime,Member
2023/01/01/00/2023010100_lt00_mem000.npy,1,256,256,2023-01-01T00:00:00Z,0,0
2023/01/01/00/2023010100_lt00_mem001.npy,1,256,256,2023-01-01T00:00:00Z,0,1
2023/01/01/00/2023010100_lt00_mem002.npy,1,256,256,2023-01-01T00:00:00Z,0,2
2023/01/01/00/2023010100_lt00_mem003.npy,1,256,256,2023-01-01T00:00:00Z,0,3

"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any, Tuple

import pandas as pd

import os
os.environ["TZ"] = "UTC"


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
    Iterate cycles (dates_sorted). For each window, require that every member in selected_members
    has all required leads. If yes, emit one record per (Date, Window) with aligned paths.
    """
    records: List[Dict[str, Any]] = []

    for date in dates_sorted:
        member_map = cycle_index.get(date, {})
        # Require that ALL selected members exist in this cycle
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
    args = ap.parse_args()

    windows = Window.parse_many(args.windows)

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

    # If members not specified, default to all members present in (filtered) CSV
    selected_members = sorted(df["Member"].unique().tolist()) if members_list is None else sorted(members_list)

    # Build index and iterate cycles in chronological order
    cycle_index = build_cycle_index(df)
    dates_sorted = sorted(cycle_index.keys())

    seq_df = find_sequences_all_members(
        cycle_index=cycle_index,
        windows=windows,
        selected_members=selected_members,
        dates_sorted=dates_sorted,
    )

    # Summary
    if seq_df.empty:
        print("No complete (Date, Window) sequences found for ALL selected members.")
    else:
        ndates = seq_df["Date"].nunique()
        print(
            f"Found {len(seq_df)} (Date, Window) sequences across {ndates} cycles "
            f"for {len(selected_members)} members."
        )
        with pd.option_context("display.max_colwidth", 80):
            print(seq_df.head(10)[["Date", "Window", "NMembers", "NFilesPerMember", "StartValidTime", "EndValidTime"]])

    # CSV (robust to empty); serialize lists to JSON for clarity
    if args.out:
        if seq_df.empty:
            cols = [
                "Date", "Window", "LeadTimes", "Members", "MemberPaths",
                "NMembers", "NFilesPerMember", "TotalFiles", "StartValidTime", "EndValidTime"
            ]
            pd.DataFrame(columns=cols).to_csv(args.out, index=False)
            print(f"Wrote 0 sequences (header only) to {args.out}")
        else:
            out_df = seq_df.copy()
            # ISO-8601 for timestamps
            for col in ["Date", "StartValidTime", "EndValidTime"]:
                out_df[col] = pd.to_datetime(out_df[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            # JSON-encode structured columns
            out_df["LeadTimes"] = out_df["LeadTimes"].apply(lambda x: json.dumps(x))
            out_df["Members"] = out_df["Members"].apply(lambda x: json.dumps(x))
            out_df["MemberPaths"] = out_df["MemberPaths"].apply(lambda x: json.dumps(x))
            out_df.to_csv(args.out, index=False)
            print(f"Wrote {len(out_df)} sequences to {args.out}")

    if args.extra_out:
        if seq_df.empty:
            pd.DataFrame(columns=["file-list"]).to_csv(args.extra_out, index=False)
            print(f"Wrote 0 files (header only) to {args.extra_out}")
        else:
            # Flatten all MemberPaths
            all_files: List[str] = []
            for paths_per_member in seq_df["MemberPaths"]:
                for member_paths in paths_per_member:
                    all_files.extend(member_paths)
            flat_df = pd.DataFrame({"file-list": all_files})
            flat_df.to_csv(args.extra_out, index=False)
            print(f"Wrote {len(flat_df)} files to {args.extra_out}")
if __name__ == "__main__":
    main()
