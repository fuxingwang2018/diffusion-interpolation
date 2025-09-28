#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_member_merged(mm_raw: str):
    """Parse JSON list in MemberMerged, handling doubled quotes like "[""a"",""b""]"."""
    if pd.isna(mm_raw):
        return None
    mm_raw = str(mm_raw).strip()
    if not mm_raw:
        return None
    try:
        return json.loads(mm_raw)
    except json.JSONDecodeError:
        return json.loads(mm_raw.replace('""', '"'))


def row_ok(row, root: Path) -> bool:
    # Parse MemberMerged
    mm_list = parse_member_merged(row.get("MemberMerged", ""))
    if not isinstance(mm_list, list):
        return False

    # Parse TotalFiles
    try:
        total = int(row.get("TotalFiles", 0))
    except (TypeError, ValueError):
        return False

    # Count existing files (resolve under root if relative)
    existing = 0
    for p in mm_list:
        pth = Path(str(p))
        if not pth.is_absolute():
            pth = root / pth
        if pth.exists():
            existing += 1

    return existing == total


def main(input_csv: Path, root: Path, out_csv: Path | None):
    root = root.resolve()
    out_csv = out_csv or input_csv.with_name(input_csv.stem + "-corrected" + input_csv.suffix)

    df = pd.read_csv(input_csv)
    mask = df.apply(lambda r: row_ok(r, root), axis=1)
    kept = df[mask].copy()

    kept.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} (kept {len(kept)}/{len(df)} records)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Filter CSV rows where MemberMerged files are missing.")
    ap.add_argument("csv", type=Path, help="Path to input CSV")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root to resolve relative file paths")
    ap.add_argument("--out", type=Path, default=None, help="Optional output CSV path (default: <input>-corrected.csv)")
    args = ap.parse_args()

    main(args.csv, args.root, args.out)
