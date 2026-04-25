#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rebalance domains CSV n_samples to a target total."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebalance n_samples in domains CSV to a target total")
    p.add_argument("--domains-csv", required=True)
    p.add_argument("--target-total", type=int, required=True, help="Desired total n_samples")
    p.add_argument("--out-csv", default="", help="Optional output CSV path. Default overwrites input.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.domains_csv)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty CSV: {path}")

    target_total = max(1, int(args.target_total))
    n = len(rows)
    base = target_total // n
    rem = target_total % n

    for i, row in enumerate(rows):
        row["n_samples"] = str(base + (1 if i < rem else 0))

    out = Path(args.out_csv) if args.out_csv else path
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out}")
    print(f"Rows: {n}  target_total={target_total}  assigned_total={sum(int(r['n_samples']) for r in rows)}")


if __name__ == "__main__":
    main()
