#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate frame-level gaze CSV into fixed-length event windows.

Input CSV is expected from `gaze_state_cls.py` and should contain at least:
  - Timestamp
  - Gaze_Class

Output CSV contains one row per non-overlapping window with majority-vote class.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate gaze frame CSV into window-level events")
    p.add_argument("--csv", required=True, help="Frame-level gaze CSV path")
    p.add_argument("--out-csv", required=True, help="Output event-level CSV path")
    p.add_argument("--window-sec", type=float, default=20.0, help="Non-overlap window size")
    p.add_argument(
        "--ignore-classes",
        nargs="*",
        default=["Other", "Unknown", ""],
        help="Classes ignored in majority vote",
    )
    return p.parse_args()


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float((x or "").strip())
    except Exception:
        return default


def majority_vote(values: List[str]) -> str:
    if not values:
        return "Unknown"
    c = Counter(values)
    maxv = max(c.values())
    cands = sorted([k for k, v in c.items() if v == maxv])
    return cands[0] if cands else "Unknown"


def main() -> None:
    args = parse_args()
    src = Path(args.csv)
    if not src.exists():
        raise FileNotFoundError(src)
    win = float(args.window_sec)
    if win <= 0:
        raise ValueError("--window-sec must be > 0")

    ignore = {x.strip() for x in args.ignore_classes}

    with src.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Empty input CSV")

    bins: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    for r in rows:
        t = _safe_float(r.get("Timestamp", ""), 0.0)
        cls = (r.get("Gaze_Class", "") or "").strip()
        idx = int(t // win)
        bins[idx].append((t, cls))

    out_rows = []
    for idx in sorted(bins.keys()):
        vals = bins[idx]
        t0 = idx * win
        t1 = (idx + 1) * win
        all_classes = [c for _, c in vals]
        valid_classes = [c for c in all_classes if c not in ignore]
        vote = majority_vote(valid_classes)
        cnt_all = Counter(all_classes)
        cnt_valid = Counter(valid_classes)
        out_rows.append(
            {
                "window_id": str(idx),
                "t_start": f"{t0:.3f}",
                "t_end": f"{t1:.3f}",
                "frame_count": str(len(all_classes)),
                "valid_count": str(len(valid_classes)),
                "event_class": vote,
                "count_forward": str(int(cnt_valid.get("Forward", 0))),
                "count_incar": str(int(cnt_valid.get("In-Car", 0))),
                "count_nonforward": str(int(cnt_valid.get("Non-Forward", 0))),
                "count_other_raw": str(int(cnt_all.get("Other", 0))),
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_id",
        "t_start",
        "t_end",
        "frame_count",
        "valid_count",
        "event_class",
        "count_forward",
        "count_incar",
        "count_nonforward",
        "count_other_raw",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {out}")
    print(f"Windows: {len(out_rows)}  window_sec={win:.1f}")


if __name__ == "__main__":
    main()
