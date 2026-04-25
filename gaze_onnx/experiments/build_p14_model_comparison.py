#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare old vs new p14 gaze inference summaries by segment and by video folder."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build p14 old-vs-new model comparison report")
    p.add_argument("--plan-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-video-csv", required=True)
    p.add_argument("--old-dir", required=True, help="Directory with old summary json backups")
    p.add_argument("--new-dir", required=True, help="Directory with new summary jsons")
    return p.parse_args()


def load_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_map(d: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in d.items()}


def get_pct(summary: Dict[str, object], key: str) -> float:
    return float(summary.get("class_percent", {}).get(key, 0.0))


def main() -> None:
    args = parse_args()
    plan_rows = list(csv.DictReader(Path(args.plan_csv).open("r", encoding="utf-8-sig", newline="")))
    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)

    rows: List[Dict[str, str]] = []
    by_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for pr in plan_rows:
        segment_uid = str(pr["segment_uid"])
        gpath = Path(pr["gaze_csv"])
        new_json = new_dir / (gpath.name + ".summary.json")
        old_json = old_dir / (gpath.name + ".summary.json")
        if not new_json.exists() or not old_json.exists():
            continue
        old_s = load_summary(old_json)
        new_s = load_summary(new_json)

        old_f = get_pct(old_s, "Forward")
        old_nf = get_pct(old_s, "Non-Forward")
        old_ic = get_pct(old_s, "In-Car")
        new_f = get_pct(new_s, "Forward")
        new_nf = get_pct(new_s, "Non-Forward")
        new_ic = get_pct(new_s, "In-Car")

        row = {
            "segment_uid": segment_uid,
            "video_folder_name": pr["video_folder_name"],
            "video_path": pr["video_path"],
            "start_sec": pr["start_sec"],
            "end_sec": pr["end_sec"],
            "duration_sec": pr["duration_sec"],
            "old_model": str(old_s.get("model", "")),
            "new_model": str(new_s.get("model", "")),
            "old_pct_forward": f"{old_f:.3f}",
            "old_pct_nonforward": f"{old_nf:.3f}",
            "old_pct_incar": f"{old_ic:.3f}",
            "new_pct_forward": f"{new_f:.3f}",
            "new_pct_nonforward": f"{new_nf:.3f}",
            "new_pct_incar": f"{new_ic:.3f}",
            "delta_forward": f"{new_f - old_f:.3f}",
            "delta_nonforward": f"{new_nf - old_nf:.3f}",
            "delta_incar": f"{new_ic - old_ic:.3f}",
        }
        rows.append(row)
        by_video[pr["video_folder_name"]].append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    video_rows: List[Dict[str, str]] = []
    for folder, rs in sorted(by_video.items()):
        def mean(k: str) -> float:
            vals = [float(r[k]) for r in rs]
            return sum(vals) / len(vals) if vals else 0.0
        video_rows.append(
            {
                "video_folder_name": folder,
                "segment_count": str(len(rs)),
                "old_pct_forward_mean": f"{mean('old_pct_forward'):.3f}",
                "old_pct_nonforward_mean": f"{mean('old_pct_nonforward'):.3f}",
                "old_pct_incar_mean": f"{mean('old_pct_incar'):.3f}",
                "new_pct_forward_mean": f"{mean('new_pct_forward'):.3f}",
                "new_pct_nonforward_mean": f"{mean('new_pct_nonforward'):.3f}",
                "new_pct_incar_mean": f"{mean('new_pct_incar'):.3f}",
                "delta_forward_mean": f"{mean('delta_forward'):.3f}",
                "delta_nonforward_mean": f"{mean('delta_nonforward'):.3f}",
                "delta_incar_mean": f"{mean('delta_incar'):.3f}",
            }
        )

    out_video_csv = Path(args.out_video_csv)
    with out_video_csv.open("w", encoding="utf-8", newline="") as f:
        fields2 = list(video_rows[0].keys()) if video_rows else []
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(video_rows)

    print(f"segment_rows={len(rows)}")
    print(f"video_rows={len(video_rows)}")
    print(f"out_csv={out_csv}")
    print(f"out_video_csv={out_video_csv}")


if __name__ == "__main__":
    main()
