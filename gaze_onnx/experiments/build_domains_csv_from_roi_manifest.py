#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build domains CSV from manually filled ROI manifest.

Input: roi_label_manifest.csv from prepare_roi_label_pack.py
Output: domains CSV compatible with create_multidomain_annotation_pack.py
"""

import argparse
import csv
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build domains CSV from ROI manifest")
    p.add_argument("--roi-manifest", required=True, help="Path to roi_label_manifest.csv")
    p.add_argument("--out-csv", required=True, help="Output domains csv")
    p.add_argument("--domain-id", required=True, help="Domain id to write, e.g. p1")
    p.add_argument("--samples-per-video", type=int, default=150, help="n_samples for each video row")
    p.add_argument("--skip-missing-roi", action="store_true", help="Skip rows without full ROI coordinates")
    return p.parse_args()


def as_int(v: str) -> int:
    return int(float(str(v).strip()))


def main() -> None:
    args = parse_args()
    src = Path(args.roi_manifest)
    dst = Path(args.out_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out: List[dict] = []
    for r in rows:
        vals = [r.get("roi_x1", "").strip(), r.get("roi_y1", "").strip(), r.get("roi_x2", "").strip(), r.get("roi_y2", "").strip()]
        complete = all(vals)
        if not complete:
            if args.skip_missing_roi:
                continue
            raise ValueError(f"Missing ROI in row video_rel={r.get('video_rel')}")
        x1, y1, x2, y2 = map(as_int, vals)
        out.append(
            {
                "domain_id": args.domain_id,
                "video": r["video_abs"],
                "roi_x1": x1,
                "roi_y1": y1,
                "roi_x2": x2,
                "roi_y2": y2,
                "n_samples": int(args.samples_per_video),
            }
        )

    fields = ["domain_id", "video", "roi_x1", "roi_y1", "roi_x2", "roi_y2", "n_samples"]
    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out:
            w.writerow(r)

    print(f"rows_in={len(rows)}")
    print(f"rows_out={len(out)}")
    print(f"out_csv={dst}")


if __name__ == "__main__":
    main()
