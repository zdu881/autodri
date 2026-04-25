#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Time-binned diagnostics for a gaze prediction CSV.

This helps answer: is the weird long-video behavior due to
- reference drift (Ref_Pitch/Ref_Yaw changing over time)
- deltas moving (Delta_Pitch/Delta_Yaw distribution shifting)
- rule triggers saturating (In-Car conditions firing too often)

Usage:
  python gaze_onnx/experiments/analyze_csv.py \
    --pred-csv gaze_onnx/output/output_gaze_smooth4_full.csv \
    --window-sec 10
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Row:
    t: float
    frame: int
    cls: str
    ref_p: Optional[float]
    ref_y: Optional[float]
    dp: Optional[float]
    dy: Optional[float]


def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_rows(path: str) -> List[Row]:
    out: List[Row] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out.append(
                    Row(
                        t=float(row["Timestamp"]),
                        frame=int(float(row["FrameID"])),
                        cls=str(row["Gaze_Class"]),
                        ref_p=_to_float(row.get("Ref_Pitch", "")),
                        ref_y=_to_float(row.get("Ref_Yaw", "")),
                        dp=_to_float(row.get("Delta_Pitch", "")),
                        dy=_to_float(row.get("Delta_Yaw", "")),
                    )
                )
            except Exception:
                continue
    return out


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pctl(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    i = int(round((len(xs2) - 1) * float(q)))
    i = max(0, min(len(xs2) - 1, i))
    return float(xs2[i])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze gaze CSV by time windows")
    p.add_argument("--pred-csv", required=True)
    p.add_argument("--window-sec", type=float, default=10.0)

    # thresholds used only to compute trigger rates
    p.add_argument("--incar-pitch-neg", type=float, default=-7.0)
    p.add_argument("--incar-pitch-pos", type=float, default=2.0)
    p.add_argument("--incar-pitch-pos-max", type=float, default=12.0)
    p.add_argument("--incar-yaw-max", type=float, default=12.0)
    p.add_argument("--nonforward-yaw-enter", type=float, default=18.0)
    p.add_argument("--nonforward-pitch-up-enter", type=float, default=15.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.pred_csv)
    if not rows:
        raise SystemExit("Empty CSV")

    win = max(1e-6, float(args.window_sec))
    t0 = rows[0].t
    t1 = rows[-1].t
    nwin = int(math.floor((t1 - t0) / win)) + 1

    print(f"Duration: {t1 - t0:.1f}s  windows={nwin}  window_sec={win}")

    for wi in range(nwin):
        a = t0 + wi * win
        b = a + win
        chunk = [r for r in rows if a <= r.t < b]
        if not chunk:
            continue

        counts: Dict[str, int] = {}
        for r in chunk:
            counts[r.cls] = counts.get(r.cls, 0) + 1

        ref_p = [r.ref_p for r in chunk if r.ref_p is not None]
        ref_y = [r.ref_y for r in chunk if r.ref_y is not None]
        dps = [r.dp for r in chunk if r.dp is not None]
        dys = [abs(r.dy) for r in chunk if r.dy is not None]

        # trigger rates under current rule (not equal to final class due to debounce)
        incar_trig = 0
        nf_trig = 0
        usable = 0
        for r in chunk:
            if r.dp is None or r.dy is None:
                continue
            usable += 1
            dp = float(r.dp)
            dy = abs(float(r.dy))
            if dy >= float(args.nonforward_yaw_enter) or dp >= float(args.nonforward_pitch_up_enter):
                nf_trig += 1
            else:
                if dy <= float(args.incar_yaw_max) and (dp <= float(args.incar_pitch_neg) or (float(args.incar_pitch_pos) <= dp <= float(args.incar_pitch_pos_max))):
                    incar_trig += 1

        total = len(chunk)
        pct = {k: v / total * 100.0 for k, v in counts.items()}
        line = (
            f"[{a:7.1f}-{b:7.1f}s] n={total:4d} "
            f"F={pct.get('Forward',0):5.1f}% N={pct.get('Non-Forward',0):5.1f}% I={pct.get('In-Car',0):5.1f}% NFc={pct.get('No Face',0):5.1f}% "
            f"refP={mean(ref_p):6.2f} refY={mean(ref_y):6.2f} "
            f"dp50={pctl(dps,0.5):6.2f} dp90={pctl(dps,0.9):6.2f} dy50={pctl(dys,0.5):6.2f} "
        )
        if usable:
            line += f"trig(InCar)={incar_trig/usable*100:5.1f}% trig(NonF)={nf_trig/usable*100:5.1f}%"
        print(line)


if __name__ == "__main__":
    main()
