#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate manual labels vs predictions, and estimate whether errors are model- or strategy-driven.

Key idea:
- We already have per-frame Delta_Pitch/Delta_Yaw in the prediction CSV.
- If changing *only thresholds* on the same deltas can greatly improve accuracy on labels,
  the main issue is likely the strategy (rules/thresholds).
- If even best thresholds can't improve much, the issue is more likely model output
  (pitch/yaw) and/or reference calibration (ref) rather than the final mapping.

Usage:
  python gaze_onnx/experiments/eval_labels.py \
    --pred-csv gaze_onnx/output/output_gaze_smooth4_full.csv \
    --labels gaze_onnx/experiments/samples_smooth4_full/labels.csv
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


CLASSES = ["Forward", "Non-Forward", "In-Car", "No Face"]


@dataclass(frozen=True)
class Joined:
    frame_id: int
    timestamp: float
    human: str
    pred: str
    delta_pitch: Optional[float]
    delta_yaw: Optional[float]
    smooth_pitch: Optional[float]
    smooth_yaw: Optional[float]


def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_pred(path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fid = int(float(row["FrameID"]))
            except Exception:
                continue
            out[fid] = row
    return out


def read_labels(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def classify_delta_v2(
    dp: float,
    dy: float,
    *,
    incar_pitch_neg: float,
    incar_pitch_pos: float,
    incar_pitch_pos_max: float,
    incar_yaw_max: float,
    nonforward_yaw_enter: float,
    nonforward_pitch_up_enter: float,
) -> str:
    dy_abs = abs(float(dy))
    dp = float(dp)

    if dy_abs >= float(nonforward_yaw_enter):
        return "Non-Forward"
    if dp >= float(nonforward_pitch_up_enter):
        return "Non-Forward"

    if dy_abs <= float(incar_yaw_max):
        if dp <= float(incar_pitch_neg):
            return "In-Car"
        if float(incar_pitch_pos) <= dp <= float(incar_pitch_pos_max):
            return "In-Car"

    return "Forward"


def confusion(rows: Iterable[Joined]) -> Dict[Tuple[str, str], int]:
    m: Dict[Tuple[str, str], int] = {}
    for r in rows:
        k = (r.human, r.pred)
        m[k] = m.get(k, 0) + 1
    return m


def prf(rows: List[Joined]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for c in CLASSES:
        tp = sum(1 for r in rows if r.human == c and r.pred == c)
        fp = sum(1 for r in rows if r.human != c and r.pred == c)
        fn = sum(1 for r in rows if r.human == c and r.pred != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        stats[c] = {"precision": prec, "recall": rec, "f1": f1, "support": float(sum(1 for r in rows if r.human == c))}
    return stats


def print_confusion(rows: List[Joined]) -> None:
    labels = [c for c in CLASSES if any(r.human == c for r in rows)]
    preds = [c for c in CLASSES if any(r.pred == c for r in rows)]
    m = confusion(rows)

    # Header
    print("Confusion (rows=Human, cols=Pred):")
    print("".ljust(14) + "".join(p.ljust(14) for p in preds))
    for h in labels:
        line = h.ljust(14)
        for p in preds:
            line += str(m.get((h, p), 0)).ljust(14)
        print(line)


def accuracy(rows: List[Joined]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.human == r.pred) / len(rows)


def join(pred: Dict[int, Dict[str, str]], labels: List[Dict[str, str]]) -> List[Joined]:
    out: List[Joined] = []
    for lr in labels:
        try:
            fid = int(float(lr["FrameID"]))
            # Support both 'label' (from web tool) and 'Human_Label' (legacy)
            human = str(lr.get("label") or lr.get("Human_Label") or "").strip()
        except Exception:
            continue
        if human in ("", "Skip"):
            continue
        if human == "Unknown":
            continue
        pr = pred.get(fid)
        if pr is None:
            continue
        out.append(
            Joined(
                frame_id=fid,
                timestamp=float(pr.get("Timestamp", "0") or 0.0),
                human=human,
                pred=str(pr.get("Gaze_Class", "")),
                delta_pitch=_to_float(pr.get("Delta_Pitch", "")),
                delta_yaw=_to_float(pr.get("Delta_Yaw", "")),
                smooth_pitch=_to_float(pr.get("Smooth_Pitch", "")),
                smooth_yaw=_to_float(pr.get("Smooth_Yaw", "")),
            )
        )
    return out


def grid_search_thresholds(rows: List[Joined]) -> Tuple[float, Dict[str, float]]:
    # candidate sets are intentionally small so it runs fast
    cand_incar_neg = [-20, -15, -12, -10, -8, -7, -6, -4, -2]
    cand_incar_pos = [0, 1, 2, 3, 4, 6]
    cand_incar_pos_max = [8, 10, 12, 15]
    cand_incar_yaw = [8, 10, 12, 15]
    cand_nf_yaw = [14, 16, 18, 20, 24]
    cand_nf_up = [10, 15, 20]

    usable = [r for r in rows if r.delta_pitch is not None and r.delta_yaw is not None]
    if not usable:
        return 0.0, {}

    best_acc = -1.0
    best = {}
    for incar_pitch_neg in cand_incar_neg:
        for incar_pitch_pos in cand_incar_pos:
            for incar_pitch_pos_max in cand_incar_pos_max:
                if incar_pitch_pos_max < incar_pitch_pos:
                    continue
                for incar_yaw_max in cand_incar_yaw:
                    for nonforward_yaw_enter in cand_nf_yaw:
                        for nonforward_pitch_up_enter in cand_nf_up:
                            ok = 0
                            for r in usable:
                                pred2 = classify_delta_v2(
                                    r.delta_pitch,
                                    r.delta_yaw,
                                    incar_pitch_neg=float(incar_pitch_neg),
                                    incar_pitch_pos=float(incar_pitch_pos),
                                    incar_pitch_pos_max=float(incar_pitch_pos_max),
                                    incar_yaw_max=float(incar_yaw_max),
                                    nonforward_yaw_enter=float(nonforward_yaw_enter),
                                    nonforward_pitch_up_enter=float(nonforward_pitch_up_enter),
                                )
                                if pred2 == r.human:
                                    ok += 1
                            acc = ok / len(usable)
                            if acc > best_acc:
                                best_acc = acc
                                best = {
                                    "incar_pitch_neg": float(incar_pitch_neg),
                                    "incar_pitch_pos": float(incar_pitch_pos),
                                    "incar_pitch_pos_max": float(incar_pitch_pos_max),
                                    "incar_yaw_max": float(incar_yaw_max),
                                    "nonforward_yaw_enter": float(nonforward_yaw_enter),
                                    "nonforward_pitch_up_enter": float(nonforward_pitch_up_enter),
                                }
    return float(best_acc), best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate gaze predictions vs manual labels")
    p.add_argument("--pred-csv", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--no-threshold-search", action="store_true", help="Skip threshold grid search")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = read_pred(args.pred_csv)
    labels = read_labels(args.labels)
    rows = join(pred, labels)
    if not rows:
        raise SystemExit("No joined labeled rows. Check FrameID matching.")

    print(f"Joined labeled frames: {len(rows)}")
    print(f"Accuracy (current strategy): {accuracy(rows)*100:.2f}%")
    print_confusion(rows)

    stats = prf(rows)
    print("\nPer-class metrics:")
    for c in CLASSES:
        if not any(r.human == c for r in rows):
            continue
        s = stats[c]
        print(f"  {c:11s}  P={s['precision']*100:5.1f}%  R={s['recall']*100:5.1f}%  F1={s['f1']*100:5.1f}%  n={int(s['support'])}")

    # Error margin diagnostics: if errors are far from thresholds, likely model/ref issue.
    mism = [r for r in rows if r.human != r.pred and r.delta_pitch is not None and r.delta_yaw is not None]
    if mism:
        print("\nSome mismatches (showing up to 15):")
        for r in mism[:15]:
            print(
                f"  Frame={r.frame_id} t={r.timestamp:.2f}  human={r.human:11s} pred={r.pred:11s}  "
                f"dp={r.delta_pitch:6.1f} dy={r.delta_yaw:6.1f}  sp={'' if r.smooth_pitch is None else f'{r.smooth_pitch:5.1f}'} "
            )

    if args.no_threshold_search:
        return

    best_acc, best = grid_search_thresholds(rows)
    if best:
        print("\nThreshold search (same deltas, only change rule thresholds):")
        print(f"  Best accuracy: {best_acc*100:.2f}%")
        print("  Best thresholds:")
        for k, v in best.items():
            print(f"    {k}: {v}")

        cur = accuracy([r for r in rows if r.delta_pitch is not None and r.delta_yaw is not None])
        gain = best_acc - cur
        print(f"  Estimated strategy headroom (delta-only): {gain*100:.2f}%")
        print("\nInterpretation:")
        print("  - 如果 headroom 很大(比如 +10%~+30%)：策略/阈值问题为主。")
        print("  - 如果 headroom 很小(比如 <+3%~+5%)：更像模型角度输出或ref校准出了系统性偏差。")


if __name__ == "__main__":
    main()
