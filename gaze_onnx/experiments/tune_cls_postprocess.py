#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune post-process parameters for gaze_state_cls.py using existing labels.

This script does not retrain the model. It searches over:
- class logit biases (Forward/In-Car/Non-Forward)
- confidence threshold
- presence thresholds (face score + face area ratio)

Usage:
  python gaze_onnx/experiments/tune_cls_postprocess.py \
    --pred-csv gaze_onnx/output/output_gaze_cls_full.csv \
    --labels gaze_onnx/experiments/samples_smooth4_full_500/labels.csv \
    --out-json gaze_onnx/output/tune_cls_postprocess.best.json
"""

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE_CLASSES = ("Forward", "In-Car", "Non-Forward")
ALL_CLASSES = ("Forward", "In-Car", "Non-Forward", "Other")


@dataclass
class Row:
    frame_id: int
    human: str
    current_pred: str
    face_score: Optional[float]
    face_ratio: Optional[float]
    probs: Optional[np.ndarray]


def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def normalize_label(x: str) -> str:
    s = (x or "").strip()
    low = s.lower()
    if low in ("", "skip", "unknown"):
        return ""
    if low in ("no face", "noface", "other"):
        return "Other"
    if low in ("forward", "in-car", "non-forward"):
        return s
    return s


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
        return list(csv.DictReader(f))


def join_rows(pred: Dict[int, Dict[str, str]], labels: List[Dict[str, str]]) -> List[Row]:
    out: List[Row] = []
    for lr in labels:
        try:
            fid = int(float(lr["FrameID"]))
        except Exception:
            continue

        human = normalize_label(str(lr.get("label") or lr.get("Human_Label") or ""))
        if human == "":
            continue

        pr = pred.get(fid)
        if pr is None:
            continue

        cur = normalize_label(str(pr.get("Gaze_Class") or pr.get("Final_Class") or ""))
        if cur == "":
            cur = "Other"

        p0 = _to_float(pr.get("Cls_Forward", ""))
        p1 = _to_float(pr.get("Cls_InCar", ""))
        p2 = _to_float(pr.get("Cls_NonForward", ""))
        probs = None
        if p0 is not None and p1 is not None and p2 is not None:
            arr = np.array([p0, p1, p2], dtype=np.float64)
            if np.all(np.isfinite(arr)) and np.all(arr >= 0):
                s = float(arr.sum())
                if s > 1e-9:
                    probs = arr / s

        out.append(
            Row(
                frame_id=fid,
                human=human,
                current_pred=cur,
                face_score=_to_float(pr.get("Face_Score", "")),
                face_ratio=_to_float(pr.get("Face_Area_Ratio", "")),
                probs=probs,
            )
        )
    return out


def confusion(rows: List[Tuple[str, str]], classes: List[str]) -> Dict[Tuple[str, str], int]:
    m: Dict[Tuple[str, str], int] = {(h, p): 0 for h in classes for p in classes}
    for h, p in rows:
        if (h, p) not in m:
            continue
        m[(h, p)] += 1
    return m


def metrics(rows: List[Tuple[str, str]], classes: List[str]) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    if not rows:
        return 0.0, 0.0, {}

    acc = sum(1 for h, p in rows if h == p) / len(rows)

    per_class = {}
    f1s = []
    for c in classes:
        tp = sum(1 for h, p in rows if h == c and p == c)
        fp = sum(1 for h, p in rows if h != c and p == c)
        fn = sum(1 for h, p in rows if h == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": float(sum(1 for h, _ in rows if h == c)),
        }
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return float(acc), macro_f1, per_class


def predict_one(
    row: Row,
    bias: np.ndarray,
    conf_thr: float,
    min_face_score: float,
    min_face_ratio: float,
) -> str:
    if row.face_score is None or row.face_score < min_face_score:
        return "Other"
    if row.face_ratio is not None and row.face_ratio < min_face_ratio:
        return "Other"
    if row.probs is None:
        return "Other"

    logits = np.log(np.clip(row.probs, 1e-8, 1.0)) + bias
    ex = np.exp(logits - float(np.max(logits)))
    probs = ex / max(1e-12, float(np.sum(ex)))
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    if conf < conf_thr:
        return "Other"
    return BASE_CLASSES[idx]


def print_confusion(rows: List[Tuple[str, str]], classes: List[str]) -> None:
    m = confusion(rows, classes)
    print("Confusion (rows=Human, cols=Pred):")
    print("".ljust(14) + "".join(c.ljust(14) for c in classes))
    for h in classes:
        line = h.ljust(14)
        for p in classes:
            line += str(m[(h, p)]).ljust(14)
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    ap.add_argument("--out-json", default="", help="Optional output path for best parameters")
    args = ap.parse_args()

    pred = read_pred(args.pred_csv)
    labels = read_labels(args.labels)
    joined = join_rows(pred, labels)

    if not joined:
        raise SystemExit("No joined rows. Check FrameID overlap and label columns.")

    label_classes = sorted(set(r.human for r in joined))
    classes = [c for c in ALL_CLASSES if c in label_classes]

    baseline_rows = [(r.human, r.current_pred if r.current_pred in ALL_CLASSES else "Other") for r in joined]
    base_acc, base_mf1, base_pc = metrics(baseline_rows, classes)

    print(f"Joined rows: {len(joined)}")
    print("\n=== Baseline ===")
    print(f"Accuracy: {base_acc * 100:.2f}%")
    print(f"Macro-F1: {base_mf1 * 100:.2f}%")
    print_confusion(baseline_rows, classes)
    print("Per-class:")
    for c in classes:
        s = base_pc[c]
        print(f"  {c:<12} P={s['precision']*100:5.1f}%  R={s['recall']*100:5.1f}%  F1={s['f1']*100:5.1f}%  n={int(s['support'])}")

    bias_vals = [-0.6, -0.3, 0.0, 0.3, 0.6]
    conf_vals = [0.0, 0.35, 0.45, 0.55, 0.65]
    face_score_vals = [0.45, 0.55, 0.65, 0.75]
    face_ratio_vals = [0.0, 0.005, 0.01, 0.015, 0.02]

    best = None
    best_rows: List[Tuple[str, str]] = []
    total = len(bias_vals) ** 3 * len(conf_vals) * len(face_score_vals) * len(face_ratio_vals)
    checked = 0

    for bf, bic, bnf in itertools.product(bias_vals, bias_vals, bias_vals):
        bias = np.array([bf, bic, bnf], dtype=np.float64)
        for conf_thr in conf_vals:
            for min_fs in face_score_vals:
                for min_fr in face_ratio_vals:
                    checked += 1
                    rows = [
                        (r.human, predict_one(r, bias, conf_thr, min_fs, min_fr))
                        for r in joined
                    ]
                    acc, mf1, _ = metrics(rows, classes)
                    target = acc if args.metric == "accuracy" else mf1
                    tie = mf1 if args.metric == "accuracy" else acc

                    if best is None or target > best["target"] or (abs(target - best["target"]) < 1e-9 and tie > best["tie"]):
                        best = {
                            "target": target,
                            "tie": tie,
                            "accuracy": acc,
                            "macro_f1": mf1,
                            "bias": [float(bf), float(bic), float(bnf)],
                            "cls_threshold": float(conf_thr),
                            "presence_min_face_score": float(min_fs),
                            "presence_min_face_ratio": float(min_fr),
                            "total": total,
                        }
                        best_rows = rows

    assert best is not None
    best_acc, best_mf1, best_pc = metrics(best_rows, classes)

    print("\n=== Best (grid search) ===")
    print(f"Search metric: {args.metric}")
    print(f"Checked combos: {checked}/{total}")
    print(f"Accuracy: {best_acc * 100:.2f}%  (delta {((best_acc - base_acc) * 100):+.2f}%)")
    print(f"Macro-F1: {best_mf1 * 100:.2f}%  (delta {((best_mf1 - base_mf1) * 100):+.2f}%)")
    print("Best params:")
    print(f"  class_bias (F,IC,NF): {best['bias']}")
    print(f"  cls_threshold: {best['cls_threshold']}")
    print(f"  presence_min_face_score: {best['presence_min_face_score']}")
    print(f"  presence_min_face_ratio: {best['presence_min_face_ratio']}")
    print_confusion(best_rows, classes)
    print("Per-class:")
    for c in classes:
        s = best_pc[c]
        print(f"  {c:<12} P={s['precision']*100:5.1f}%  R={s['recall']*100:5.1f}%  F1={s['f1']*100:5.1f}%  n={int(s['support'])}")

    if args.out_json:
        out = {
            "pred_csv": args.pred_csv,
            "labels": args.labels,
            "metric": args.metric,
            "joined_rows": len(joined),
            "classes": classes,
            "baseline": {
                "accuracy": base_acc,
                "macro_f1": base_mf1,
            },
            "best": {
                "accuracy": best_acc,
                "macro_f1": best_mf1,
                "class_bias": best["bias"],
                "cls_threshold": best["cls_threshold"],
                "presence_min_face_score": best["presence_min_face_score"],
                "presence_min_face_ratio": best["presence_min_face_ratio"],
            },
        }
        os_dir = args.out_json.rsplit("/", 1)[0] if "/" in args.out_json else ""
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved best params to: {args.out_json}")


if __name__ == "__main__":
    main()
