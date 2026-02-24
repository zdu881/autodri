#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple OpenCV labeling UI for sampled PNGs.

Keys:
  1 / f : Forward
  2 / n : Non-Forward
  3 / i : In-Car
  4 / o : Other
  0 / u : Unknown (skip from eval)
  s     : Skip (no label written)
  b     : Back (remove last label)
  q / ESC: Quit

Typical:
  python gaze_onnx/experiments/label_tool.py \
    --samples-dir gaze_onnx/experiments/samples_smooth4_full
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2


LABELS = {
    ord("1"): "Forward",
    ord("f"): "Forward",
    ord("F"): "Forward",
    ord("2"): "Non-Forward",
    ord("n"): "Non-Forward",
    ord("N"): "Non-Forward",
    ord("3"): "In-Car",
    ord("i"): "In-Car",
    ord("I"): "In-Car",
    ord("4"): "Other",
    ord("o"): "Other",
    ord("O"): "Other",
    ord("0"): "Unknown",
    ord("u"): "Unknown",
    ord("U"): "Unknown",
}


@dataclass
class Sample:
    img: str
    frame_id: int
    timestamp: float
    pred_class: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label sampled gaze frames")
    p.add_argument("--samples-dir", required=True, help="Directory containing manifest.csv and PNGs")
    p.add_argument("--labels", default=None, help="Output labels CSV path (default: <samples-dir>/labels.csv)")
    p.add_argument("--start", type=int, default=0, help="Start index within manifest")
    return p.parse_args()


def read_manifest(samples_dir: str) -> List[Sample]:
    manifest = os.path.join(samples_dir, "manifest.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(manifest)

    out: List[Sample] = []
    with open(manifest, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                Sample(
                    img=row["img"],
                    frame_id=int(row["FrameID"]),
                    timestamp=float(row["Timestamp"]),
                    pred_class=str(row.get("Pred_Class", "")),
                )
            )
    return out


def load_existing(labels_path: str) -> Tuple[Set[int], List[Dict[str, str]]]:
    if not os.path.exists(labels_path):
        return set(), []

    labeled: Set[int] = set()
    rows: List[Dict[str, str]] = []
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fid = int(row["FrameID"])
                labeled.add(fid)
                rows.append(row)
            except Exception:
                continue
    return labeled, rows


def write_all(labels_path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
    with open(labels_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["FrameID", "Timestamp", "Human_Label", "Pred_Class", "Img"],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    samples_dir = args.samples_dir
    labels_path = args.labels
    if labels_path is None:
        labels_path = os.path.join(samples_dir, "labels.csv")

    samples = read_manifest(samples_dir)
    labeled_set, labeled_rows = load_existing(labels_path)

    # Build a dict for quick overwrite and to support back
    by_frame: Dict[int, Dict[str, str]] = {int(r["FrameID"]): r for r in labeled_rows if "FrameID" in r}
    ordered_done: List[int] = [int(r["FrameID"]) for r in labeled_rows if "FrameID" in r]

    win = "Label Tool (1=F 2=N 3=I 4=O 0=U s=skip b=back q=quit)"

    i = max(0, int(args.start))
    while i < len(samples):
        s = samples[i]
        if s.frame_id in labeled_set:
            i += 1
            continue

        img_path = os.path.join(samples_dir, s.img)
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to read", img_path)
            i += 1
            continue

        # Add a small hint bar
        hint = img.copy()
        cv2.putText(
            hint,
            f"[{i+1}/{len(samples)}] Frame={s.frame_id} t={s.timestamp:.2f}s  pred={s.pred_class}",
            (12, hint.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win, hint)
        key = cv2.waitKey(0)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            i += 1
            continue
        if key in (ord("b"), ord("B")):
            if ordered_done:
                last = ordered_done.pop()
                if last in by_frame:
                    del by_frame[last]
                labeled_set.discard(last)
                write_all(labels_path, list(by_frame.values()))
                print("Back: removed label for FrameID", last)
            continue

        if key not in LABELS:
            print("Unknown key.")
            continue

        label = LABELS[key]
        row = {
            "FrameID": str(s.frame_id),
            "Timestamp": f"{s.timestamp:.3f}",
            "Human_Label": label,
            "Pred_Class": s.pred_class,
            "Img": s.img,
        }

        by_frame[s.frame_id] = row
        labeled_set.add(s.frame_id)
        ordered_done.append(s.frame_id)
        write_all(labels_path, list(by_frame.values()))
        i += 1

    cv2.destroyAllWindows()
    print("Labels:", labels_path)
    print(f"Labeled {len(labeled_set)}/{len(samples)}")


if __name__ == "__main__":
    main()
