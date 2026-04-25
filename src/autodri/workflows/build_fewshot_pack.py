#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a smaller annotation pack (e.g. 200-shot) from an existing pack.

Input pack:
  <src-pack>/
    manifest.csv
    images/...
    labels.csv (optional)

Output pack:
  <out-pack>/
    manifest.csv
    labels.csv   (prefilled from source labels when available)
    images/...   (copy/hardlink/symlink from source)
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build few-shot annotation pack from a larger pack")
    p.add_argument("--src-pack", required=True, help="Source annotation pack directory")
    p.add_argument("--out-pack", required=True, help="Output annotation pack directory")
    p.add_argument("--num-samples", type=int, default=200, help="Target sample count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sample-mode",
        choices=["by_video_uniform", "random"],
        default="by_video_uniform",
        help="by_video_uniform keeps coverage over videos; random is pure random sample.",
    )
    p.add_argument(
        "--link-mode",
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="How to materialize image files in out-pack",
    )
    p.add_argument(
        "--keep-labeled",
        action="store_true",
        help="Always include already-labeled rows from source labels.csv first",
    )
    return p.parse_args()


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float((x or "").strip()))
    except Exception:
        return default


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        dst.write_bytes(src.read_bytes())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"unknown link mode: {mode}")


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if "img" not in rows[0]:
        raise ValueError(f"manifest missing column 'img': {path}")
    return rows


def load_label_map(labels_csv: Path) -> Dict[str, str]:
    if not labels_csv.exists():
        return {}
    out: Dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            img = (row.get("img") or "").strip()
            lab = (row.get("label") or row.get("Human_Label") or "").strip()
            if img:
                out[img] = lab
    return out


def pick_rows(rows: List[Dict[str, str]], n: int, seed: int, mode: str) -> List[Dict[str, str]]:
    n = max(1, min(int(n), len(rows)))
    rng = random.Random(seed)

    if mode == "random":
        picks = rng.sample(rows, n)
        picks.sort(
            key=lambda r: (
                str(r.get("Domain", "")),
                str(r.get("Video", "")),
                _safe_int(r.get("FrameID", "0")),
            )
        )
        return picks

    by_video: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_video[str(r.get("Video", "")).strip()].append(r)

    videos = sorted(by_video.keys())
    if not videos:
        raise ValueError("No Video field available in manifest rows")

    for v in videos:
        by_video[v].sort(key=lambda r: _safe_int(r.get("FrameID", "0")))

    base = n // len(videos)
    rem = n % len(videos)

    selected: List[Dict[str, str]] = []
    leftovers: Dict[str, List[Dict[str, str]]] = {}
    for i, v in enumerate(videos):
        k = base + (1 if i < rem else 0)
        pool = by_video[v]
        if k <= 0:
            leftovers[v] = pool[:]
            continue
        if len(pool) <= k:
            selected.extend(pool)
            leftovers[v] = []
            continue
        picks = rng.sample(pool, k)
        selected.extend(picks)
        pick_set = {id(x) for x in picks}
        leftovers[v] = [x for x in pool if id(x) not in pick_set]

    while len(selected) < n:
        candidates = [v for v in videos if leftovers.get(v)]
        if not candidates:
            break
        v = rng.choice(candidates)
        idx = rng.randrange(len(leftovers[v]))
        selected.append(leftovers[v].pop(idx))

    if len(selected) > n:
        selected = rng.sample(selected, n)

    selected.sort(
        key=lambda r: (
            str(r.get("Domain", "")),
            str(r.get("Video", "")),
            _safe_int(r.get("FrameID", "0")),
        )
    )
    return selected


def main() -> None:
    args = parse_args()
    src_pack = Path(args.src_pack)
    out_pack = Path(args.out_pack)
    src_manifest = src_pack / "manifest.csv"
    src_labels = src_pack / "labels.csv"

    rows = load_manifest(src_manifest)
    label_map = load_label_map(src_labels)
    n_target = int(args.num_samples)

    if args.keep_labeled and label_map:
        labeled_rows = [r for r in rows if (label_map.get(str(r.get("img", "")).strip(), "")).strip()]
        # Dedup by image path to avoid duplicates.
        seen = set()
        labeled_rows_u = []
        for r in labeled_rows:
            k = str(r.get("img", "")).strip()
            if k in seen:
                continue
            seen.add(k)
            labeled_rows_u.append(r)

        if len(labeled_rows_u) >= n_target:
            selected = pick_rows(labeled_rows_u, n=n_target, seed=int(args.seed), mode=str(args.sample_mode))
        else:
            labeled_set = {str(r.get("img", "")).strip() for r in labeled_rows_u}
            remaining = [r for r in rows if str(r.get("img", "")).strip() not in labeled_set]
            need = n_target - len(labeled_rows_u)
            fill = pick_rows(remaining, n=need, seed=int(args.seed), mode=str(args.sample_mode)) if remaining else []
            selected = labeled_rows_u + fill
            # Stable order for UI.
            selected.sort(
                key=lambda r: (
                    str(r.get("Domain", "")),
                    str(r.get("Video", "")),
                    _safe_int(r.get("FrameID", "0")),
                )
            )
    else:
        selected = pick_rows(rows, n=n_target, seed=int(args.seed), mode=str(args.sample_mode))

    out_images = out_pack / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # Materialize images first; drop any row with missing source image.
    kept: List[Dict[str, str]] = []
    for r in selected:
        rel = str(r.get("img", "")).strip()
        if not rel:
            continue
        src_img = src_pack / rel
        dst_img = out_pack / rel
        if not src_img.exists():
            continue
        materialize(src_img, dst_img, mode=str(args.link_mode))
        kept.append(r)

    if not kept:
        raise SystemExit("No selected images materialized. Check source pack.")

    # Write manifest with original field order.
    fields = list(kept[0].keys())
    out_manifest = out_pack / "manifest.csv"
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept:
            w.writerow(r)

    # Write labels.csv in web_label_tool format, prefilled from source labels.
    out_labels = out_pack / "labels.csv"
    with out_labels.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "label", "FrameID", "Timestamp", "Pred_Class", "Domain", "Video"])
        for r in kept:
            rel = str(r.get("img", "")).strip()
            w.writerow(
                [
                    rel,
                    label_map.get(rel, ""),
                    str(r.get("FrameID", "")),
                    str(r.get("Timestamp", "")),
                    str(r.get("Pred_Class", "")),
                    str(r.get("Domain", "")),
                    str(r.get("Video", "")),
                ]
            )

    n_prefill = sum(1 for r in kept if (label_map.get(str(r.get("img", "")).strip(), "")).strip())
    print(f"Saved pack: {out_pack}")
    print(f"Selected: {len(kept)} / source {len(rows)}")
    print(f"Prefilled labels: {n_prefill}")
    print(f"Manifest: {out_manifest}")
    print(f"Labels:   {out_labels}")


if __name__ == "__main__":
    main()
