import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO


@dataclass
class EvalRow:
    name: str
    dataset_dir: Path
    weights: Path


@dataclass
class Sample:
    image_path: Path
    label: str
    domain: str
    video: str
    timestamp: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-domain frame/event evaluation for gaze classification."
    )
    p.add_argument(
        "--eval-item",
        action="append",
        required=True,
        help=(
            "One item: name|dataset_dir|weights. "
            "Example: car1_to_car2|gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2_genv3|"
            "runs/classify/gaze_onnx/experiments/runs_cls/gaze_holdout_car2_genv3_cpu3/weights/best.pt"
        ),
    )
    p.add_argument(
        "--event-window-sec",
        type=float,
        default=30.0,
        help="Non-overlap window size for event-level majority evaluation.",
    )
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out-csv",
        default="",
        help="Optional path to save table CSV.",
    )
    return p.parse_args()


def parse_eval_items(items: List[str]) -> List[EvalRow]:
    out: List[EvalRow] = []
    for item in items:
        parts = item.split("|")
        if len(parts) != 3:
            raise ValueError(f"Bad --eval-item format: {item}")
        name, ds, w = parts
        out.append(EvalRow(name=name.strip(), dataset_dir=Path(ds.strip()), weights=Path(w.strip())))
    return out


def load_val_samples(dataset_dir: Path) -> List[Sample]:
    manifest = dataset_dir / "split_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest}")

    out: List[Sample] = []
    with open(manifest, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "label", "domain", "video", "timestamp", "dst_rel"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for r in reader:
            if r["split"] != "val":
                continue
            img = dataset_dir / r["dst_rel"]
            out.append(
                Sample(
                    image_path=img,
                    label=r["label"],
                    domain=r["domain"],
                    video=r["video"],
                    timestamp=float(r["timestamp"]),
                )
            )
    if not out:
        raise ValueError(f"No val samples found in {manifest}")
    return out


def majority_vote(values: List[str]) -> str:
    c = Counter(values)
    max_count = max(c.values())
    candidates = [k for k, v in c.items() if v == max_count]
    # deterministic tie-break: lexicographic
    return sorted(candidates)[0]


def event_accuracy(samples: List[Sample], preds: List[str], window_sec: float) -> Tuple[float, int]:
    grouped: Dict[Tuple[str, str, int], List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        win = int(s.timestamp // window_sec) if window_sec > 0 else 0
        grouped[(s.domain, s.video, win)].append(idx)

    total = 0
    correct = 0
    for ids in grouped.values():
        gt_major = majority_vote([samples[i].label for i in ids])
        pd_major = majority_vote([preds[i] for i in ids])
        total += 1
        if gt_major == pd_major:
            correct += 1
    acc = correct / total if total > 0 else 0.0
    return acc, total


def run_item(item: EvalRow, imgsz: int, batch: int, device: str, event_window_sec: float) -> Dict[str, str]:
    samples = load_val_samples(item.dataset_dir)
    image_paths = [str(s.image_path) for s in samples]
    gts = [s.label for s in samples]

    model = YOLO(str(item.weights))
    name_map = model.names
    outputs = model.predict(
        source=image_paths,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        stream=False,
    )
    preds: List[str] = []
    for r in outputs:
        cls_idx = int(r.probs.top1)
        preds.append(str(name_map[cls_idx]))

    frame_total = len(gts)
    frame_correct = sum(1 for gt, pd in zip(gts, preds) if gt == pd)
    frame_acc = frame_correct / frame_total

    event_acc, event_count = event_accuracy(samples, preds, window_sec=event_window_sec)
    val_domains = sorted({s.domain for s in samples})
    val_videos = sorted({s.video for s in samples})

    return {
        "name": item.name,
        "frame_acc": f"{frame_acc:.4f}",
        "frame_total": str(frame_total),
        "event_acc": f"{event_acc:.4f}",
        "event_total": str(event_count),
        "event_window_sec": f"{event_window_sec:.1f}",
        "val_domains": ",".join(val_domains),
        "val_videos": ",".join(val_videos),
        "dataset_dir": str(item.dataset_dir),
        "weights": str(item.weights),
    }


def print_table(rows: List[Dict[str, str]]) -> None:
    print("=== Cross-domain Evaluation Table ===")
    print("| name | frame_acc | frame_total | event_acc | event_total | event_window_sec | val_domains |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        print(
            f"| {r['name']} | {r['frame_acc']} | {r['frame_total']} | "
            f"{r['event_acc']} | {r['event_total']} | {r['event_window_sec']} | {r['val_domains']} |"
        )


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "frame_acc",
                "frame_total",
                "event_acc",
                "event_total",
                "event_window_sec",
                "val_domains",
                "val_videos",
                "dataset_dir",
                "weights",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = parse_args()
    items = parse_eval_items(args.eval_item)
    rows: List[Dict[str, str]] = []
    for item in items:
        rows.append(
            run_item(
                item,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                event_window_sec=args.event_window_sec,
            )
        )
    print_table(rows)
    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
