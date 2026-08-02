from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from autodri.aoi.equivalence import DEFAULT_LABELS, load_split_manifest, write_csv_rows


def build_lopo_dataset(
    participant_datasets: Mapping[str, Path | str],
    *,
    holdout_participant: str,
    out_dir: Path | str,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> dict[str, int]:
    label_set = set(labels)
    out_path = Path(out_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    internal_val_groups = {
        participant: {
            sample.group_key()
            for sample in load_split_manifest(Path(dataset_raw) / "split_manifest.csv")
            if participant != holdout_participant
            and sample.split == "val"
            and sample.label in label_set
            and not sample.augmented
        }
        for participant, dataset_raw in participant_datasets.items()
    }
    for participant, dataset_raw in sorted(participant_datasets.items()):
        dataset_dir = Path(dataset_raw)
        samples = load_split_manifest(dataset_dir / "split_manifest.csv")
        for sample in samples:
            if sample.label not in label_set:
                continue
            if participant == holdout_participant:
                if sample.split not in {"train", "val"} or sample.augmented:
                    continue
                split = "test"
            elif sample.split == "train":
                if sample.group_key() in internal_val_groups[participant]:
                    continue
                split = "train"
            elif sample.split == "val":
                if sample.augmented:
                    continue
                split = "internal_val"
            else:
                continue

            src = dataset_dir / sample.dst_rel
            if not src.exists():
                raise FileNotFoundError(src)
            rel_name = sample.dst_rel.replace("/", "__")
            dst_rel = f"{split}/{sample.label}/{participant}__{rel_name}"
            dst = out_path / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())
            counts[split] += 1
            rows_out.append(
                {
                    "split": split,
                    "label": sample.label,
                    "domain": participant,
                    "frame_id": sample.frame_id,
                    "timestamp": f"{sample.timestamp:.6f}",
                    "video": sample.video,
                    "src_rel": str(src.relative_to(dataset_dir)),
                    "dst_rel": dst_rel,
                    "augmented": int(sample.augmented),
                    "participant": participant,
                    "source_split": sample.split,
                    "source_dataset": str(dataset_dir),
                }
            )

    _ensure_yolo_val_alias(out_path)
    write_csv_rows(
        out_path / "split_manifest.csv",
        rows_out,
        fieldnames=[
            "split",
            "label",
            "domain",
            "frame_id",
            "timestamp",
            "video",
            "src_rel",
            "dst_rel",
            "augmented",
            "participant",
            "source_split",
            "source_dataset",
        ],
    )
    return {split: counts[split] for split in ("train", "internal_val", "test")}


def _ensure_yolo_val_alias(out_path: Path) -> None:
    val_alias = out_path / "val"
    internal_val_dir = out_path / "internal_val"
    if val_alias.exists() or val_alias.is_symlink():
        if val_alias.is_symlink() or val_alias.is_file():
            val_alias.unlink()
        else:
            shutil.rmtree(val_alias)
    if internal_val_dir.exists():
        val_alias.symlink_to(internal_val_dir.resolve(), target_is_directory=True)


__all__ = ["build_lopo_dataset"]
