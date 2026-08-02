from __future__ import annotations

import csv
from pathlib import Path

from autodri.aoi.participant_lopo import build_lopo_dataset


def _write_source_dataset(root: Path, participant: str, rows: list[dict[str, str]]) -> Path:
    dataset = root / participant
    for row in rows:
        image_path = dataset / row["dst_rel"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
    with (dataset / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "domain", "frame_id", "timestamp", "video", "src_rel", "dst_rel", "augmented"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return dataset


def test_build_lopo_dataset_keeps_holdout_participant_only_in_test(tmp_path: Path) -> None:
    p1 = _write_source_dataset(
        tmp_path,
        "p1",
        [
            {
                "split": "train",
                "label": "Forward",
                "domain": "p1",
                "frame_id": "1",
                "timestamp": "1.0",
                "video": "p1_train.mp4",
                "src_rel": "images/p1_train.jpg",
                "dst_rel": "train/Forward/p1_train.jpg",
                "augmented": "0",
            },
            {
                "split": "val",
                "label": "In-Car",
                "domain": "p1",
                "frame_id": "2",
                "timestamp": "2.0",
                "video": "p1_val.mp4",
                "src_rel": "images/p1_val.jpg",
                "dst_rel": "val/In-Car/p1_val.jpg",
                "augmented": "0",
            },
            {
                "split": "train",
                "label": "Forward",
                "domain": "p1",
                "frame_id": "22",
                "timestamp": "2.5",
                "video": "p1_val.mp4",
                "src_rel": "images/p1_train_same_window.jpg",
                "dst_rel": "train/Forward/p1_train_same_window.jpg",
                "augmented": "0",
            },
        ],
    )
    p2 = _write_source_dataset(
        tmp_path,
        "p2",
        [
            {
                "split": "train",
                "label": "Non-Forward",
                "domain": "p2",
                "frame_id": "3",
                "timestamp": "3.0",
                "video": "p2_train.mp4",
                "src_rel": "images/p2_train.jpg",
                "dst_rel": "train/Non-Forward/p2_train.jpg",
                "augmented": "0",
            },
            {
                "split": "val",
                "label": "Forward",
                "domain": "p2",
                "frame_id": "4",
                "timestamp": "4.0",
                "video": "p2_val.mp4",
                "src_rel": "images/p2_val.jpg",
                "dst_rel": "val/Forward/p2_val.jpg",
                "augmented": "0",
            },
            {
                "split": "train",
                "label": "In-Car",
                "domain": "p2",
                "frame_id": "5",
                "timestamp": "5.0",
                "video": "p2_aug.mp4",
                "src_rel": "images/p2_aug.jpg",
                "dst_rel": "train/In-Car/p2_aug.jpg",
                "augmented": "1",
            },
        ],
    )

    summary = build_lopo_dataset(
        {"p1": p1, "p2": p2},
        holdout_participant="p2",
        out_dir=tmp_path / "lopo_p2",
        labels=("Forward", "In-Car", "Non-Forward"),
    )

    rows = list(csv.DictReader((tmp_path / "lopo_p2" / "split_manifest.csv").open()))
    by_source = {(row["participant"], row["source_split"], row["label"]): row["split"] for row in rows}
    assert by_source[("p1", "train", "Forward")] == "train"
    assert by_source[("p1", "val", "In-Car")] == "internal_val"
    assert by_source[("p2", "train", "Non-Forward")] == "test"
    assert by_source[("p2", "val", "Forward")] == "test"
    assert summary["train"] == 1
    assert summary["internal_val"] == 1
    assert summary["test"] == 2
    assert ("p1", "train", "train/Forward/p1_train_same_window.jpg") not in {
        (row["participant"], row["source_split"], row["src_rel"]) for row in rows
    }
    assert all(row["augmented"] == "0" for row in rows if row["split"] in {"internal_val", "test"})
    assert (tmp_path / "lopo_p2" / "internal_val").is_dir()
    assert (tmp_path / "lopo_p2" / "val").is_symlink()
    assert (tmp_path / "lopo_p2" / "test" / "Forward").is_dir()
