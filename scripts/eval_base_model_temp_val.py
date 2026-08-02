from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CLASSES = ["Forward", "In-Car", "Non-Forward", "Other"]


def read_by_img(path: Path, key: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            row["img"]: (row.get(key) or "").strip()
            for row in csv.DictReader(f)
        }


def main() -> None:
    pack_dir = Path("gaze_onnx/experiments/base_model_temp_val_review")
    labels = read_by_img(pack_dir / "labels.csv", "label")
    preds = read_by_img(pack_dir / "model_predictions.csv", "model_pred")

    missing = [img for img, label in labels.items() if not label]
    if missing:
        raise SystemExit(
            f"{len(missing)} samples are still unlabeled. Finish labeling before evaluation."
        )

    imgs = [img for img in labels if img in preds]
    y_true = [labels[img] for img in imgs]
    y_pred = [preds[img] for img in imgs]

    out_dir = pack_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"n={len(y_true)}\n")
        f.write(f"accuracy={acc:.6f}\n\n")
        f.write(report_text)

    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *CLASSES])
        for cls, row in zip(CLASSES, cm):
            w.writerow([cls, *[int(x) for x in row]])

    print(f"n={len(y_true)}")
    print(f"accuracy={acc:.4f}")
    print(f"label_counts={dict(Counter(y_true))}")
    print()
    print(report_text)
    print(f"wrote {out_dir / 'classification_report.txt'}")
    print(f"wrote {out_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
