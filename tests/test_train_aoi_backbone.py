from __future__ import annotations

from autodri.aoi.equivalence import ManifestSample
from autodri.workflows.train_aoi_backbone import (
    assign_training_splits,
    resolve_training_labels,
    has_split_integrity_errors,
    should_drop_last_batch,
)


def test_should_drop_last_batch_only_when_remainder_is_singleton() -> None:
    assert should_drop_last_batch(dataset_size=3, batch_size=2) is True
    assert should_drop_last_batch(dataset_size=4, batch_size=2) is False
    assert should_drop_last_batch(dataset_size=2, batch_size=8) is False
    assert should_drop_last_batch(dataset_size=1, batch_size=1) is False


def test_assign_training_splits_can_use_physical_internal_val_layout() -> None:
    samples = [
        ManifestSample("train", "Forward", "d", 1, 1.0, "v1", "", "train/Forward/a.jpg", False),
        ManifestSample("internal_val", "In-Car", "d", 2, 2.0, "v2", "", "internal_val/In-Car/b.jpg", False),
        ManifestSample("val", "Non-Forward", "d", 3, 3.0, "v3", "", "val/Non-Forward/c.jpg", False),
        ManifestSample("test", "Other", "d", 4, 4.0, "v4", "", "test/Other/d.jpg", False),
    ]

    assignment = assign_training_splits(samples, use_physical_splits=True)

    assert assignment["train/Forward/a.jpg"] == "train"
    assert assignment["internal_val/In-Car/b.jpg"] == "internal_val"
    assert assignment["val/Non-Forward/c.jpg"] == "internal_val"
    assert assignment["test/Other/d.jpg"] == "test"


def test_physical_split_integrity_allows_val_as_internal_validation() -> None:
    integrity = {
        "group_leak_count": 0,
        "augmented_not_train_count": 0,
        "frozen_val_not_test_count": 3,
    }

    assert has_split_integrity_errors(integrity, use_physical_splits=True) is False
    assert has_split_integrity_errors(integrity, use_physical_splits=False) is True


def test_resolve_training_labels_can_use_primary_three_only() -> None:
    labels = resolve_training_labels(["Forward", "In-Car", "Non-Forward"])

    assert labels == ("Forward", "In-Car", "Non-Forward")
