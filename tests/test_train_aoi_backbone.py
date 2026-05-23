from __future__ import annotations

from autodri.workflows.train_aoi_backbone import should_drop_last_batch


def test_should_drop_last_batch_only_when_remainder_is_singleton() -> None:
    assert should_drop_last_batch(dataset_size=3, batch_size=2) is True
    assert should_drop_last_batch(dataset_size=4, batch_size=2) is False
    assert should_drop_last_batch(dataset_size=2, batch_size=8) is False
    assert should_drop_last_batch(dataset_size=1, batch_size=1) is False
