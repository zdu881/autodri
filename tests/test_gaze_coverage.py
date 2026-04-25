from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autodri.workflows.build_participants_results_summary import summarize_metrics
from autodri.workflows.compute_p1_window_metrics import (
    compute_one_window,
    expected_gaze_rows_for_window,
    infer_nominal_gaze_fps,
)


def test_infer_nominal_gaze_fps_picks_supported_rate() -> None:
    t25 = np.arange(0.0, 2.0, 1.0 / 25.0)
    t30 = np.arange(0.0, 2.0, 1.0 / 30.0)
    assert infer_nominal_gaze_fps(t25) == 25.0
    assert infer_nominal_gaze_fps(t30) == 30.0
    assert expected_gaze_rows_for_window(20.0, 25.0) == 500
    assert expected_gaze_rows_for_window(20.0, 30.0) == 600


def test_compute_one_window_adds_gaze_coverage_fields() -> None:
    gaze_df = pd.DataFrame(
        {
            "t": np.arange(0.0, 20.0, 1.0 / 25.0),
            "gaze": ["Forward"] * 500,
        }
    )
    gaze_df.attrs["nominal_gaze_fps"] = 25.0
    wheel_df = pd.DataFrame({"t": np.arange(0.0, 20.0, 1.0), "wheel": ["ON"] * 20})
    row = compute_one_window(
        video_path="demo.mp4",
        w0=0.0,
        w1=20.0,
        gaze_df=gaze_df,
        wheel_df=wheel_df,
        max_gap=0.35,
        gaze_coverage_threshold=0.98,
    )
    assert row["expected_gaze_rows"] == "500"
    assert row["gaze_coverage_ok"] == "1"
    assert row["gaze_qc_reason"] == ""


def test_summarize_metrics_uses_only_coverage_ok_windows(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                "status,gaze_coverage_ok,gaze_qc_reason,gaze_coverage_ratio,pct_time_off_path,glance_rate_per_min,offpath_count_ge_1p6s,offpath_count_ge_2p0s,wheel_on_ratio_overall",
                "ok,1,,1.0,10,6,1,0,0.8",
                "ok,0,low_gaze_coverage,0.5,30,3,2,1,0.2",
                "ok,0,zero_gaze_rows,0.0,50,1,3,2,0.1",
            ]
        ),
        encoding="utf-8",
    )
    out = summarize_metrics(metrics_csv, tmp_path / "missing_event_summary.csv")
    assert out["n_windows"] == "1"
    assert out["coverage_ok_windows"] == "1"
    assert out["coverage_fail_windows"] == "2"
    assert out["coverage_zero_windows"] == "1"
    assert out["mean_pct_time_off_path"] == "10.0000"


def test_summarize_metrics_keeps_old_metrics_csv_compatible(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics_old.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                "status,pct_time_off_path,glance_rate_per_min,offpath_count_ge_1p6s,offpath_count_ge_2p0s,wheel_on_ratio_overall",
                "ok,10,6,1,0,0.8",
                "ok,30,3,2,1,0.2",
            ]
        ),
        encoding="utf-8",
    )
    out = summarize_metrics(metrics_csv, tmp_path / "missing_event_summary.csv")
    assert out["n_windows"] == "2"
    assert out["coverage_ok_windows"] == ""
    assert out["coverage_fail_windows"] == ""
    assert out["mean_pct_time_off_path"] == "20.0000"
