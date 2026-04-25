#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a current participants results summary from repository outputs.

The summary prioritizes the metrics already emphasized in the repository:
- target_videos_matched
- infer plan coverage
- gaze class distribution
- off-path ratio on valid gaze frames
- window-level metrics when available
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from autodri.common.paths import (
    manifests_current_root,
    models_root,
    participant_analysis_dir,
    reports_root,
    resolve_workspace_or_repo_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build participants results summary")
    p.add_argument(
        "--out-csv",
        default=str(reports_root() / "participants_results_summary.current.csv"),
        help="Output summary CSV",
    )
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt_time(ts: float) -> str:
    if not ts:
        return ""
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def get_paths(participant: str) -> Dict[str, Path]:
    if participant == "p1":
        analysis_dir = participant_analysis_dir("p1")
        return {
            "target": manifests_current_root() / "p1_target_videos.current.csv",
            "plan": analysis_dir / "p1_infer_plan.segment.csv",
            "gmap": analysis_dir / "p1_gaze_map.segment.csv",
            "metrics": analysis_dir / "p1_window_metrics.20s.csv",
            "event_summary": analysis_dir / "p1_event20s_summary.csv",
        }
    analysis_dir = participant_analysis_dir(participant)
    return {
        "target": manifests_current_root() / f"{participant}_target_videos.current.csv",
        "plan": analysis_dir / f"{participant}_infer_plan.current.csv",
        "gmap": analysis_dir / f"{participant}_gaze_map.current.csv",
        "metrics": analysis_dir / f"{participant}_window_metrics.20s.current.csv",
        "event_summary": analysis_dir / f"{participant}_event20s_summary.csv",
    }


def summarize_gaze(gaze_files: List[Path]) -> Dict[str, object]:
    c = Counter()
    valid = 0
    off = 0
    total = 0
    latest = 0.0
    for path in gaze_files:
        if not path.exists():
            continue
        latest = max(latest, path.stat().st_mtime)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = str(r.get("Gaze_Class", "")).strip()
                if not g:
                    continue
                c[g] += 1
                total += 1
                if g in {"Forward", "Non-Forward", "In-Car"}:
                    valid += 1
                    if g != "Forward":
                        off += 1
    pct = {}
    if total > 0:
        pct = {k: round(v / total * 100.0, 1) for k, v in c.items()}
    dominant = max(c.items(), key=lambda kv: kv[1])[0] if c else ""
    off_ratio = round(off / valid * 100.0, 1) if valid else ""
    return {
        "total_frames": total,
        "counts": dict(c),
        "pct": pct,
        "dominant": dominant,
        "off_ratio": off_ratio,
        "latest": latest,
    }


def summarize_metrics(metrics_csv: Path, event_summary_csv: Path) -> Dict[str, object]:
    if not metrics_csv.exists():
        return {
            "n_windows": "",
            "coverage_ok_windows": "",
            "coverage_fail_windows": "",
            "coverage_zero_windows": "",
            "mean_gaze_coverage_ratio": "",
            "mean_pct_time_off_path": "",
            "mean_glance_rate_per_min": "",
            "mean_offpath_ge_1p6s_per_window": "",
            "mean_offpath_ge_2p0s_per_window": "",
            "mean_wheel_on_ratio_overall": "",
            "latest": 0.0,
        }

    rows = read_csv(metrics_csv)
    ok = [r for r in rows if str(r.get("status", "")).strip().lower() == "ok"]
    has_coverage = any("gaze_coverage_ok" in r for r in rows)
    if has_coverage:
        ok_for_summary = [r for r in ok if str(r.get("gaze_coverage_ok", "0")).strip() == "1"]
    else:
        ok_for_summary = ok

    def mean(col: str) -> str:
        vals = []
        for r in ok_for_summary:
            try:
                vals.append(float(r[col]))
            except Exception:
                pass
        return f"{sum(vals)/len(vals):.4f}" if vals else ""

    def count(pred) -> str:
        return str(sum(1 for r in ok if pred(r)))

    return {
        "n_windows": str(len(ok_for_summary)),
        "coverage_ok_windows": count(lambda r: str(r.get("gaze_coverage_ok", "0")).strip() == "1") if has_coverage else "",
        "coverage_fail_windows": count(lambda r: str(r.get("gaze_coverage_ok", "0")).strip() == "0") if has_coverage else "",
        "coverage_zero_windows": count(lambda r: str(r.get("gaze_qc_reason", "")).strip() == "zero_gaze_rows") if has_coverage else "",
        "mean_gaze_coverage_ratio": mean("gaze_coverage_ratio") if has_coverage else "",
        "mean_pct_time_off_path": mean("pct_time_off_path"),
        "mean_glance_rate_per_min": mean("glance_rate_per_min"),
        "mean_offpath_ge_1p6s_per_window": mean("offpath_count_ge_1p6s"),
        "mean_offpath_ge_2p0s_per_window": mean("offpath_count_ge_2p0s"),
        "mean_wheel_on_ratio_overall": mean("wheel_on_ratio_overall"),
        "latest": metrics_csv.stat().st_mtime,
    }


def infer_model(participant: str, gaze_files: List[Path]) -> str:
    if participant == "p1":
        p = models_root() / "gaze_cls_p1_200shot_driveonly_ft_v1.onnx"
        return str(p) if p.exists() else ""
    v2 = models_root() / f"gaze_cls_{participant}_audit120_driveonly_ft_v2.onnx"
    if v2.exists():
        return str(v2)
    v1 = models_root() / f"gaze_cls_{participant}_200shot_driveonly_ft_v1.onnx"
    if v1.exists():
        return str(v1)
    for g in gaze_files:
        js = Path(str(g) + ".summary.json")
        if js.exists():
            try:
                data = json.loads(js.read_text(encoding="utf-8"))
                m = str(data.get("model", "")).strip()
                if m:
                    return m
            except Exception:
                pass
    return ""


def main() -> None:
    args = parse_args()
    participants = ["p1", "p2", "p4", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18"]
    out_rows: List[Dict[str, str]] = []

    for p in participants:
        ps = get_paths(p)
        target_rows = read_csv(ps["target"])
        target_matched = ""
        if target_rows:
            target_matched = str(sum(1 for r in target_rows if str(r.get("video", "") or r.get("video_path", "")).strip()))

        plan_rows = read_csv(ps["plan"])
        plan_total = str(len(plan_rows)) if plan_rows else ""
        plan_ok = str(sum(1 for r in plan_rows if str(r.get("status", "")).strip().lower() == "ok")) if plan_rows else ""

        gmap_rows = read_csv(ps["gmap"])
        gaze_files: List[Path] = []
        for r in gmap_rows:
            gv = str(r.get("gaze_csv", "")).strip()
            if not gv:
                continue
            gaze_files.append(resolve_workspace_or_repo_path(gv))
        existing_gaze_files = [f for f in gaze_files if f.exists()]
        gaze = summarize_gaze(existing_gaze_files)
        metrics = summarize_metrics(ps["metrics"], ps["event_summary"])
        model_path = infer_model(p, existing_gaze_files)

        out_rows.append(
            {
                "participant": p,
                "target_videos_matched": target_matched,
                "infer_plan_ok": plan_ok,
                "infer_plan_total": plan_total,
                "gaze_segments_done": str(len(existing_gaze_files)),
                "current_model": model_path,
                "gaze_total_frames": str(gaze["total_frames"]) if gaze["total_frames"] else "",
                "dominant_gaze": str(gaze["dominant"]),
                "pct_forward": str(gaze["pct"].get("Forward", "")),
                "pct_nonforward": str(gaze["pct"].get("Non-Forward", "")),
                "pct_incar": str(gaze["pct"].get("In-Car", "")),
                "pct_other": str(gaze["pct"].get("Other", "")),
                "offpath_ratio_valid": str(gaze["off_ratio"]),
                "n_windows": str(metrics["n_windows"]),
                "coverage_ok_windows": str(metrics["coverage_ok_windows"]),
                "coverage_fail_windows": str(metrics["coverage_fail_windows"]),
                "coverage_zero_windows": str(metrics["coverage_zero_windows"]),
                "mean_gaze_coverage_ratio": str(metrics["mean_gaze_coverage_ratio"]),
                "mean_pct_time_off_path": str(metrics["mean_pct_time_off_path"]),
                "mean_glance_rate_per_min": str(metrics["mean_glance_rate_per_min"]),
                "mean_offpath_ge_1p6s_per_window": str(metrics["mean_offpath_ge_1p6s_per_window"]),
                "mean_offpath_ge_2p0s_per_window": str(metrics["mean_offpath_ge_2p0s_per_window"]),
                "mean_wheel_on_ratio_overall": str(metrics["mean_wheel_on_ratio_overall"]),
                "gaze_updated": fmt_time(float(gaze["latest"])),
                "metrics_updated": fmt_time(float(metrics["latest"])),
            }
        )

    out_csv = Path(args.out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"participants={len(out_rows)}")
    print(f"out_csv={out_csv}")


if __name__ == "__main__":
    main()
