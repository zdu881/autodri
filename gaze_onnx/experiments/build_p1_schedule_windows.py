#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse p1 schedule CSV and build trimmed 20s windows for analysis.

Rules (confirmed):
1) Use provided AD start/end ranges per video.
2) Exclude first 60s after AD start and last 60s before AD end.
3) Split remaining interval into non-overlap fixed windows (default: 20s).
4) Drop segments shorter than 120s (no effective interval).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


TIME_RANGE_RE = re.compile(
    r"([0-9]{1,2}[：:][0-9]{2}(?:[：:][0-9]{2})?)\s*[-~—–－到]+\s*([0-9]{1,2}[：:][0-9]{2}(?:[：:][0-9]{2})?)"
)
RANGE_TOKEN_RE = re.compile(r"([0-9]{6}-[0-9]{6})")
DATE_TOKEN_RE = re.compile(r"(?:(20[0-9]{2})[./-]?([0-9]{2})[./-]?([0-9]{2})|([0-9]{1,2})[.]([0-9]{1,2}))")
UUID_LIKE_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class VideoEntry:
    folder_name: str
    folder_path: str
    video_path: str
    range_token: str
    mmdd: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build p1 schedule segments and 20s windows")
    p.add_argument("--schedule-csv", required=True, help="Raw p1 schedule CSV path")
    p.add_argument(
        "--videos-root",
        default="data/natural_driving_p1/p1_剪辑好的视频",
        help="Root directory containing p1 video folders",
    )
    p.add_argument("--window-sec", type=float, default=20.0)
    p.add_argument("--trim-sec", type=float, default=60.0, help="Trim from both start/end")
    p.add_argument(
        "--segments-out",
        default="data/natural_driving_p1/analysis/p1_segments.parsed.csv",
        help="Output parsed segment CSV",
    )
    p.add_argument(
        "--windows-out",
        default="data/natural_driving_p1/analysis/p1_windows.20s.csv",
        help="Output fixed-window CSV",
    )
    return p.parse_args()


def parse_time_token(token: str) -> Optional[float]:
    t = (token or "").strip().replace("：", ":")
    if not t:
        return None
    parts = t.split(":")
    try:
        vals = [int(x) for x in parts]
    except Exception:
        return None
    if len(vals) == 2:
        m, s = vals
        if s < 0 or s >= 60:
            return None
        return float(m * 60 + s)
    if len(vals) == 3:
        h, m, s = vals
        if m < 0 or m >= 60 or s < 0 or s >= 60:
            return None
        return float(h * 3600 + m * 60 + s)
    return None


def sec_to_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    s = int(round(sec))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_mmdd(text: str) -> str:
    m = DATE_TOKEN_RE.search(text or "")
    if not m:
        return ""
    if m.group(1) and m.group(2) and m.group(3):
        mm = int(m.group(2))
        dd = int(m.group(3))
    else:
        mm = int(m.group(4))
        dd = int(m.group(5))
    return f"{mm:02d}.{dd:02d}"


def extract_range_token(text: str) -> str:
    m = RANGE_TOKEN_RE.search(text or "")
    return m.group(1) if m else ""


def choose_video_file(folder: Path) -> Optional[Path]:
    cands = sorted([p for p in folder.glob("*.mp4") if p.is_file()])
    if not cands:
        return None
    non_uuid = [p for p in cands if not UUID_LIKE_RE.match(p.stem)]
    if non_uuid:
        # Prefer a human-readable named file if available.
        return sorted(non_uuid, key=lambda x: (len(x.name), x.name))[0]
    return cands[0]


def build_video_index(videos_root: Path) -> List[VideoEntry]:
    out: List[VideoEntry] = []
    for folder in sorted(videos_root.glob("*/*")):
        if not folder.is_dir():
            continue
        video_file = choose_video_file(folder)
        if video_file is None:
            continue
        folder_name = folder.name
        out.append(
            VideoEntry(
                folder_name=folder_name,
                folder_path=str(folder),
                video_path=str(video_file),
                range_token=extract_range_token(folder_name),
                mmdd=extract_mmdd(folder_name),
            )
        )
    return out


def resolve_video(label: str, index: List[VideoEntry]) -> Tuple[str, str, str]:
    """Return (video_path, folder_name, status)."""
    label_n = normalize_label(label)
    if not label_n:
        return "", "", "missing_label"

    # Direct folder-name match first.
    for e in index:
        if normalize_label(e.folder_name) == label_n:
            return e.video_path, e.folder_name, "exact"

    range_tok = extract_range_token(label_n)
    mmdd = extract_mmdd(label_n)

    if range_tok:
        cands = [e for e in index if e.range_token == range_tok]
        if len(cands) == 1:
            return cands[0].video_path, cands[0].folder_name, "range"
        if len(cands) > 1 and mmdd:
            cands2 = [e for e in cands if e.mmdd == mmdd]
            if len(cands2) == 1:
                return cands2[0].video_path, cands2[0].folder_name, "range+date"
            if cands2:
                return cands2[0].video_path, cands2[0].folder_name, "range+date_ambiguous"
            return cands[0].video_path, cands[0].folder_name, "range_ambiguous"

    # Fallback: contains relation.
    cands = [e for e in index if range_tok and range_tok in e.folder_name]
    if len(cands) == 1:
        return cands[0].video_path, cands[0].folder_name, "contains"

    return "", "", "unmatched"


def parse_schedule_rows(schedule_csv: Path, index: List[VideoEntry]) -> List[Dict[str, str]]:
    df = pd.read_csv(schedule_csv)
    if "视频文件夹" not in df.columns or "备注" not in df.columns:
        raise ValueError(f"{schedule_csv} missing required columns: 视频文件夹, 备注")

    rows: List[Dict[str, str]] = []
    current_label = ""
    seg_uid = 0

    for i, row in df.iterrows():
        raw_label = normalize_label("" if pd.isna(row["视频文件夹"]) else str(row["视频文件夹"]))
        remark = "" if pd.isna(row["备注"]) else str(row["备注"])
        remark = remark.strip()

        if raw_label:
            current_label = raw_label
        label = current_label

        if not label:
            continue
        if not remark:
            continue

        matches = list(TIME_RANGE_RE.finditer(remark))
        if not matches:
            continue

        video_path, folder_name, map_status = resolve_video(label, index)
        skip_hint = ("没到一分钟" in remark) or ("没有" in remark)

        for j, m in enumerate(matches):
            t0s = parse_time_token(m.group(1))
            t1s = parse_time_token(m.group(2))
            if t0s is None or t1s is None:
                continue
            if t1s <= t0s:
                continue
            seg_uid += 1
            rows.append(
                {
                    "segment_uid": f"seg_{seg_uid:05d}",
                    "src_row_idx": str(int(i)),
                    "segment_idx_in_row": str(int(j)),
                    "video_label_raw": label,
                    "video_folder_name": folder_name,
                    "video_path": video_path,
                    "video_map_status": map_status,
                    "remark_raw": remark.replace("\n", " | "),
                    "remark_has_short_hint": "1" if skip_hint else "0",
                    "start_text": m.group(1).replace("：", ":"),
                    "end_text": m.group(2).replace("：", ":"),
                    "start_sec": f"{t0s:.3f}",
                    "end_sec": f"{t1s:.3f}",
                    "duration_sec": f"{(t1s - t0s):.3f}",
                }
            )
    return rows


def build_windows(
    segments: List[Dict[str, str]], trim_sec: float, window_sec: float
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    seg_out: List[Dict[str, str]] = []
    win_out: List[Dict[str, str]] = []

    for seg in segments:
        t0 = float(seg["start_sec"])
        t1 = float(seg["end_sec"])
        raw_dur = t1 - t0
        eff_start = t0 + trim_sec
        eff_end = t1 - trim_sec
        eff_dur = eff_end - eff_start
        ok = eff_dur >= window_sec and raw_dur > 2.0 * trim_sec
        nwin = int(math.floor(eff_dur / window_sec)) if ok else 0

        seg2 = dict(seg)
        seg2["trim_sec"] = f"{trim_sec:.3f}"
        seg2["window_sec"] = f"{window_sec:.3f}"
        seg2["effective_start_sec"] = f"{eff_start:.3f}"
        seg2["effective_end_sec"] = f"{eff_end:.3f}"
        seg2["effective_duration_sec"] = f"{eff_dur:.3f}"
        seg2["effective_start_hhmmss"] = sec_to_hhmmss(eff_start)
        seg2["effective_end_hhmmss"] = sec_to_hhmmss(eff_end)
        seg2["valid_after_trim"] = "1" if ok else "0"
        seg2["window_count"] = str(int(nwin))
        seg_out.append(seg2)

        if not ok:
            continue

        for k in range(nwin):
            ws = eff_start + k * window_sec
            we = ws + window_sec
            win_out.append(
                {
                    "window_uid": f"{seg['segment_uid']}_w{k:03d}",
                    "segment_uid": seg["segment_uid"],
                    "window_index_in_segment": str(int(k)),
                    "video_label_raw": seg["video_label_raw"],
                    "video_folder_name": seg["video_folder_name"],
                    "video_path": seg["video_path"],
                    "video_map_status": seg["video_map_status"],
                    "window_start_sec": f"{ws:.3f}",
                    "window_end_sec": f"{we:.3f}",
                    "window_duration_sec": f"{window_sec:.3f}",
                    "window_start_hhmmss": sec_to_hhmmss(ws),
                    "window_end_hhmmss": sec_to_hhmmss(we),
                    "segment_start_sec_raw": seg["start_sec"],
                    "segment_end_sec_raw": seg["end_sec"],
                }
            )

    return seg_out, win_out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["empty"])
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    schedule_csv = Path(args.schedule_csv)
    videos_root = Path(args.videos_root)
    segments_out = Path(args.segments_out)
    windows_out = Path(args.windows_out)

    if not schedule_csv.exists():
        raise FileNotFoundError(schedule_csv)
    if not videos_root.exists():
        raise FileNotFoundError(videos_root)

    index = build_video_index(videos_root)
    segments = parse_schedule_rows(schedule_csv, index)
    seg_rows, win_rows = build_windows(
        segments=segments,
        trim_sec=float(args.trim_sec),
        window_sec=float(args.window_sec),
    )

    write_csv(segments_out, seg_rows)
    write_csv(windows_out, win_rows)

    n_match = sum(1 for r in seg_rows if r.get("video_path", ""))
    n_valid = sum(1 for r in seg_rows if r.get("valid_after_trim", "0") == "1")
    print(f"Videos indexed: {len(index)}")
    print(f"Parsed segments: {len(seg_rows)}  mapped={n_match}  valid_after_trim={n_valid}")
    print(f"Windows ({float(args.window_sec):.1f}s): {len(win_rows)}")
    print(f"Segments CSV: {segments_out}")
    print(f"Windows CSV:  {windows_out}")


if __name__ == "__main__":
    main()

