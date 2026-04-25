#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build trimmed fixed windows from unified target segments for one participant."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed windows from unified target segments")
    p.add_argument("--segments-csv", required=True, help="Unified target segments CSV")
    p.add_argument("--participant", required=True, help="Participant id, e.g. p2")
    p.add_argument("--window-sec", type=float, default=20.0)
    p.add_argument("--trim-sec", type=float, default=60.0, help="Trim from both start/end")
    p.add_argument("--segments-out", required=True, help="Output parsed segment CSV")
    p.add_argument("--windows-out", required=True, help="Output fixed-window CSV")
    return p.parse_args()


def sec_to_hhmmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    s = int(round(sec))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def load_segments(path: Path, participant: str) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = [
        r
        for r in rows
        if str(r.get("participant", "")).strip().lower() == participant.lower()
    ]
    return out


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
                    "participant": seg["participant"],
                    "source_xlsx": seg["source_xlsx"],
                    "source_sheet": seg["source_sheet"],
                    "video_label_raw": seg["sheet_label"],
                    "video_folder_name": seg["matched_folder"],
                    "video_path": seg["video_path"],
                    "video_map_status": seg["match_status"],
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
            csv.writer(f).writerow(["empty"])
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    segments = load_segments(Path(args.segments_csv), participant=args.participant)
    if not segments:
        raise SystemExit(f"No segment rows found for participant={args.participant}")

    matched_segments = [r for r in segments if str(r.get("video_path", "")).strip()]
    seg_rows, win_rows = build_windows(
        segments=matched_segments,
        trim_sec=float(args.trim_sec),
        window_sec=float(args.window_sec),
    )

    write_csv(Path(args.segments_out), seg_rows)
    write_csv(Path(args.windows_out), win_rows)

    n_valid = sum(1 for r in seg_rows if r.get("valid_after_trim", "0") == "1")
    print(f"participant={args.participant} segments={len(seg_rows)} valid_after_trim={n_valid}")
    print(f"windows={len(win_rows)}")
    print(f"segments_out={args.segments_out}")
    print(f"windows_out={args.windows_out}")


if __name__ == "__main__":
    main()
