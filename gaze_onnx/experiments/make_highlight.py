#!/usr/bin/env python3
"""Extract representative clips from the full gaze classification video.

Picks segments where each class is dominant, concatenates them into
a single short highlight video with H.264 encoding (small file).
"""

import csv
import subprocess
import sys
import os

CSV_PATH = "gaze_onnx/output/output_gaze_cls_full.csv"
FULL_VIDEO = "gaze_onnx/output/output_gaze_cls_full.mp4"
OUT_VIDEO = "gaze_onnx/output/highlight_cls.mp4"
FPS = 30.0
CLIP_SEC = 8  # seconds per clip


def find_segments(csv_path: str, target_class: str, min_run: int = 120):
    """Find long runs of a given class in the CSV, return (start_frame, end_frame) list."""
    segments = []
    run_start = None
    run_len = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row["Gaze_Class"]
            fid = int(row["FrameID"])
            if cls == target_class:
                if run_start is None:
                    run_start = fid
                run_len += 1
            else:
                if run_start is not None and run_len >= min_run:
                    segments.append((run_start, run_start + run_len - 1))
                run_start = None
                run_len = 0
        if run_start is not None and run_len >= min_run:
            segments.append((run_start, run_start + run_len - 1))
    return segments


def main():
    # Find good segments for each class
    clips = []  # (label, start_sec, duration_sec)

    for cls_name in ["Forward", "In-Car", "Non-Forward"]:
        segs = find_segments(CSV_PATH, cls_name, min_run=int(CLIP_SEC * FPS * 0.6))
        if not segs:
            # relax requirement
            segs = find_segments(CSV_PATH, cls_name, min_run=30)
        if segs:
            # Pick the longest segment
            best = max(segs, key=lambda s: s[1] - s[0])
            mid_frame = (best[0] + best[1]) // 2
            start_frame = max(0, mid_frame - int(CLIP_SEC * FPS / 2))
            start_sec = start_frame / FPS
            clips.append((cls_name, start_sec, CLIP_SEC))
            print(f"  {cls_name}: frames {best[0]}-{best[1]} (len={best[1]-best[0]+1}), "
                  f"clip @ {start_sec:.1f}s")
        else:
            print(f"  {cls_name}: no segment found, skipping")

    if not clips:
        print("No clips found!")
        sys.exit(1)

    # Use ffmpeg to extract and concatenate clips
    tmp_files = []
    for i, (label, start, dur) in enumerate(clips):
        tmp = f"/tmp/clip_{i}_{label}.mp4"
        tmp_files.append(tmp)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-i", FULL_VIDEO,
            "-t", f"{dur:.2f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",  # no audio
            tmp,
        ]
        print(f"Extracting {label} clip: {start:.1f}s + {dur}s ...")
        subprocess.run(cmd, capture_output=True)

    # Write concat list
    concat_file = "/tmp/concat_list.txt"
    with open(concat_file, "w") as f:
        for tmp in tmp_files:
            f.write(f"file '{tmp}'\n")

    # Concatenate
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        OUT_VIDEO,
    ]
    print("Concatenating clips...")
    subprocess.run(cmd, capture_output=True)

    # Cleanup
    for tmp in tmp_files:
        os.remove(tmp)
    os.remove(concat_file)

    size_mb = os.path.getsize(OUT_VIDEO) / (1024 * 1024)
    print(f"\nDone! Output: {OUT_VIDEO} ({size_mb:.1f} MB)")
    print(f"Clips: {len(clips)} x {CLIP_SEC}s = {len(clips) * CLIP_SEC}s total")


if __name__ == "__main__":
    main()
