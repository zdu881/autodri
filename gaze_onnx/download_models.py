#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model downloader helper.

What it can do automatically:
- Download SCRFD face detector ONNX from InsightFace GitHub release.

What it cannot do automatically (in many locked-down environments):
- Download from Google Drive (often blocked)
- Download from Baidu Pan without login/cookies/tools

This script prints clear next steps for those cases.
"""

import argparse
import os
import sys
import urllib.request


SCRFD_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_person_2.5g.onnx"


def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "autodri"})
    with urllib.request.urlopen(req) as r, open(out_path, "wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                sys.stdout.write(f"\rDownloading {os.path.basename(out_path)}: {pct:5.1f}%")
                sys.stdout.flush()
        if total:
            sys.stdout.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Download ONNX models into ./models")
    p.add_argument("--models-dir", default="../models", help="Target models directory")
    args = p.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    scrfd_path = os.path.join(models_dir, "scrfd_person_2.5g.onnx")
    if os.path.exists(scrfd_path) and os.path.getsize(scrfd_path) > 1024 * 1024:
        print(f"SCRFD already exists: {scrfd_path}")
    else:
        print(f"Downloading SCRFD -> {scrfd_path}")
        download(SCRFD_URL, scrfd_path)
        print(f"Saved: {scrfd_path}")

    print("\nL2CS model notes:")
    print("- Your script expects an ONNX named L2CSNet_gaze360.onnx")
    print("- In this environment, Google Drive is unreachable (gdown fails).")
    print("- A public repo README references an ONNX Baidu Pan link:")
    print("  https://pan.baidu.com/s/1r3rPRCM8AjNk_eLEOnVDzw?pwd=to14")
    print("  Download it manually and place it as:")
    print(f"  {os.path.join(models_dir, 'L2CSNet_gaze360.onnx')}")


if __name__ == "__main__":
    main()
