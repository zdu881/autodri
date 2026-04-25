#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parse run_p1_infer_plan / gaze_state_cls log text into per-segment summary JSON files."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse gaze inference log into summary json files")
    p.add_argument("--log", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


RE_CSV = re.compile(r"^CSV:\s+(.*)$")
RE_JSON = re.compile(r"^JSON:\s+(.*)$")
RE_COUNTS = re.compile(r"^Counts:\s+(\{.*\})$")
RE_PERCENT = re.compile(r"^Percent:\s+(\{.*\})$")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current: Dict[str, object] = {}
    saved = 0

    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        m = RE_COUNTS.match(line)
        if m:
            try:
                current["class_counts"] = ast.literal_eval(m.group(1))
            except Exception:
                current["class_counts"] = {}
            continue

        m = RE_PERCENT.match(line)
        if m:
            try:
                current["class_percent"] = ast.literal_eval(m.group(1))
            except Exception:
                current["class_percent"] = {}
            continue

        m = RE_CSV.match(line)
        if m:
            current["csv"] = m.group(1).strip()
            continue

        m = RE_JSON.match(line)
        if m:
            current["json_path"] = m.group(1).strip()
            csv_name = Path(str(current.get("csv", ""))).name
            if csv_name:
                out_path = out_dir / f"{csv_name}.summary.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "csv": current.get("csv", ""),
                            "class_counts": current.get("class_counts", {}),
                            "class_percent": current.get("class_percent", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                saved += 1
            current = {}

    print(f"saved={saved}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
