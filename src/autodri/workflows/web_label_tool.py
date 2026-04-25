#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Web labeling tool for SSH / headless servers.

Why:
- `label_tool.py` uses OpenCV windows which require a GUI.
- Over SSH, the simplest workflow is: run a small web server on the remote box,
  then open it in your local browser via SSH port-forwarding.

Usage (recommended):
  # On your *local* machine:
  ssh -L 8000:127.0.0.1:8000 <user>@<server>

  # On the *server* (in repo root):
  conda run -n adri python gaze_onnx/experiments/web_label_tool.py \
    --samples-dir gaze_onnx/experiments/samples_smooth4_full_500 \
    --port 8000

Then open in your local browser:
  http://127.0.0.1:8000

Keys:
- 1: Forward
- 2: Non-Forward
- 3: In-Car
- 4 / o: Other
- 0: Unknown
- n: Next
- b: Back

Outputs:
- Writes/updates `labels.csv` in the samples directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import io
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2


@dataclass(frozen=True)
class Sample:
    idx: int
    img: str
    frame_id: int
    timestamp: float
    domain: str
    video: str
    pred_class: str
    raw_pitch: str
    raw_yaw: str
    smooth_pitch: str
    smooth_yaw: str
    ref_pitch: str
    ref_yaw: str
    delta_pitch: str
    delta_yaw: str
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    confidence: str


LABELS = ["Forward", "Non-Forward", "In-Car", "Other", "Unknown"]
LABEL_KEYS = {
    "1": "Forward",
    "2": "Non-Forward",
    "3": "In-Car",
    "4": "Other",
    "o": "Other",
    "O": "Other",
    "0": "Unknown",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web labeling tool (SSH-friendly)")
    p.add_argument("--samples-dir", required=True, help="Directory with PNGs + manifest.csv")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def load_manifest(samples_dir: Path) -> List[Sample]:
    manifest = samples_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest}")

    out: List[Sample] = []
    with manifest.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = ["img", "FrameID", "Timestamp"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise ValueError(f"manifest.csv missing column '{k}'. Found: {r.fieldnames}")

        rows = list(r)

    # Preserve manifest order by default. If a future manifest provides SortKey,
    # it can already be honored here without breaking older packs.
    if "SortKey" in (rows[0].keys() if rows else []):
        rows.sort(key=lambda row: int(float(str(row.get("SortKey", "0")) or "0")))
    for i, row in enumerate(rows):
        out.append(
            Sample(
                idx=i,
                img=str(row["img"]),
                frame_id=int(float(row["FrameID"])),
                timestamp=float(row["Timestamp"]),
                domain=str(row.get("Domain", "")),
                video=str(row.get("Video", "")),
                pred_class=str(row.get("Pred_Class", "")),
                raw_pitch=str(row.get("Raw_Pitch", "")),
                raw_yaw=str(row.get("Raw_Yaw", "")),
                smooth_pitch=str(row.get("Smooth_Pitch", "")),
                smooth_yaw=str(row.get("Smooth_Yaw", "")),
                ref_pitch=str(row.get("Ref_Pitch", "")),
                ref_yaw=str(row.get("Ref_Yaw", "")),
                delta_pitch=str(row.get("Delta_Pitch", "")),
                delta_yaw=str(row.get("Delta_Yaw", "")),
                roi_x1=int(float(str(row.get("ROI_X1", "0")) or "0")),
                roi_y1=int(float(str(row.get("ROI_Y1", "0")) or "0")),
                roi_x2=int(float(str(row.get("ROI_X2", "0")) or "0")),
                roi_y2=int(float(str(row.get("ROI_Y2", "0")) or "0")),
                confidence=str(row.get("Confidence", "")),
            )
        )

    return out


def load_labels(samples_dir: Path) -> Dict[str, str]:
    labels_path = samples_dir / "labels.csv"
    if not labels_path.exists():
        return {}

    out: Dict[str, str] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        # Compatible with a few possible schemas.
        for row in r:
            img = (row.get("img") or row.get("image") or row.get("filename") or "").strip()
            lab = (row.get("label") or row.get("Label") or row.get("Human_Label") or row.get("Gaze_Class") or "").strip()
            if img and lab:
                out[img] = lab
    return out


def save_labels(samples_dir: Path, labels_by_img: Dict[str, str], samples: List[Sample]) -> None:
    labels_path = samples_dir / "labels.csv"
    tmp_path = samples_dir / ".labels.csv.tmp"

    # Write in manifest order for stability.
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "label", "FrameID", "Timestamp", "Pred_Class", "Domain", "Video"])
        for s in samples:
            lab = labels_by_img.get(s.img, "")
            w.writerow([s.img, lab, s.frame_id, f"{s.timestamp:.3f}", s.pred_class, s.domain, s.video])

    os.replace(tmp_path, labels_path)


def _page(title: str, body: str) -> bytes:
        # NOTE: Avoid f-strings here because the HTML/JS contains many `{}` braces.
        # We use simple token replacement to prevent accidental Python formatting.
        doc = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>__TITLE__</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; }}
    .wrap {{ display: grid; grid-template-columns: 1fr; gap: 12px; max-width: 1100px; margin: 0 auto; }}
    .bar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .btn {{ padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fafafa; cursor: pointer; text-decoration: none; color: #111; }}
    .btn:hover {{ background: #f0f0f0; }}
    .btn.primary {{ border-color: #2d6cdf; background: #2d6cdf; color: white; }}
    .btn.danger {{ border-color: #c62828; background: #c62828; color: white; }}
    .meta {{ padding: 10px 12px; border: 1px solid #eee; border-radius: 12px; background: #fcfcfc; }}
    .k {{ font-weight: 600; }}
    img {{ max-width: 100%; height: auto; border-radius: 12px; border: 1px solid #eee; }}
    .hint {{ color: #666; font-size: 13px; }}
  </style>
</head>
<body>
<div class=\"wrap\">
__BODY__
</div>
<script>
  window.addEventListener('keydown', (e) => {{
    const k = e.key;
    const m = {{'1':'Forward','2':'Non-Forward','3':'In-Car','4':'Other','o':'Other','O':'Other','0':'Unknown'}};
    if (k in m) {{
      const a = document.querySelector(`a[data-label='${m[k]}']`);
      if (a) window.location = a.href;
    }} else if (k === 'n') {{
      const a = document.querySelector('a[data-nav="next"]');
      if (a) window.location = a.href;
    }} else if (k === 'b') {{
      const a = document.querySelector('a[data-nav="back"]');
      if (a) window.location = a.href;
    }}
  }});
</script>
</body>
</html>"""
        doc = doc.replace("__TITLE__", html.escape(title)).replace("__BODY__", body)
        return doc.encode("utf-8")


class App:
    def __init__(self, samples_dir: Path):
        self.samples_dir = samples_dir
        self.samples = load_manifest(samples_dir)
        self.labels_by_img = load_labels(samples_dir)
        self.sample_by_img = {s.img: s for s in self.samples}

    def get(self, idx: int) -> Sample:
        idx2 = max(0, min(len(self.samples) - 1, int(idx)))
        return self.samples[idx2]

    def progress(self) -> Tuple[int, int]:
        done = sum(1 for s in self.samples if self.labels_by_img.get(s.img))
        return done, len(self.samples)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _render_dynamic_crop(self, s: Sample) -> Optional[bytes]:
            video = Path(s.video)
            if not video.exists():
                return None
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.frame_id))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            h, w = frame.shape[:2]
            x1 = max(0, min(int(s.roi_x1), max(0, w - 1)))
            y1 = max(0, min(int(s.roi_y1), max(0, h - 1)))
            x2 = max(1, min(int(s.roi_x2), max(1, w)))
            y2 = max(1, min(int(s.roi_y2), max(1, h)))
            if x2 <= x1 or y2 <= y1:
                crop = frame
            else:
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    crop = frame
            ok_enc, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok_enc:
                return None
            return bytes(buf)

        def do_GET(self):  # noqa: N802
            try:
                u = urlparse(self.path)
                path = u.path
                qs = parse_qs(u.query)

                if path in ("/", ""):
                    self._redirect("/item/0")
                    return

                if path.startswith("/img/"):
                    name = unquote(path[len("/img/") :])
                    img_path = (app.samples_dir / name).resolve()
                    data = None
                    ctype = "image/jpeg"
                    if img_path.exists() and img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        if app.samples_dir.resolve() not in img_path.parents:
                            self._send_text(HTTPStatus.FORBIDDEN, "forbidden")
                            return
                        data = img_path.read_bytes()
                        ctype = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                    else:
                        s = app.sample_by_img.get(name)
                        if s is None:
                            self._send_text(HTTPStatus.NOT_FOUND, "not found")
                            return
                        data = self._render_dynamic_crop(s)
                        if data is None:
                            self._send_text(HTTPStatus.NOT_FOUND, "failed to render frame")
                            return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                if path.startswith("/set"):
                    try:
                        idx = int(qs.get("idx", ["0"])[0])
                    except Exception:
                        idx = 0
                    label = (qs.get("label", [""])[0] or "").strip()
                    if label not in LABELS:
                        self._send_text(HTTPStatus.BAD_REQUEST, "bad label")
                        return

                    s = app.get(idx)
                    if label == "Unknown":
                        app.labels_by_img.pop(s.img, None)
                    else:
                        app.labels_by_img[s.img] = label

                    save_labels(app.samples_dir, app.labels_by_img, app.samples)

                    # navigate
                    nav = (qs.get("nav", ["next"])[0] or "next").strip()
                    if nav == "back":
                        self._redirect(f"/item/{max(0, idx - 1)}")
                    else:
                        self._redirect(f"/item/{min(len(app.samples) - 1, idx + 1)}")
                    return

                if path.startswith("/item/"):
                    try:
                        idx = int(path[len("/item/") :])
                    except Exception:
                        idx = 0

                    s = app.get(idx)
                    done, total = app.progress()
                    current_label = app.labels_by_img.get(s.img, "")

                    def link_set(label: str, nav: str = "next") -> str:
                        return f"/set?idx={s.idx}&label={quote(label)}&nav={quote(nav)}"

                    body = "".join(
                        [
                            f"<div class='meta'>"
                            f"<div><span class='k'>Progress:</span> {done}/{total}</div>"
                            f"<div><span class='k'>Index:</span> {s.idx+1}/{total}</div>"
                            f"<div><span class='k'>Frame:</span> {s.frame_id} &nbsp; <span class='k'>t:</span> {s.timestamp:.3f}s</div>"
                            f"<div><span class='k'>Domain:</span> {html.escape(s.domain or '-')} &nbsp; <span class='k'>Video:</span> {html.escape(s.video or '-')}</div>"
                            f"<div><span class='k'>Pred:</span> {html.escape(s.pred_class)}"
                            f" &nbsp; <span class='k'>Conf:</span> {html.escape(s.confidence or '-')}"
                            f" &nbsp; <span class='k'>Label:</span> {html.escape(current_label or '(unlabeled)')}</div>"
                            f"<div class='hint'>Keys: 1=F 2=N 3=I 4=Other 0=Unknown, n=next, b=back</div>"
                            f"</div>",
                            "<div class='bar'>"
                            f"<a class='btn' data-nav='back' href='/item/{max(0, s.idx-1)}'>Back (b)</a>"
                            f"<a class='btn' data-nav='next' href='/item/{min(len(app.samples)-1, s.idx+1)}'>Next (n)</a>"
                            "</div>",
                            "<div class='bar'>"
                            f"<a class='btn primary' data-label='Forward' href='{html.escape(link_set('Forward'))}'>1 Forward</a>"
                            f"<a class='btn primary' data-label='Non-Forward' href='{html.escape(link_set('Non-Forward'))}'>2 Non-Forward</a>"
                            f"<a class='btn primary' data-label='In-Car' href='{html.escape(link_set('In-Car'))}'>3 In-Car</a>"
                            f"<a class='btn primary' data-label='Other' href='{html.escape(link_set('Other'))}'>4 Other</a>"
                            f"<a class='btn danger' data-label='Unknown' href='{html.escape(link_set('Unknown'))}'>0 Unknown/Clear</a>"
                            "</div>",
                            f"<div class='meta'>"
                            f"<div><span class='k'>raw:</span> pitch={html.escape(s.raw_pitch)} yaw={html.escape(s.raw_yaw)}</div>"
                            f"<div><span class='k'>smooth:</span> pitch={html.escape(s.smooth_pitch)} yaw={html.escape(s.smooth_yaw)}</div>"
                            f"<div><span class='k'>ref:</span> pitch={html.escape(s.ref_pitch)} yaw={html.escape(s.ref_yaw)}</div>"
                            f"<div><span class='k'>delta:</span> pitch={html.escape(s.delta_pitch)} yaw={html.escape(s.delta_yaw)}</div>"
                            f"</div>",
                            f"<div><img src='/img/{quote(s.img)}' alt='{html.escape(s.img)}'/></div>",
                        ]
                    )

                    data = _page("Label", body)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                self._send_text(HTTPStatus.NOT_FOUND, "not found")
            except Exception:
                tb = traceback.format_exc()
                try:
                    sys.stderr.write(tb + "\n")
                    sys.stderr.flush()
                except Exception:
                    pass
                self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, tb)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # keep server quiet
            return

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.end_headers()

        def _send_text(self, status: HTTPStatus, msg: str) -> None:
            b = msg.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    return Handler


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    app = App(samples_dir)

    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(app))
    print(f"Serving labels for: {samples_dir}")
    print(f"Open (via SSH port-forward): http://127.0.0.1:{int(args.port)}/")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
