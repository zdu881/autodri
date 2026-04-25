#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Serve a dual-ROI review pack over HTTP.

Supports editing both gaze ROI and wheel ROI on the same page.
Writes:
- roi_label_manifest.csv
- p1_gaze_rois.manual.csv
- p1_wheel_rois.manual.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2


GAZE_COLOR = (0, 200, 255)
WHEEL_COLOR = (255, 160, 0)


@dataclass(frozen=True)
class Item:
    idx: int
    video_rel: str
    video_abs: str
    ref_raw: str
    ref_grid: str
    frame_idx: int
    timestamp_sec: float
    width: int
    height: int
    gaze_roi_x1: str
    gaze_roi_y1: str
    gaze_roi_x2: str
    gaze_roi_y2: str
    wheel_roi_x1: str
    wheel_roi_y1: str
    wheel_roi_x2: str
    wheel_roi_y2: str
    roi_note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve dual ROI review pack")
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8226)
    return p.parse_args()


def load_items(pack_dir: Path) -> list[Item]:
    manifest = pack_dir / "roi_label_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        out.append(
            Item(
                idx=i,
                video_rel=str(r.get("video_rel", "")),
                video_abs=str(r.get("video_abs", "")),
                ref_raw=str(r.get("ref_raw", "")),
                ref_grid=str(r.get("ref_grid", "")),
                frame_idx=int(float(str(r.get("frame_idx", "0")) or "0")),
                timestamp_sec=float(str(r.get("timestamp_sec", "0")) or "0"),
                width=int(float(str(r.get("width", "0")) or "0")),
                height=int(float(str(r.get("height", "0")) or "0")),
                gaze_roi_x1=str(r.get("gaze_roi_x1", "")),
                gaze_roi_y1=str(r.get("gaze_roi_y1", "")),
                gaze_roi_x2=str(r.get("gaze_roi_x2", "")),
                gaze_roi_y2=str(r.get("gaze_roi_y2", "")),
                wheel_roi_x1=str(r.get("wheel_roi_x1", "")),
                wheel_roi_y1=str(r.get("wheel_roi_y1", "")),
                wheel_roi_x2=str(r.get("wheel_roi_x2", "")),
                wheel_roi_y2=str(r.get("wheel_roi_y2", "")),
                roi_note=str(r.get("roi_note", "")),
            )
        )
    return out


def write_export_csvs(pack_dir: Path, items: list[Item]) -> None:
    gaze_csv = pack_dir / "p1_gaze_rois.manual.csv"
    wheel_csv = pack_dir / "p1_wheel_rois.manual.csv"
    fields = [
        "domain_id",
        "video",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "n_samples",
        "source_swapped",
        "source_uncertain",
        "roi_note",
    ]
    with gaze_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in items:
            w.writerow(
                {
                    "domain_id": "p1",
                    "video": it.video_abs,
                    "roi_x1": it.gaze_roi_x1,
                    "roi_y1": it.gaze_roi_y1,
                    "roi_x2": it.gaze_roi_x2,
                    "roi_y2": it.gaze_roi_y2,
                    "n_samples": "1",
                    "source_swapped": "",
                    "source_uncertain": "",
                    "roi_note": it.roi_note,
                }
            )
    with wheel_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in items:
            w.writerow(
                {
                    "domain_id": "p1",
                    "video": it.video_abs,
                    "roi_x1": it.wheel_roi_x1,
                    "roi_y1": it.wheel_roi_y1,
                    "roi_x2": it.wheel_roi_x2,
                    "roi_y2": it.wheel_roi_y2,
                    "n_samples": "1",
                    "source_swapped": "",
                    "source_uncertain": "",
                    "roi_note": it.roi_note,
                }
            )


def save_items(pack_dir: Path, items: list[Item]) -> None:
    manifest = pack_dir / "roi_label_manifest.csv"
    tmp = pack_dir / ".roi_label_manifest.csv.tmp"
    fields = [
        "video_rel",
        "video_abs",
        "ref_raw",
        "ref_grid",
        "frame_idx",
        "timestamp_sec",
        "width",
        "height",
        "gaze_roi_x1",
        "gaze_roi_y1",
        "gaze_roi_x2",
        "gaze_roi_y2",
        "wheel_roi_x1",
        "wheel_roi_y1",
        "wheel_roi_x2",
        "wheel_roi_y2",
        "roi_note",
    ]
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in items:
            w.writerow(
                {
                    "video_rel": x.video_rel,
                    "video_abs": x.video_abs,
                    "ref_raw": x.ref_raw,
                    "ref_grid": x.ref_grid,
                    "frame_idx": x.frame_idx,
                    "timestamp_sec": f"{x.timestamp_sec:.3f}",
                    "width": x.width,
                    "height": x.height,
                    "gaze_roi_x1": x.gaze_roi_x1,
                    "gaze_roi_y1": x.gaze_roi_y1,
                    "gaze_roi_x2": x.gaze_roi_x2,
                    "gaze_roi_y2": x.gaze_roi_y2,
                    "wheel_roi_x1": x.wheel_roi_x1,
                    "wheel_roi_y1": x.wheel_roi_y1,
                    "wheel_roi_x2": x.wheel_roi_x2,
                    "wheel_roi_y2": x.wheel_roi_y2,
                    "roi_note": x.roi_note,
                }
            )
    os.replace(tmp, manifest)
    write_export_csvs(pack_dir, items)


def page(title: str, body: str) -> bytes:
    doc = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 16px; background: #f7f7f7; color: #111; }
.wrap { max-width: 1380px; margin: 0 auto; display: grid; gap: 12px; }
.meta, .form { border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: #fff; }
.btn { padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fafafa; text-decoration: none; color: #111; display: inline-block; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.tag { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.gaze { background: #fff4d6; color: #8a5a00; }
.wheel { background: #e2f0ff; color: #004f8a; }
label { font-weight: 600; display: block; }
input[type=text] { width: 110px; padding: 8px; }
textarea { width: 100%; min-height: 70px; padding: 8px; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 12px; background: #fff; }
</style></head><body><div class="wrap">__BODY__</div></body></html>"""
    return doc.replace("__TITLE__", html.escape(title)).replace("__BODY__", body).encode("utf-8")


class App:
    def __init__(self, pack_dir: Path):
        self.pack_dir = pack_dir
        self.items = load_items(pack_dir)
        write_export_csvs(pack_dir, self.items)

    def get(self, idx: int) -> Item:
        idx = max(0, min(len(self.items) - 1, idx))
        return self.items[idx]


def to_int_or_none(v: str):
    s = str(v or "").strip()
    if not s:
        return None
    return int(float(s))


def draw_roi(img, roi, color, label):
    if any(v is None for v in roi):
        return
    x1, y1, x2, y2 = roi
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, label, (x1 + 10, max(30, y1 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _overlay_bytes(self, it: Item) -> bytes:
            raw = (app.pack_dir / it.ref_raw).resolve()
            img = cv2.imread(str(raw))
            if img is None:
                raise FileNotFoundError(raw)
            gaze_roi = (
                to_int_or_none(it.gaze_roi_x1),
                to_int_or_none(it.gaze_roi_y1),
                to_int_or_none(it.gaze_roi_x2),
                to_int_or_none(it.gaze_roi_y2),
            )
            wheel_roi = (
                to_int_or_none(it.wheel_roi_x1),
                to_int_or_none(it.wheel_roi_y1),
                to_int_or_none(it.wheel_roi_x2),
                to_int_or_none(it.wheel_roi_y2),
            )
            draw_roi(img, gaze_roi, GAZE_COLOR, "GAZE")
            draw_roi(img, wheel_roi, WHEEL_COLOR, "WHEEL")
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise RuntimeError("failed to encode overlay image")
            return bytes(buf)

        def do_GET(self):  # noqa: N802
            try:
                u = urlparse(self.path)
                path = u.path
                qs = parse_qs(u.query)
                if path in ("", "/"):
                    self._redirect("/item/0")
                    return
                if path.startswith("/img/"):
                    rel = unquote(path[len("/img/"):])
                    f = (app.pack_dir / rel).resolve()
                    if app.pack_dir.resolve() not in f.parents or not f.exists():
                        self._send_text(HTTPStatus.NOT_FOUND, "not found")
                        return
                    data = f.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path.startswith("/file/"):
                    rel = unquote(path[len("/file/"):])
                    f = (app.pack_dir / rel).resolve()
                    if app.pack_dir.resolve() not in f.parents or not f.exists():
                        self._send_text(HTTPStatus.NOT_FOUND, "not found")
                        return
                    data = f.read_bytes()
                    ctype = "text/csv; charset=utf-8" if f.suffix.lower() == ".csv" else "application/octet-stream"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path.startswith("/overlay/"):
                    idx = int(path[len("/overlay/"):] or "0")
                    it = app.get(idx)
                    data = self._overlay_bytes(it)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path.startswith("/save"):
                    idx = int((qs.get("idx", ["0"])[0] or "0"))
                    item = app.get(idx)
                    new_item = Item(
                        idx=item.idx,
                        video_rel=item.video_rel,
                        video_abs=item.video_abs,
                        ref_raw=item.ref_raw,
                        ref_grid=item.ref_grid,
                        frame_idx=item.frame_idx,
                        timestamp_sec=item.timestamp_sec,
                        width=item.width,
                        height=item.height,
                        gaze_roi_x1=(qs.get("gaze_roi_x1", [item.gaze_roi_x1])[0] or "").strip(),
                        gaze_roi_y1=(qs.get("gaze_roi_y1", [item.gaze_roi_y1])[0] or "").strip(),
                        gaze_roi_x2=(qs.get("gaze_roi_x2", [item.gaze_roi_x2])[0] or "").strip(),
                        gaze_roi_y2=(qs.get("gaze_roi_y2", [item.gaze_roi_y2])[0] or "").strip(),
                        wheel_roi_x1=(qs.get("wheel_roi_x1", [item.wheel_roi_x1])[0] or "").strip(),
                        wheel_roi_y1=(qs.get("wheel_roi_y1", [item.wheel_roi_y1])[0] or "").strip(),
                        wheel_roi_x2=(qs.get("wheel_roi_x2", [item.wheel_roi_x2])[0] or "").strip(),
                        wheel_roi_y2=(qs.get("wheel_roi_y2", [item.wheel_roi_y2])[0] or "").strip(),
                        roi_note=(qs.get("roi_note", [item.roi_note])[0] or "").strip(),
                    )
                    app.items[item.idx] = new_item
                    save_items(app.pack_dir, app.items)
                    nav = (qs.get("nav", ["next"])[0] or "next").strip()
                    j = max(0, item.idx - 1) if nav == "back" else min(len(app.items) - 1, item.idx + 1)
                    self._redirect(f"/item/{j}")
                    return
                if path.startswith("/item/"):
                    idx = int(path[len("/item/"):] or "0")
                    it = app.get(idx)
                    body = (
                        f"<div class='meta'>"
                        f"<div><b>Index:</b> {it.idx + 1}/{len(app.items)}</div>"
                        f"<div><b>Video:</b> {html.escape(it.video_rel)}</div>"
                        f"<div><b>Frame:</b> {it.frame_idx} &nbsp; <b>t:</b> {it.timestamp_sec:.3f}s &nbsp; <b>Size:</b> {it.width}x{it.height}</div>"
                        f"<div class='row'><span class='tag gaze'>GAZE ROI</span> ({html.escape(it.gaze_roi_x1)}, {html.escape(it.gaze_roi_y1)}, {html.escape(it.gaze_roi_x2)}, {html.escape(it.gaze_roi_y2)})</div>"
                        f"<div class='row'><span class='tag wheel'>WHEEL ROI</span> ({html.escape(it.wheel_roi_x1)}, {html.escape(it.wheel_roi_y1)}, {html.escape(it.wheel_roi_x2)}, {html.escape(it.wheel_roi_y2)})</div>"
                        f"<div><b>Note:</b> {html.escape(it.roi_note or '-')}</div>"
                        f"<div class='row' style='margin-top:8px'><a class='btn' href='/item/{max(0, it.idx - 1)}'>Back</a><a class='btn' href='/item/{min(len(app.items) - 1, it.idx + 1)}'>Next</a><a class='btn' href='/file/{quote('p1_gaze_rois.manual.csv')}'>Gaze CSV</a><a class='btn' href='/file/{quote('p1_wheel_rois.manual.csv')}'>Wheel CSV</a></div>"
                        f"</div>"
                        f"<div><img src='/overlay/{it.idx}' alt='overlay'/></div>"
                        f"<div class='cols'>"
                        f"<div><img src='/img/{quote(it.ref_grid)}' alt='grid'/></div>"
                        f"<div><img src='/img/{quote(it.ref_raw)}' alt='raw'/></div>"
                        f"</div>"
                        f"<form class='form' action='/save' method='get'>"
                        f"<input type='hidden' name='idx' value='{it.idx}'/>"
                        f"<div class='cols'>"
                        f"<div>"
                        f"<div class='row'><span class='tag gaze'>GAZE ROI</span></div>"
                        f"<div class='row'>"
                        f"<label>x1 <input type='text' name='gaze_roi_x1' value='{html.escape(it.gaze_roi_x1)}'/></label>"
                        f"<label>y1 <input type='text' name='gaze_roi_y1' value='{html.escape(it.gaze_roi_y1)}'/></label>"
                        f"<label>x2 <input type='text' name='gaze_roi_x2' value='{html.escape(it.gaze_roi_x2)}'/></label>"
                        f"<label>y2 <input type='text' name='gaze_roi_y2' value='{html.escape(it.gaze_roi_y2)}'/></label>"
                        f"</div>"
                        f"</div>"
                        f"<div>"
                        f"<div class='row'><span class='tag wheel'>WHEEL ROI</span></div>"
                        f"<div class='row'>"
                        f"<label>x1 <input type='text' name='wheel_roi_x1' value='{html.escape(it.wheel_roi_x1)}'/></label>"
                        f"<label>y1 <input type='text' name='wheel_roi_y1' value='{html.escape(it.wheel_roi_y1)}'/></label>"
                        f"<label>x2 <input type='text' name='wheel_roi_x2' value='{html.escape(it.wheel_roi_x2)}'/></label>"
                        f"<label>y2 <input type='text' name='wheel_roi_y2' value='{html.escape(it.wheel_roi_y2)}'/></label>"
                        f"</div>"
                        f"</div>"
                        f"</div>"
                        f"<div style='margin-top:10px'><label>note</label><textarea name='roi_note'>{html.escape(it.roi_note)}</textarea></div>"
                        f"<div class='row' style='margin-top:10px'>"
                        f"<button class='btn' type='submit' name='nav' value='back'>Save &amp; Back</button>"
                        f"<button class='btn' type='submit' name='nav' value='next'>Save &amp; Next</button>"
                        f"</div></form>"
                    )
                    data = page("P1 Dual ROI Review", body)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
            except Exception:
                self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, traceback.format_exc())

        def log_message(self, fmt: str, *args) -> None:
            return

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.end_headers()

        def _send_text(self, status: HTTPStatus, text: str) -> None:
            b = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    return Handler


def main() -> None:
    args = parse_args()
    pack_dir = Path(args.pack_dir)
    app = App(pack_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving dual ROI review for: {pack_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
