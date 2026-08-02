#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Drag-to-draw dual gaze/wheel ROI review server."""

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


MANIFEST_FIELDS = [
    "participant",
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
    "gaze_review_status",
    "wheel_review_status",
    "roi_note",
]
EXPORT_FIELDS = [
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


@dataclass(frozen=True)
class Item:
    idx: int
    participant: str
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
    gaze_review_status: str
    wheel_review_status: str
    roi_note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve drag-to-draw dual ROI review pack")
    p.add_argument("--pack-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8220)
    return p.parse_args()


def int_text(value: object, default: int = 0) -> str:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return str(default)
    return str(int(round(float(raw))))


def load_items(pack_dir: Path) -> list[Item]:
    manifest = pack_dir / "roi_label_manifest.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[Item] = []
    for i, row in enumerate(rows):
        out.append(
            Item(
                idx=i,
                participant=str(row.get("participant", "") or row.get("domain_id", "")).strip(),
                video_rel=str(row.get("video_rel", "")),
                video_abs=str(row.get("video_abs", "")),
                ref_raw=str(row.get("ref_raw", "")),
                ref_grid=str(row.get("ref_grid", "")),
                frame_idx=int(float(str(row.get("frame_idx", "0") or "0"))),
                timestamp_sec=float(str(row.get("timestamp_sec", "0") or "0")),
                width=int(float(str(row.get("width", "0") or "0"))),
                height=int(float(str(row.get("height", "0") or "0"))),
                gaze_roi_x1=str(row.get("gaze_roi_x1", "")),
                gaze_roi_y1=str(row.get("gaze_roi_y1", "")),
                gaze_roi_x2=str(row.get("gaze_roi_x2", "")),
                gaze_roi_y2=str(row.get("gaze_roi_y2", "")),
                wheel_roi_x1=str(row.get("wheel_roi_x1", "")),
                wheel_roi_y1=str(row.get("wheel_roi_y1", "")),
                wheel_roi_x2=str(row.get("wheel_roi_x2", "")),
                wheel_roi_y2=str(row.get("wheel_roi_y2", "")),
                gaze_review_status=str(row.get("gaze_review_status", "pending") or "pending"),
                wheel_review_status=str(row.get("wheel_review_status", "pending") or "pending"),
                roi_note=str(row.get("roi_note", "")),
            )
        )
    return out


def item_to_row(item: Item) -> dict[str, object]:
    return {
        "participant": item.participant,
        "video_rel": item.video_rel,
        "video_abs": item.video_abs,
        "ref_raw": item.ref_raw,
        "ref_grid": item.ref_grid,
        "frame_idx": item.frame_idx,
        "timestamp_sec": f"{item.timestamp_sec:.3f}",
        "width": item.width,
        "height": item.height,
        "gaze_roi_x1": item.gaze_roi_x1,
        "gaze_roi_y1": item.gaze_roi_y1,
        "gaze_roi_x2": item.gaze_roi_x2,
        "gaze_roi_y2": item.gaze_roi_y2,
        "wheel_roi_x1": item.wheel_roi_x1,
        "wheel_roi_y1": item.wheel_roi_y1,
        "wheel_roi_x2": item.wheel_roi_x2,
        "wheel_roi_y2": item.wheel_roi_y2,
        "gaze_review_status": item.gaze_review_status,
        "wheel_review_status": item.wheel_review_status,
        "roi_note": item.roi_note,
    }


def append_status_note(note: str, roi_type: str, status: str) -> str:
    status = str(status or "").strip()
    base = str(note or "").strip()
    suffix = f"{roi_type}_review_status={status or 'pending'}"
    return f"{base}; {suffix}" if base else suffix


def export_row(item: Item, roi_type: str) -> dict[str, str]:
    prefix = "gaze" if roi_type == "gaze" else "wheel"
    return {
        "domain_id": item.participant,
        "video": item.video_abs,
        "roi_x1": getattr(item, f"{prefix}_roi_x1"),
        "roi_y1": getattr(item, f"{prefix}_roi_y1"),
        "roi_x2": getattr(item, f"{prefix}_roi_x2"),
        "roi_y2": getattr(item, f"{prefix}_roi_y2"),
        "n_samples": "1",
        "source_swapped": "",
        "source_uncertain": "1" if getattr(item, f"{prefix}_review_status") in {"pending", "uncertain"} else "0",
        "roi_note": append_status_note(item.roi_note, prefix, getattr(item, f"{prefix}_review_status")),
    }


def write_export_csvs(pack_dir: Path, items: list[Item]) -> None:
    participants = sorted({item.participant for item in items if item.participant})
    for participant in participants:
        part_items = [item for item in items if item.participant == participant]
        for roi_type in ("gaze", "wheel"):
            path = pack_dir / f"{participant}_{roi_type}_rois.manual.csv"
            tmp = pack_dir / f".{participant}_{roi_type}_rois.manual.csv.tmp"
            with tmp.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
                writer.writeheader()
                for item in part_items:
                    writer.writerow(export_row(item, roi_type))
            os.replace(tmp, path)


def save_items(pack_dir: Path, items: list[Item]) -> None:
    manifest = pack_dir / "roi_label_manifest.csv"
    tmp = pack_dir / ".roi_label_manifest.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(item_to_row(item))
    os.replace(tmp, manifest)
    write_export_csvs(pack_dir, items)


def page(title: str, body: str) -> bytes:
    doc = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
:root {
  --bg: oklch(0.973 0.006 245);
  --panel: oklch(0.995 0.004 245);
  --ink: oklch(0.225 0.018 250);
  --muted: oklch(0.51 0.03 250);
  --line: oklch(0.88 0.015 250);
  --gaze: oklch(0.72 0.18 82);
  --wheel: oklch(0.65 0.16 230);
  --danger: oklch(0.58 0.17 28);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
}
.app {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  min-height: 100vh;
}
.stage {
  padding: 14px;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 10px;
}
.topbar, .side, .canvasWrap {
  border: 1px solid var(--line);
  background: var(--panel);
}
.topbar {
  min-height: 58px;
  padding: 10px 12px;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}
.title { font-weight: 760; }
.meta { color: var(--muted); font-size: 13px; line-height: 1.45; }
.nav { display: flex; gap: 8px; flex-wrap: wrap; }
a.btn, button {
  min-height: 36px;
  padding: 0 11px;
  border: 1px solid var(--line);
  background: oklch(0.985 0.006 245);
  color: var(--ink);
  text-decoration: none;
  font: inherit;
  cursor: pointer;
}
button.primary { background: oklch(0.74 0.12 150); border-color: oklch(0.64 0.12 150); }
button.danger { color: var(--danger); }
.canvasWrap {
  min-height: 0;
  overflow: auto;
  padding: 10px;
}
#canvas {
  display: block;
  width: min(100%, 1280px);
  height: auto;
  background: oklch(0.94 0.004 245);
  cursor: crosshair;
}
.side {
  padding: 14px;
  display: grid;
  align-content: start;
  gap: 14px;
  border-top: 0;
  border-right: 0;
  border-bottom: 0;
}
.mode { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mode button.active[data-mode="gaze"] { border-color: var(--gaze); background: oklch(0.95 0.05 82); }
.mode button.active[data-mode="wheel"] { border-color: var(--wheel); background: oklch(0.95 0.04 230); }
.section {
  display: grid;
  gap: 9px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}
.section:first-child { border-top: 0; padding-top: 0; }
.sectionTitle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  font-weight: 720;
}
.chip {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 720;
}
.chip.gaze { background: oklch(0.94 0.07 82); }
.chip.wheel { background: oklch(0.93 0.05 230); }
.coords { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); font-weight: 650; }
input, textarea, select {
  width: 100%;
  border: 1px solid var(--line);
  background: oklch(0.99 0.004 245);
  color: var(--ink);
  padding: 8px;
  font: inherit;
}
textarea { min-height: 78px; resize: vertical; }
.statusButtons { display: grid; grid-template-columns: 1fr; gap: 6px; }
.statusButtons button.active { outline: 2px solid oklch(0.58 0.13 150); outline-offset: 1px; }
.hint { color: var(--muted); font-size: 12px; line-height: 1.45; }
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
@media (max-width: 960px) {
  .app { grid-template-columns: 1fr; }
  .side { border-left: 0; border-top: 1px solid var(--line); }
}
</style>
</head>
<body>__BODY__</body></html>"""
    return doc.replace("__TITLE__", html.escape(title)).replace("__BODY__", body).encode("utf-8")


class App:
    def __init__(self, pack_dir: Path):
        self.pack_dir = pack_dir.resolve()
        self.items = load_items(self.pack_dir)
        write_export_csvs(self.pack_dir, self.items)

    def get(self, idx: int) -> Item:
        if not self.items:
            raise IndexError("empty ROI review manifest")
        idx = max(0, min(len(self.items) - 1, idx))
        return self.items[idx]


def q(value: str) -> str:
    return quote(value, safe="")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            try:
                u = urlparse(self.path)
                path = u.path
                qs = parse_qs(u.query)
                if path in ("", "/"):
                    self._redirect("/item/0")
                    return
                if path.startswith("/img/"):
                    self._serve_file(unquote(path[len("/img/") :]), image=True)
                    return
                if path.startswith("/file/"):
                    self._serve_file(unquote(path[len("/file/") :]), image=False)
                    return
                if path.startswith("/save"):
                    self._save(qs)
                    return
                if path.startswith("/item/"):
                    idx = int(path[len("/item/") :] or "0")
                    self._item_page(idx)
                    return
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
            except Exception:
                self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, traceback.format_exc())

        def log_message(self, fmt: str, *args) -> None:
            return

        def _serve_file(self, rel: str, *, image: bool) -> None:
            f = (app.pack_dir / rel).resolve()
            if app.pack_dir not in f.parents and f != app.pack_dir:
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
                return
            if not f.exists() or not f.is_file():
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
                return
            data = f.read_bytes()
            if image:
                ctype = "image/jpeg"
            elif f.suffix.lower() == ".csv":
                ctype = "text/csv; charset=utf-8"
            else:
                ctype = "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _save(self, qs: dict[str, list[str]]) -> None:
            idx = int((qs.get("idx", ["0"])[0] or "0"))
            item = app.get(idx)
            new_item = Item(
                idx=item.idx,
                participant=item.participant,
                video_rel=item.video_rel,
                video_abs=item.video_abs,
                ref_raw=item.ref_raw,
                ref_grid=item.ref_grid,
                frame_idx=item.frame_idx,
                timestamp_sec=item.timestamp_sec,
                width=item.width,
                height=item.height,
                gaze_roi_x1=int_text(qs.get("gaze_roi_x1", [item.gaze_roi_x1])[0]),
                gaze_roi_y1=int_text(qs.get("gaze_roi_y1", [item.gaze_roi_y1])[0]),
                gaze_roi_x2=int_text(qs.get("gaze_roi_x2", [item.gaze_roi_x2])[0]),
                gaze_roi_y2=int_text(qs.get("gaze_roi_y2", [item.gaze_roi_y2])[0]),
                wheel_roi_x1=int_text(qs.get("wheel_roi_x1", [item.wheel_roi_x1])[0]),
                wheel_roi_y1=int_text(qs.get("wheel_roi_y1", [item.wheel_roi_y1])[0]),
                wheel_roi_x2=int_text(qs.get("wheel_roi_x2", [item.wheel_roi_x2])[0]),
                wheel_roi_y2=int_text(qs.get("wheel_roi_y2", [item.wheel_roi_y2])[0]),
                gaze_review_status=str(qs.get("gaze_review_status", [item.gaze_review_status])[0] or "pending"),
                wheel_review_status=str(qs.get("wheel_review_status", [item.wheel_review_status])[0] or "pending"),
                roi_note=str(qs.get("roi_note", [item.roi_note])[0] or "").strip(),
            )
            app.items[item.idx] = new_item
            save_items(app.pack_dir, app.items)
            nav = str(qs.get("nav", ["next"])[0] or "next")
            if nav == "stay":
                target = item.idx
            elif nav == "back":
                target = max(0, item.idx - 1)
            else:
                target = min(len(app.items) - 1, item.idx + 1)
            self._redirect(f"/item/{target}")

        def _item_page(self, idx: int) -> None:
            item = app.get(idx)
            body = item_html(item, len(app.items))
            data = page("Dual ROI Review", body)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.end_headers()

        def _send_text(self, status: HTTPStatus, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def status_buttons(prefix: str, current: str) -> str:
    labels = [
        ("original_correct", "Original correct"),
        ("corrected", "Corrected"),
        ("uncertain", "Uncertain"),
        ("pending", "Pending"),
    ]
    return "".join(
        f"<button type='button' class='statusBtn{' active' if current == value else ''}' "
        f"data-target='{prefix}_review_status' data-value='{value}'>{label}</button>"
        for value, label in labels
    )


def input_box(name: str, value: str) -> str:
    return f"<label>{html.escape(name.rsplit('_', 1)[-1])}<input id='{name}' name='{name}' value='{html.escape(value)}'/></label>"


def item_html(item: Item, total: int) -> str:
    manifest_links = "".join(
        f"<a class='btn' href='/file/{q(f'{item.participant}_{roi_type}_rois.manual.csv')}'>{roi_type.title()} CSV</a>"
        for roi_type in ("gaze", "wheel")
    )
    form = f"""
<form id="reviewForm" action="/save" method="get">
  <input type="hidden" name="idx" value="{item.idx}"/>
  <input type="hidden" id="gaze_review_status" name="gaze_review_status" value="{html.escape(item.gaze_review_status)}"/>
  <input type="hidden" id="wheel_review_status" name="wheel_review_status" value="{html.escape(item.wheel_review_status)}"/>
  <div class="section">
    <div class="mode">
      <button type="button" class="active" data-mode="gaze">Gaze</button>
      <button type="button" data-mode="wheel">Wheel</button>
    </div>
    <div class="hint">Drag on the image to replace the active ROI. Use the status buttons to mark whether the original ROI was correct.</div>
  </div>
  <div class="section">
    <div class="sectionTitle"><span class="chip gaze">Gaze</span><span id="gazeStatusText">{html.escape(item.gaze_review_status)}</span></div>
    <div class="coords">
      {input_box("gaze_roi_x1", item.gaze_roi_x1)}
      {input_box("gaze_roi_y1", item.gaze_roi_y1)}
      {input_box("gaze_roi_x2", item.gaze_roi_x2)}
      {input_box("gaze_roi_y2", item.gaze_roi_y2)}
    </div>
    <div class="statusButtons">{status_buttons("gaze", item.gaze_review_status)}</div>
  </div>
  <div class="section">
    <div class="sectionTitle"><span class="chip wheel">Wheel</span><span id="wheelStatusText">{html.escape(item.wheel_review_status)}</span></div>
    <div class="coords">
      {input_box("wheel_roi_x1", item.wheel_roi_x1)}
      {input_box("wheel_roi_y1", item.wheel_roi_y1)}
      {input_box("wheel_roi_x2", item.wheel_roi_x2)}
      {input_box("wheel_roi_y2", item.wheel_roi_y2)}
    </div>
    <div class="statusButtons">{status_buttons("wheel", item.wheel_review_status)}</div>
  </div>
  <div class="section">
    <label>Note<textarea name="roi_note">{html.escape(item.roi_note)}</textarea></label>
    <div class="actions">
      <button type="submit" name="nav" value="stay">Save</button>
      <button class="primary" type="submit" name="nav" value="next">Save Next</button>
      <button type="submit" name="nav" value="back">Save Back</button>
      <button type="button" class="danger" id="clearActive">Clear Active</button>
    </div>
  </div>
</form>"""
    data = {
        "img": f"/img/{q(item.ref_raw)}",
        "grid": f"/img/{q(item.ref_grid)}",
        "width": item.width,
        "height": item.height,
        "gaze": {
            "x1": item.gaze_roi_x1,
            "y1": item.gaze_roi_y1,
            "x2": item.gaze_roi_x2,
            "y2": item.gaze_roi_y2,
        },
        "wheel": {
            "x1": item.wheel_roi_x1,
            "y1": item.wheel_roi_y1,
            "x2": item.wheel_roi_x2,
            "y2": item.wheel_roi_y2,
        },
    }
    return f"""
<div class="app">
  <main class="stage">
    <div class="topbar">
      <div>
        <div class="title">{html.escape(item.participant)} ROI review {item.idx + 1}/{total}</div>
        <div class="meta">{html.escape(item.video_rel)}<br/>frame {item.frame_idx}, t={item.timestamp_sec:.3f}s, size {item.width}x{item.height}</div>
      </div>
      <div class="nav">
        <a class="btn" href="/item/{max(0, item.idx - 1)}">Back</a>
        <a class="btn" href="/item/{min(total - 1, item.idx + 1)}">Next</a>
        <a class="btn" href="/file/{q('roi_label_manifest.csv')}">Manifest</a>
        {manifest_links}
      </div>
    </div>
    <div class="canvasWrap">
      <canvas id="canvas" width="{item.width}" height="{item.height}"></canvas>
    </div>
  </main>
  <aside class="side">{form}</aside>
</div>
<script>
const data = {data!r};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const img = new Image();
let mode = 'gaze';
let dragging = false;
let start = null;
const rois = {{
  gaze: readRoi('gaze'),
  wheel: readRoi('wheel')
}};

function num(v) {{
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}}
function readRoi(prefix) {{
  return {{
    x1: num(document.getElementById(`${{prefix}}_roi_x1`).value),
    y1: num(document.getElementById(`${{prefix}}_roi_y1`).value),
    x2: num(document.getElementById(`${{prefix}}_roi_x2`).value),
    y2: num(document.getElementById(`${{prefix}}_roi_y2`).value)
  }};
}}
function writeRoi(prefix, roi) {{
  const x1 = Math.max(0, Math.min(data.width, Math.round(Math.min(roi.x1, roi.x2))));
  const y1 = Math.max(0, Math.min(data.height, Math.round(Math.min(roi.y1, roi.y2))));
  const x2 = Math.max(0, Math.min(data.width, Math.round(Math.max(roi.x1, roi.x2))));
  const y2 = Math.max(0, Math.min(data.height, Math.round(Math.max(roi.y1, roi.y2))));
  rois[prefix] = {{x1, y1, x2, y2}};
  for (const key of ['x1', 'y1', 'x2', 'y2']) {{
    document.getElementById(`${{prefix}}_roi_${{key}}`).value = rois[prefix][key];
  }}
  draw();
}}
function canvasPoint(ev) {{
  const rect = canvas.getBoundingClientRect();
  return {{
    x: (ev.clientX - rect.left) * canvas.width / rect.width,
    y: (ev.clientY - rect.top) * canvas.height / rect.height
  }};
}}
function drawRoi(prefix, color, label) {{
  const r = rois[prefix];
  const w = Math.max(0, r.x2 - r.x1);
  const h = Math.max(0, r.y2 - r.y1);
  if (w < 2 || h < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 8;
  ctx.strokeRect(r.x1, r.y1, w, h);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.16;
  ctx.fillRect(r.x1, r.y1, w, h);
  ctx.globalAlpha = 1;
  ctx.font = '48px ui-sans-serif, system-ui';
  ctx.fillText(label, r.x1 + 18, Math.max(58, r.y1 + 56));
  ctx.restore();
}}
function draw() {{
  if (!img.complete) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  drawRoi('gaze', 'rgb(236, 168, 20)', 'GAZE');
  drawRoi('wheel', 'rgb(38, 133, 218)', 'WHEEL');
  if (dragging && start) {{
    const r = rois[mode];
    ctx.save();
    ctx.strokeStyle = mode === 'gaze' ? 'rgb(236, 168, 20)' : 'rgb(38, 133, 218)';
    ctx.setLineDash([22, 14]);
    ctx.lineWidth = 6;
    ctx.strokeRect(Math.min(start.x, r.x2), Math.min(start.y, r.y2), Math.abs(r.x2 - start.x), Math.abs(r.y2 - start.y));
    ctx.restore();
  }}
}}
img.onload = draw;
img.src = data.img;

for (const btn of document.querySelectorAll('.mode button')) {{
  btn.addEventListener('click', () => {{
    mode = btn.dataset.mode;
    document.querySelectorAll('.mode button').forEach(x => x.classList.toggle('active', x === btn));
  }});
}}
for (const input of document.querySelectorAll('input[id$="_roi_x1"], input[id$="_roi_y1"], input[id$="_roi_x2"], input[id$="_roi_y2"]')) {{
  input.addEventListener('input', () => {{
    rois.gaze = readRoi('gaze');
    rois.wheel = readRoi('wheel');
    draw();
  }});
}}
for (const btn of document.querySelectorAll('.statusBtn')) {{
  btn.addEventListener('click', () => {{
    const target = document.getElementById(btn.dataset.target);
    target.value = btn.dataset.value;
    const prefix = btn.dataset.target.split('_')[0];
    document.getElementById(`${{prefix}}StatusText`).textContent = btn.dataset.value;
    document.querySelectorAll(`[data-target="${{btn.dataset.target}}"]`).forEach(x => x.classList.toggle('active', x === btn));
  }});
}}
canvas.addEventListener('pointerdown', ev => {{
  dragging = true;
  start = canvasPoint(ev);
  canvas.setPointerCapture(ev.pointerId);
  writeRoi(mode, {{x1: start.x, y1: start.y, x2: start.x, y2: start.y}});
}});
canvas.addEventListener('pointermove', ev => {{
  if (!dragging || !start) return;
  const p = canvasPoint(ev);
  writeRoi(mode, {{x1: start.x, y1: start.y, x2: p.x, y2: p.y}});
}});
canvas.addEventListener('pointerup', ev => {{
  if (!dragging) return;
  dragging = false;
  const p = canvasPoint(ev);
  writeRoi(mode, {{x1: start.x, y1: start.y, x2: p.x, y2: p.y}});
  start = null;
}});
document.getElementById('clearActive').addEventListener('click', () => writeRoi(mode, {{x1: 0, y1: 0, x2: 0, y2: 0}}));
</script>"""


def main() -> None:
    args = parse_args()
    app = App(Path(args.pack_dir))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving dual ROI bbox review for: {app.pack_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
