#!/usr/bin/env python3
"""Draw compact supplemental figures for paper/autoui.tex.

The figures intentionally use deterministic jittered strokes so the PDF gets a
hand-drawn visual aid without depending on SVG text rendering.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_OBL = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"

INK = (33, 43, 54, 255)
MUTED = (82, 92, 104, 255)
PAPER = (249, 247, 239, 255)
GRID = (218, 214, 202, 150)
BLUE = (41, 93, 153, 255)
ORANGE = (181, 92, 42, 255)
GREEN = (63, 125, 89, 255)
RED = (165, 65, 62, 255)
PURPLE = (105, 82, 146, 255)
TEAL = (45, 125, 132, 255)
YELLOW = (199, 147, 45, 255)
STEP_BLUE = (219, 235, 247, 255)
STEP_YELLOW = (251, 242, 214, 255)
STEP_GREEN = (226, 240, 221, 255)
STEP_PURPLE = (237, 229, 245, 255)
BUDGETS = [25, 50, 100, 200]
EVIDENCE_PANEL_HEIGHT = 284
EVIDENCE_DETAIL_A_Y = 198
EVIDENCE_DETAIL_A_SIZE = 31
EVIDENCE_DETAIL_B_Y = 242
EVIDENCE_DETAIL_B_SIZE = 24
EVIDENCE_DETAIL_B_BOTTOM_PAD = 16
FEWSHOT_BOUNDS = (235, 230, 1490, 650)
FEWSHOT_AXIS_TITLE_Y = 176
FEWSHOT_AXIS_TITLE_TICK_GAP = 10
FEWSHOT_CALLOUT_TEXT = "ResNet gains through 100 labels; YOLO leads at 200"
FEWSHOT_CALLOUT_BOX = (430, 752, 1245, 808)
FEWSHOT_RESNET_LABEL_OFFSETS = {
    200: (-48, 31),
}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_paper_grid(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    for x in range(44, w, 92):
        draw.line([(x, 0), (x, h)], fill=GRID, width=1)
    for y in range(38, h, 92):
        draw.line([(0, y), (w, y)], fill=GRID, width=1)


def jittered_points(
    a: tuple[float, float],
    b: tuple[float, float],
    rng: random.Random,
    segments: int = 16,
    jitter: float = 3.0,
) -> list[tuple[float, float]]:
    points = []
    for i in range(segments + 1):
        t = i / segments
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if i not in (0, segments):
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        points.append((x, y))
    return points


def rough_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
    rng: random.Random,
    passes: int = 2,
    jitter: float = 3.0,
) -> None:
    for _ in range(passes):
        draw.line(jittered_points(a, b, rng, jitter=jitter), fill=color, width=width, joint="curve")


def rough_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    width: int,
    rng: random.Random,
    fill: tuple[int, int, int, int] | None = None,
) -> None:
    x0, y0, x1, y1 = xy
    if fill is not None:
        draw.rounded_rectangle(xy, radius=18, fill=fill)
    rough_line(draw, (x0, y0), (x1, y0), outline, width, rng)
    rough_line(draw, (x1, y0), (x1, y1), outline, width, rng)
    rough_line(draw, (x1, y1), (x0, y1), outline, width, rng)
    rough_line(draw, (x0, y1), (x0, y0), outline, width, rng)


def arrow(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
    rng: random.Random,
) -> None:
    rough_line(draw, a, b, color, width, rng, jitter=2.0)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    length = 24
    spread = 0.55
    head_left = (b[0] - length * math.cos(ang - spread), b[1] - length * math.sin(ang - spread))
    head_right = (b[0] - length * math.cos(ang + spread), b[1] - length * math.sin(ang + spread))
    rough_line(draw, b, head_left, color, width, rng, jitter=1.5)
    rough_line(draw, b, head_right, color, width, rng, jitter=1.5)


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int] = INK,
) -> None:
    tw, th = text_size(draw, text, fnt)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=fnt, fill=fill)


def fit_center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font_path: str,
    max_size: int,
    min_size: int,
    fill: tuple[int, int, int, int] = INK,
) -> None:
    x0, y0, x1, y1 = box
    for size in range(max_size, min_size - 1, -1):
        fnt = font(font_path, size)
        tw, th = text_size(draw, text, fnt)
        if tw <= x1 - x0 and th <= y1 - y0:
            center_text(draw, box, text, fnt, fill)
            return
    center_text(draw, box, text, font(font_path, min_size), fill)


def center_multiline_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int] = INK,
    line_gap: int = 5,
) -> None:
    x0, y0, x1, y1 = box
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, line_h in zip(lines, heights):
        line_w, _ = text_size(draw, line, fnt)
        draw.text((x0 + (x1 - x0 - line_w) / 2, y), line, font=fnt, fill=fill)
        y += line_h + line_gap


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_size(draw, candidate, fnt)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    max_width: int,
    line_gap: int = 7,
) -> int:
    x, y = xy
    cursor = y
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, cursor), line, font=fnt, fill=fill)
        cursor += text_size(draw, line or "A", fnt)[1] + line_gap
    return cursor


def wrapped_text_height(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    max_width: int,
    line_gap: int,
) -> int:
    lines = wrap_text(draw, text, fnt, max_width)
    if not lines:
        return 0
    line_heights = [text_size(draw, line or "A", fnt)[1] for line in lines]
    return sum(line_heights) + line_gap * (len(lines) - 1)


def wrapped_text_fits(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    max_width: int,
    max_height: int,
    line_gap: int,
) -> bool:
    lines = wrap_text(draw, text, fnt, max_width)
    if any(text_size(draw, line, fnt)[0] > max_width for line in lines):
        return False
    return wrapped_text_height(draw, text, fnt, max_width, line_gap) <= max_height


def draw_step_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    x0, y0, x1, y1 = xy
    rough_rect(draw, xy, INK, 4, rng, fill=(255, 253, 247, 250))
    draw.rounded_rectangle((x0 + 18, y0 + 18, x0 + 62, y0 + 62), radius=13, fill=accent)
    center_text(draw, (x0 + 18, y0 + 18, x0 + 62, y0 + 62), title, font(FONT_BOLD, 27), PAPER)
    max_width = x1 - x0 - 118
    max_height = y1 - y0 - 28
    for size in range(28, 19, -1):
        fnt = font(FONT_BOLD, size)
        if wrapped_text_fits(draw, body, fnt, max_width, max_height, 5):
            break
    draw_wrapped_text(draw, (x0 + 92, y0 + 18), body, fnt, INK, max_width, line_gap=5)


def draw_column_header(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    fill: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=20, fill=fill)
    rough_rect(draw, xy, INK, 3, rng)
    fit_center_text(draw, (x0 + 18, y0 + 10, x1 - 18, y1 - 10), title, FONT_BOLD, 28, 20, PAPER)


def draw_label_workflow() -> None:
    rng = random.Random(20260526)
    w, h = 1650, 1040
    im = Image.new("RGBA", (w, h), PAPER)
    draw = ImageDraw.Draw(im)
    draw_paper_grid(draw, w, h)

    draw.text((72, 42), "Human-in-the-loop workflow", font=font(FONT_BOLD, 56), fill=INK)
    draw.text((76, 112), "reviewable checkpoints from ROI to behavior windows", font=font(FONT_OBL, 32), fill=MUTED)

    margin = 58
    gap = 28
    top = 185
    bottom = 860
    col_w = (w - 2 * margin - 3 * gap) // 4
    fills = [STEP_BLUE, STEP_YELLOW, STEP_GREEN, STEP_PURPLE]
    headers = [
        "1  Inputs",
        "2  AOI Models",
        "3  Inference",
        "4  Windows",
    ]
    columns: list[tuple[int, int, int, int]] = []
    for i, header in enumerate(headers):
        x0 = margin + i * (col_w + gap)
        x1 = x0 + col_w
        columns.append((x0, top, x1, bottom))
        draw.rounded_rectangle((x0, top, x1, bottom), radius=24, fill=fills[i])
        rough_rect(draw, (x0, top, x1, bottom), INK, 3, rng)
        draw_column_header(draw, (x0 + 18, bottom + 24, x1 - 18, h - 46), header, (10, 63, 113, 255), rng)

    boxes = [
        [
            ("A", "Study spreadsheet + videos"),
            ("B", "Driver ROI review"),
            ("C", "Sample extraction"),
            ("D", "Four AOI labels"),
        ],
        [
            ("E", "AOI calibration"),
            ("F", "Backbone comparison"),
            ("G", "ONNX parity checks"),
        ],
        [
            ("H", "Segment-level gaze inference"),
            ("I", "GroundingDINO hand/wheel evidence"),
            ("J", "Temporal state rules"),
        ],
        [
            ("K", "Gaze + hand-state alignment"),
            ("L", "Window-level Behavior Metrics"),
            ("M", "Participant summaries"),
        ],
    ]
    accents = [BLUE, GREEN, TEAL, ORANGE]
    box_positions: list[list[tuple[int, int, int, int]]] = []
    for col, col_boxes in zip(columns, boxes):
        x0, _, x1, _ = col
        count = len(col_boxes)
        box_h = 112 if count == 4 else 130
        y = top + 42
        step = 142 if count == 4 else 176
        positions = []
        for idx, (label, body) in enumerate(col_boxes):
            xy = (x0 + 28, y + idx * step, x1 - 28, y + idx * step + box_h)
            draw_step_box(draw, xy, label, body, accents[len(box_positions)], rng)
            positions.append(xy)
        box_positions.append(positions)

    for positions in box_positions:
        for a, b in zip(positions, positions[1:]):
            arrow(draw, ((a[0] + a[2]) / 2, a[3] + 4), ((b[0] + b[2]) / 2, b[1] - 4), INK, 4, rng)
    for left, right in zip(box_positions, box_positions[1:]):
        a = left[-1]
        b = right[0]
        arrow(draw, (a[2] + 6, (a[1] + a[3]) / 2), (b[0] - 8, (b[1] + b[3]) / 2), INK, 4, rng)

    im = im.convert("RGB")
    im.save(FIGURES / "label_workflow.png", dpi=(220, 220), quality=95)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    idx: str,
    title: str,
    metric: str,
    detail_a: str,
    detail_b: str,
    accent: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    x0, y0, x1, y1 = xy
    fill = (255, 253, 246, 245)
    rough_rect(draw, xy, INK, 5, rng, fill=fill)
    draw.ellipse((x0 + 26, y0 + 24, x0 + 88, y0 + 86), fill=(255, 251, 235, 255), outline=accent, width=5)
    center_text(draw, (x0 + 26, y0 + 24, x0 + 88, y0 + 86), idx, font(FONT_BOLD, 34), accent)

    draw.text((x0 + 112, y0 + 28), title, font=font(FONT_BOLD, 42), fill=INK)
    draw.text((x0 + 42, y0 + 116), metric, font=font(FONT_BOLD, 54), fill=accent)
    draw.text((x0 + 46, y0 + EVIDENCE_DETAIL_A_Y), detail_a, font=font(FONT_BOLD, EVIDENCE_DETAIL_A_SIZE), fill=INK)
    draw.text((x0 + 46, y0 + EVIDENCE_DETAIL_B_Y), detail_b, font=font(FONT_OBL, EVIDENCE_DETAIL_B_SIZE), fill=MUTED)


def draw_evidence_checkpoints() -> None:
    rng = random.Random(240524)
    w, h = 1650, 1160
    im = Image.new("RGBA", (w, h), PAPER)
    draw = ImageDraw.Draw(im)
    draw_paper_grid(draw, w, h)

    draw.text((72, 45), "Evidence checkpoints", font=font(FONT_BOLD, 54), fill=INK)
    draw.text((76, 112), "artifact trail for the current WIP claim", font=font(FONT_OBL, 34), fill=MUTED)

    panels = [
        ("1", "ROI validation", "15 participants", "24/24 high-risk reviewed", "40/40 consistency check", BLUE),
        ("2", "AOI semantics", "280 LOPO jobs", "11/14 non-YOLO NI", "five seeds per split", GREEN),
        ("3", "Few-shot", "25-200 labels", "ResNet NI in 8/12", "three no-leak panels", ORANGE),
        ("4", "Temporal rules", "-84.91% switches", "279.82 to 42.22/min", "60 two-context segments", RED),
        ("5", "Hand-state review", "18/24 + 21/21", "two review passes", "UNCERTAIN retained", PURPLE),
        ("6", "Deployment", "280/280 LOPO exports", "top-1 parity >= .99", "ONNX check passed", TEAL),
    ]

    margin_x = 70
    top = 190
    gap_x = 46
    gap_y = 34
    pw = (w - 2 * margin_x - gap_x) // 2
    ph = EVIDENCE_PANEL_HEIGHT
    for i, panel in enumerate(panels):
        col = i % 2
        row = i // 2
        x0 = margin_x + col * (pw + gap_x)
        y0 = top + row * (ph + gap_y)
        draw_panel(draw, (x0, y0, x0 + pw, y0 + ph), *panel, rng)

    im = im.convert("RGB")
    im.save(FIGURES / "evidence_checkpoints.png", dpi=(220, 220), quality=95)


def plot_xy(x: float, y: float, bounds: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bounds
    y_min, y_max = 0.30, 0.75
    if x not in BUDGETS:
        raise ValueError(f"Unsupported label budget for categorical axis: {x}")
    tx = BUDGETS.index(int(x)) / (len(BUDGETS) - 1)
    ty = (y - y_min) / (y_max - y_min)
    return x0 + tx * (x1 - x0), y1 - ty * (y1 - y0)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    p: tuple[float, float],
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    x, y = p
    r = 13
    draw.ellipse((x - r, y - r, x + r, y + r), fill=PAPER, outline=color, width=6)
    for _ in range(2):
        rough_line(draw, (x - r - 4, y), (x + r + 4, y), color, 2, rng, jitter=1.2)


def draw_series(
    draw: ImageDraw.ImageDraw,
    data: list[tuple[int, float]],
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    pts = [plot_xy(x, y, bounds) for x, y in data]
    for a, b in zip(pts, pts[1:]):
        rough_line(draw, a, b, color, 7, rng, passes=2, jitter=3.5)
    for p in pts:
        draw_marker(draw, p, color, rng)


def draw_fewshot_curve() -> None:
    rng = random.Random(20260524)
    w, h = 1650, 900
    im = Image.new("RGBA", (w, h), PAPER)
    draw = ImageDraw.Draw(im)
    draw_paper_grid(draw, w, h)

    draw.text((72, 46), "Few-shot participant adaptation", font=font(FONT_BOLD, 54), fill=INK)
    draw.text((76, 113), "macro-F1 over three no-leak participant-specific test sets", font=font(FONT_OBL, 34), fill=MUTED)

    bounds = FEWSHOT_BOUNDS
    x0, y0, x1, y1 = bounds
    rough_line(draw, (x0, y1), (x1, y1), INK, 5, rng)
    rough_line(draw, (x0, y0), (x0, y1), INK, 5, rng)

    for y in [0.35, 0.45, 0.55, 0.65, 0.75]:
        py = plot_xy(25, y, bounds)[1]
        draw.line((x0 - 16, py, x1 + 4, py), fill=(190, 187, 176, 140), width=2)
        draw.text((72, py - 18), f"{y:.2f}", font=font(FONT_REG, 31), fill=MUTED)

    for x in BUDGETS:
        px = plot_xy(x, 0.30, bounds)[0]
        draw.line((px, y1 - 8, px, y1 + 16), fill=INK, width=4)
        center_text(draw, (int(px - 56), y1 + 34, int(px + 56), y1 + 78), str(x), font(FONT_BOLD, 34), INK)

    center_text(draw, (560, 818, 1165, 866), "participant-specific labels", font(FONT_BOLD, 35), INK)
    draw.text((82, FEWSHOT_AXIS_TITLE_Y), "macro-F1", font=font(FONT_BOLD, 35), fill=INK)

    resnet = [(25, 0.441), (50, 0.517), (100, 0.644), (200, 0.662)]
    yolo = [(25, 0.326), (50, 0.425), (100, 0.633), (200, 0.712)]
    draw_series(draw, resnet, bounds, BLUE, rng)
    draw_series(draw, yolo, bounds, ORANGE, rng)

    label_font = font(FONT_BOLD, 30)
    for x, y in resnet:
        px, py = plot_xy(x, y, bounds)
        dx, dy = FEWSHOT_RESNET_LABEL_OFFSETS.get(x, (-38, -54))
        draw.text((px + dx, py + dy), f"{y:.3f}", font=label_font, fill=BLUE)
    for x, y in yolo:
        px, py = plot_xy(x, y, bounds)
        offset = 32 if x < 200 else -55
        draw.text((px - 38, py + offset), f"{y:.3f}", font=label_font, fill=ORANGE)

    legend_x, legend_y = 1125, 64
    rough_line(draw, (legend_x, legend_y + 18), (legend_x + 82, legend_y + 18), BLUE, 7, rng, jitter=2)
    draw.text((legend_x + 102, legend_y - 3), "ResNet50", font=font(FONT_BOLD, 34), fill=INK)
    rough_line(draw, (legend_x, legend_y + 70), (legend_x + 82, legend_y + 70), ORANGE, 7, rng, jitter=2)
    draw.text((legend_x + 102, legend_y + 49), "YOLOv8s-cls", font=font(FONT_BOLD, 34), fill=INK)

    callout_box = FEWSHOT_CALLOUT_BOX
    draw.rounded_rectangle(callout_box, radius=18, fill=(255, 253, 246, 235))
    rough_rect(draw, callout_box, MUTED, 2, rng)
    center_text(
        draw,
        callout_box,
        FEWSHOT_CALLOUT_TEXT,
        font(FONT_OBL, 28),
        MUTED,
    )

    im = im.convert("RGB")
    im.save(FIGURES / "fewshot_curve.png", dpi=(220, 220), quality=95)


def draw_distillation_schematic() -> None:
    rng = random.Random(20260604)
    w, h = 1650, 930
    im = Image.new("RGBA", (w, h), PAPER)
    draw = ImageDraw.Draw(im)
    draw_paper_grid(draw, w, h)

    draw.text((72, 46), "Hand-state distillation check", font=font(FONT_BOLD, 54), fill=INK)
    draw.text(
        (76, 113),
        "GroundingDINO teacher labels compact YOLOv8s-cls student states",
        font=font(FONT_OBL, 31),
        fill=MUTED,
    )

    stage_y0, stage_y1 = 214, 610
    stage_w = 300
    gap = 34
    left = 70
    stages = [
        ("1", "Teacher-labeled probes", "2 initial probes\n60 s + 140 s\nGroundingDINO boxes", BLUE),
        ("2", "Stable state labels", "ON / OFF / UNCERTAIN\nteacher pipeline states\n5,000 ROI crops", ORANGE),
        ("3", "Student training", "YOLOv8s-cls\nbalanced clean crops\ntime-block validation", GREEN),
        ("4", "Held-out checks", "ON support: 300 crops\nON P/R: .969 / .930\n27.2x speedup", TEAL),
    ]

    boxes: list[tuple[int, int, int, int]] = []
    for idx, (num, title, detail, accent) in enumerate(stages):
        x0 = left + idx * (stage_w + gap)
        x1 = x0 + stage_w
        boxes.append((x0, stage_y0, x1, stage_y1))
        rough_rect(draw, (x0, stage_y0, x1, stage_y1), INK, 4, rng, fill=(255, 253, 246, 245))
        draw.ellipse((x0 + 24, stage_y0 + 26, x0 + 86, stage_y0 + 88), fill=(255, 251, 235, 255), outline=accent, width=5)
        center_text(draw, (x0 + 24, stage_y0 + 26, x0 + 86, stage_y0 + 88), num, font(FONT_BOLD, 34), accent)
        draw_wrapped_text(draw, (x0 + 102, stage_y0 + 30), title, font(FONT_BOLD, 31), INK, x1 - x0 - 126, 5)
        draw_wrapped_text(draw, (x0 + 36, stage_y0 + 142), detail, font(FONT_BOLD, 27), MUTED, x1 - x0 - 72, 8)

    for a, b in zip(boxes, boxes[1:]):
        arrow(draw, (a[2] + 10, (a[1] + a[3]) / 2), (b[0] - 10, (b[1] + b[3]) / 2), INK, 4, rng)

    metric_boxes = [
        ((190, 682, 515, 825), "Agreement", ["0.987 initial", "0.923 clean held-out"], GREEN),
        ((662, 682, 987, 825), "Runtime", ["6.342 s / 5,000 crops", "1.268 ms per crop"], TEAL),
        ((1134, 682, 1459, 825), "Boundary", ["support audit", "teacher-state check", "not replacement"], RED),
    ]
    for xy, title, detail_lines, accent in metric_boxes:
        x0, y0, x1, y1 = xy
        draw.rounded_rectangle(xy, radius=18, fill=(255, 253, 246, 235))
        rough_rect(draw, xy, accent, 3, rng)
        center_text(draw, (x0 + 18, y0 + 12, x1 - 18, y0 + 48), title, font(FONT_BOLD, 28), INK)
        center_multiline_text(draw, (x0 + 18, y0 + 54, x1 - 18, y1 - 16), detail_lines, font(FONT_OBL, 22), MUTED, 6)

    center_text(
        draw,
        (240, 855, 1410, 900),
        "Use teacher cost once on selected clips; test student only as an acceleration checkpoint",
        font(FONT_OBL, 27),
        MUTED,
    )

    im = im.convert("RGB")
    im.save(FIGURES / "distillation_schematic.png", dpi=(220, 220), quality=95)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_label_workflow()
    draw_evidence_checkpoints()
    draw_fewshot_curve()
    draw_distillation_schematic()


if __name__ == "__main__":
    main()
