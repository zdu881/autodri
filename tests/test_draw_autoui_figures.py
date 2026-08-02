from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_draw_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "draw_autoui_figures.py"
    spec = importlib.util.spec_from_file_location("draw_autoui_figures", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fewshot_budget_positions_are_categorical() -> None:
    module = _load_draw_module()
    bounds = (190, 185, 1490, 705)

    xs = [module.plot_xy(budget, 0.30, bounds)[0] for budget in [25, 50, 100, 200]]
    gaps = [round(b - a, 3) for a, b in zip(xs, xs[1:])]

    assert len(set(gaps)) == 1


def test_main_generates_all_paper_figures(tmp_path: Path, monkeypatch) -> None:
    module = _load_draw_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "FIGURES", tmp_path / "figures")

    module.main()

    assert (tmp_path / "lable_workflow.png").is_file()
    assert (tmp_path / "figures" / "evidence_checkpoints.png").is_file()
    assert (tmp_path / "figures" / "fewshot_curve.png").is_file()
    assert (tmp_path / "figures" / "distillation_schematic.png").is_file()


def test_generated_figure_text_uses_current_paper_wording() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "draw_autoui_figures.py").read_text(
        encoding="utf-8"
    )

    assert "GroundingDINO" in source
    assert "Teacher-labeled probes" in source
    assert "ON support: 300 crops" in source
    assert "ON P/R: .969 / .930" in source
    assert "0.923 clean held-out" in source
    assert "support audit" in source
    assert "27.2x speedup" in source
    assert "Window-level Behavior Metrics" in source
    assert "280/280 LOPO exports" in source
    assert "0.95 gate" not in source
    assert "95%" not in source
    assert "GroundedDINO" not in source
    assert "Behavior Distraction" not in source
    assert "p8--p10" not in source
    for text in ["p1 40/40", "p1, p2, p4", "p2+p8", "p1 probes", "p1-derived"]:
        assert text not in source


def test_wrapped_text_fit_rejects_overwide_long_words() -> None:
    module = _load_draw_module()
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (400, 200)))
    fnt = module.font(module.FONT_BOLD, 28)

    assert not module.wrapped_text_fits(draw, "GroundingDINO hand/wheel evidence", fnt, 189, 120, 5)


def test_evidence_panel_detail_stays_above_bottom_border() -> None:
    module = _load_draw_module()
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (900, 340)))
    fnt = module.font(module.FONT_OBL, module.EVIDENCE_DETAIL_B_SIZE)
    y0 = 10
    panel_bottom = y0 + module.EVIDENCE_PANEL_HEIGHT
    _, detail_bottom = module.text_size(draw, "UNCERTAIN retained", fnt)

    assert y0 + module.EVIDENCE_DETAIL_B_Y + detail_bottom + module.EVIDENCE_DETAIL_B_BOTTOM_PAD <= panel_bottom


def test_fewshot_axis_title_clears_top_tick_label() -> None:
    module = _load_draw_module()
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (400, 240)))
    axis_font = module.font(module.FONT_BOLD, 35)
    tick_font = module.font(module.FONT_REG, 31)
    bounds = module.FEWSHOT_BOUNDS
    tick_top = module.plot_xy(25, 0.75, bounds)[1] - 18
    _, axis_height = module.text_size(draw, "macro-F1", axis_font)
    _, tick_height = module.text_size(draw, "0.75", tick_font)
    axis_bottom = module.FEWSHOT_AXIS_TITLE_Y + axis_height
    tick_bottom = tick_top + tick_height

    assert axis_bottom + module.FEWSHOT_AXIS_TITLE_TICK_GAP <= tick_top or tick_bottom <= module.FEWSHOT_AXIS_TITLE_Y


def test_fewshot_callout_text_fits_inside_border(tmp_path: Path, monkeypatch) -> None:
    module = _load_draw_module()

    captured: dict[str, object] = {}
    real_center_text = module.center_text

    def capture_center_text(draw, box, text, fnt, fill=module.INK):
        if text == "ResNet gains through 100 labels; YOLO leads at 200":
            captured["box"] = box
            captured["text"] = text
            captured["font"] = fnt
        return real_center_text(draw, box, text, fnt, fill)

    monkeypatch.setattr(module, "FIGURES", tmp_path)
    monkeypatch.setattr(module, "center_text", capture_center_text)

    module.draw_fewshot_curve()

    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1650, 900)))
    x0, _, x1, _ = captured["box"]
    text_width, _ = module.text_size(draw, captured["text"], captured["font"])

    assert text_width + 40 <= x1 - x0


def test_fewshot_final_resnet_label_sits_below_marker(tmp_path: Path, monkeypatch) -> None:
    module = _load_draw_module()
    from PIL import ImageDraw

    captured: list[tuple[float, float]] = []
    real_text = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        if text == "0.662":
            captured.append(xy)
        return real_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(module, "FIGURES", tmp_path)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)

    module.draw_fewshot_curve()

    from PIL import Image, ImageDraw as PILImageDraw

    draw = PILImageDraw.Draw(Image.new("RGB", (1650, 900)))
    label_font = module.font(module.FONT_BOLD, 30)
    text_width, text_height = module.text_size(draw, "0.662", label_font)
    label_x, label_y = captured[0]
    label_box = (label_x, label_y, label_x + text_width, label_y + text_height)
    marker_x, marker_y = module.plot_xy(200, 0.662, module.FEWSHOT_BOUNDS)
    marker_radius = 21
    marker_box = (
        marker_x - marker_radius,
        marker_y - marker_radius,
        marker_x + marker_radius,
        marker_y + marker_radius,
    )
    label_center_x = label_x + text_width / 2

    assert label_box[1] >= marker_box[3] + 8
    assert abs(label_center_x - marker_x) <= 6
