import argparse
import csv
import os
from collections import Counter, deque
from importlib.util import find_spec
from pathlib import Path
from typing import List, Dict, Any, Deque, Tuple, Optional

import cv2
import numpy as np
import torch

from autodri.common.paths import resolve_existing_path, workspace_root

CLASSES = ["hand", "steering wheel"] # For display/logic
# Expanded prompts for better recall
PROMPTS_HAND = ["hand", "arm", "wrist", "fingers"]
PROMPTS_WHEEL = ["steering wheel", "wheel", "driving wheel"]
# Combined list for the model
PROMPT_LIST = PROMPTS_HAND + PROMPTS_WHEEL
# Map prompt index to class index (0 for hand, 1 for wheel)
PROMPT_TO_CLASS = [0] * len(PROMPTS_HAND) + [1] * len(PROMPTS_WHEEL)

COLOR_MAP = {
    0: (0, 255, 0),      # hand - green
    1: (255, 0, 0),      # steering wheel - blue
}

STATE_ON = "ON"
STATE_OFF = "OFF"
STATE_UNCERTAIN = "UNCERTAIN"
STATE_TO_NUM = {
    STATE_ON: 1,
    STATE_OFF: 0,
    STATE_UNCERTAIN: -1,
}


def resolve_groundingdino_config(config_arg: str) -> str:
    """Resolve GroundingDINO config path from arg or installed package."""
    if config_arg:
        return str(resolve_existing_path(config_arg, description="GroundingDINO config"))

    spec = find_spec("groundingdino")
    if spec and spec.submodule_search_locations:
        pkg_root = Path(list(spec.submodule_search_locations)[0])
        cfg = pkg_root / "config" / "GroundingDINO_SwinT_OGC.py"
        if cfg.exists():
            return str(cfg)

    return str(
        resolve_existing_path(
            "",
            workspace_rel="sources/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            legacy_rels=("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",),
            description="GroundingDINO config",
        )
    )


def resolve_groundingdino_weights(weights_arg: str) -> str:
    """Resolve GroundingDINO weights path from arg or common local locations."""
    if weights_arg:
        return str(resolve_existing_path(weights_arg, description="GroundingDINO weights"))

    return str(
        resolve_existing_path(
            "",
            workspace_rel="models/groundingdino_swint_ogc.pth",
            legacy_rels=(
                "models/groundingdino_swint_ogc.pth",
                "GroundingDINO/weights/groundingdino_swint_ogc.pth",
            ),
            description="GroundingDINO weights",
        )
    )


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_id = det["class_id"]
        conf = det["confidence"]
        color = COLOR_MAP.get(class_id, (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[class_id]} {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def build_detections(detections) -> List[Dict[str, Any]]:
    if detections is None or detections.xyxy is None or len(detections.xyxy) == 0:
        return []

    class_ids = detections.class_id
    valid_mask = np.array([c is not None for c in class_ids])

    if valid_mask.sum() == 0:
        return []

    boxes = detections.xyxy[valid_mask].astype(int)
    confs = detections.confidence[valid_mask]
    class_ids = class_ids[valid_mask].astype(int)

    results = []
    for box, conf, class_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = box.tolist()
        results.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(class_id),
            }
        )
    return results


def parse_class_id_set(spec: str) -> set:
    s = (spec or "").strip()
    if not s:
        return set()
    out = set()
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        out.add(int(t))
    return out


def build_detections_from_yolo(
    frame_crop: np.ndarray,
    model: Any,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    hand_class_ids: set,
    wheel_class_ids: set,
    roi_x1: int,
    roi_y1: int,
) -> List[Dict[str, Any]]:
    if frame_crop.size == 0:
        return []
    preds = model.predict(
        source=frame_crop,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        device=device,
        verbose=False,
    )
    if not preds:
        return []
    p0 = preds[0]
    if p0.boxes is None or len(p0.boxes) == 0:
        return []
    xyxy = p0.boxes.xyxy.detach().cpu().numpy()
    confs = p0.boxes.conf.detach().cpu().numpy()
    clss = p0.boxes.cls.detach().cpu().numpy().astype(int)
    out: List[Dict[str, Any]] = []
    for box, score, raw_cid in zip(xyxy, confs, clss):
        if raw_cid in hand_class_ids:
            cid = 0
        elif raw_cid in wheel_class_ids:
            cid = 1
        else:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
        out.append(
            {
                "box": [x1 + int(roi_x1), y1 + int(roi_y1), x2 + int(roi_x1), y2 + int(roi_y1)],
                "confidence": float(score),
                "class_id": int(cid),
            }
        )
    return out


def compute_iou(hand_boxes: np.ndarray, wheel_boxes: np.ndarray) -> float:
    if hand_boxes.size == 0 or wheel_boxes.size == 0:
        return 0.0
    hand_tensor = torch.tensor(hand_boxes, dtype=torch.float32)
    wheel_tensor = torch.tensor(wheel_boxes, dtype=torch.float32)
    # NxM IoU matrix, implemented locally to avoid hard import dependency.
    lt = torch.max(hand_tensor[:, None, :2], wheel_tensor[None, :, :2])
    rb = torch.min(hand_tensor[:, None, 2:], wheel_tensor[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    hand_area = (hand_tensor[:, 2] - hand_tensor[:, 0]).clamp(min=0) * (
        hand_tensor[:, 3] - hand_tensor[:, 1]
    ).clamp(min=0)
    wheel_area = (wheel_tensor[:, 2] - wheel_tensor[:, 0]).clamp(min=0) * (
        wheel_tensor[:, 3] - wheel_tensor[:, 1]
    ).clamp(min=0)
    union = hand_area[:, None] + wheel_area[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return float(iou.max().item()) if iou.numel() else 0.0


def vote_in_window(
    window: Deque[Tuple[float, str]], previous_state: str
) -> Tuple[str, int, int, int]:
    """Return majority state in current window; tie falls back to previous state."""
    if not window:
        return previous_state, 0, 0, 0

    counts = Counter(state for _, state in window)
    on_count = int(counts.get(STATE_ON, 0))
    off_count = int(counts.get(STATE_OFF, 0))
    uncertain_count = int(counts.get(STATE_UNCERTAIN, 0))
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    if len(candidates) == 1:
        return candidates[0], on_count, off_count, uncertain_count
    if previous_state in candidates:
        return previous_state, on_count, off_count, uncertain_count
    # Deterministic fallback for ties without previous-state hit.
    for s in (STATE_ON, STATE_OFF, STATE_UNCERTAIN):
        if s in candidates:
            return s, on_count, off_count, uncertain_count
    return previous_state, on_count, off_count, uncertain_count


def update_raw_state(
    previous_state: str,
    iou_max: float,
    max_hand_conf: float,
    max_wheel_conf: float,
    now_sec: float,
    last_reliable_sec: float,
    iou_on_threshold: float,
    iou_off_threshold: float,
    min_hand_conf: float,
    min_wheel_conf: float,
    uncertain_grace_sec: float,
    enable_uncertain: bool,
) -> Tuple[str, float]:
    """Hysteresis + uncertainty handling."""
    reliable = (max_hand_conf >= min_hand_conf) and (max_wheel_conf >= min_wheel_conf)
    if reliable:
        last_reliable_sec = now_sec
        if previous_state == STATE_ON:
            if iou_max <= iou_off_threshold:
                return STATE_OFF, last_reliable_sec
            return STATE_ON, last_reliable_sec
        if previous_state == STATE_OFF:
            if iou_max >= iou_on_threshold:
                return STATE_ON, last_reliable_sec
            return STATE_OFF, last_reliable_sec
        # previous uncertain
        if iou_max >= iou_on_threshold:
            return STATE_ON, last_reliable_sec
        if iou_max <= iou_off_threshold:
            return STATE_OFF, last_reliable_sec
        return STATE_UNCERTAIN, last_reliable_sec

    if (now_sec - last_reliable_sec) <= uncertain_grace_sec:
        return previous_state, last_reliable_sec
    if enable_uncertain:
        return STATE_UNCERTAIN, last_reliable_sec
    return previous_state, last_reliable_sec


def select_roi_interactive(cap: cv2.VideoCapture, artifacts_dir: str) -> tuple:
    # Read one frame for reference
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video pointer
    if not ret:
        print("Error: Could not read video frame.")
        return None

    h, w = frame.shape[:2]
    
    # 1. Create helper image with grid
    grid_img = frame.copy()
    step = 200  # Grid spacing
    
    # Vertical lines
    for x in range(0, w, step):
        cv2.line(grid_img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(grid_img, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Horizontal lines
    for y in range(0, h, step):
        cv2.line(grid_img, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(grid_img, str(y), (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save helper
    helper_path = os.path.join(artifacts_dir, "roi_helper.jpg")
    cv2.imwrite(helper_path, grid_img)
    
    print("\n" + "="*60)
    print(f"【步骤 1/3】辅助参考图已保存至: {os.path.abspath(helper_path)}")
    print("请在 VS Code 中打开该图片，确定左上角区域的像素范围。")
    print(f"图像原始尺寸: {w} x {h}")
    print("="*60)

    while True:
        # 2. Get user input
        print("\n请输入 ROI 区域坐标 (格式: x_min y_min x_max y_max)")
        print("例如左上角 800x600 区域请输入: 0 0 800 600")
        print("输入 'all' 或直接回车以使用全图。")
        user_input = input("ROI > ").strip()
        
        if not user_input or user_input.lower() == 'all':
            return 0, 0, w, h
            
        try:
            coords = list(map(int, user_input.split()))
            if len(coords) != 4:
                print("错误: 请输入 4 个整数。")
                continue
            
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                print("错误: 区域无效 (宽/高必须大于0)。")
                continue

            # 3. Generate preview
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
            preview_path = os.path.join(artifacts_dir, "roi_preview.jpg")
            cv2.imwrite(preview_path, preview)
            
            print(f"【步骤 2/3】预览图已保存至: {os.path.abspath(preview_path)}")
            confirm = input("确认使用此区域? (y/n): ").lower()
            if confirm == 'y':
                print(f"【步骤 3/3】ROI 已确认: {x1}, {y1}, {x2}, {y2}")
                return x1, y1, x2, y2
        except ValueError:
            print("错误: 输入包含非数字字符。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect hand on steering wheel with GroundingDINO or YOLO")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default="driver_monitor/output/hand_on_wheel.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable mp4 writing and only save state CSV.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time in seconds for segment inference (default: 0, from video start).",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Segment duration in seconds (default: 0, process to video end).",
    )
    parser.add_argument(
        "--detector",
        choices=["groundingdino", "yolo"],
        default="groundingdino",
        help="Detection backend for hand/wheel boxes.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="driver_monitor/output",
        help="Directory for helper images (ROI guides/previews)",
    )
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Path to GroundingDINO config. "
            "If omitted, auto-resolve from installed groundingdino package."
        ),
    )
    parser.add_argument(
        "--weights",
        default="",
        help=(
            "Path to GroundingDINO weights. "
            "If omitted, auto-search models/groundingdino_swint_ogc.pth."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        default="",
        help="YOLO model path (required when --detector yolo).",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument(
        "--yolo-hand-class-ids",
        default="0",
        help="Comma-separated YOLO class ids mapped to 'hand'.",
    )
    parser.add_argument(
        "--yolo-wheel-class-ids",
        default="1",
        help="Comma-separated YOLO class ids mapped to 'steering wheel'.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    parser.add_argument(
        "--iou-on-threshold",
        type=float,
        default=None,
        help="State-machine ON threshold (defaults to --iou-threshold).",
    )
    parser.add_argument(
        "--iou-off-threshold",
        type=float,
        default=None,
        help="State-machine OFF threshold (defaults to --iou-threshold).",
    )
    parser.add_argument(
        "--min-hand-conf",
        type=float,
        default=0.20,
        help="Minimum hand confidence to treat frame as reliable.",
    )
    parser.add_argument(
        "--min-wheel-conf",
        type=float,
        default=0.20,
        help="Minimum wheel confidence to treat frame as reliable.",
    )
    parser.add_argument(
        "--uncertain-grace-sec",
        type=float,
        default=0.8,
        help="How long to keep previous state when detections are unreliable.",
    )
    parser.add_argument(
        "--disable-uncertain",
        action="store_true",
        help="Disable UNCERTAIN state and keep previous state when unreliable.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Run detection at this FPS and reuse results for skipped frames",
    )
    parser.add_argument(
        "--decision-window-sec",
        type=float,
        default=0.0,
        help=(
            "Temporal majority-vote window in seconds. "
            "0 disables smoothing and keeps per-frame decision."
        ),
    )
    parser.add_argument(
        "--state-csv",
        default="",
        help=(
            "Optional path to save per-frame raw/stable decisions for analysis "
            "(useful for poster metrics)."
        ),
    )
    parser.add_argument(
        "--det-csv",
        default="",
        help=(
            "Optional path to save sampled-frame detections (boxes/classes/conf) "
            "for downstream YOLO dataset building."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop processing after this many seconds (0 means full video).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Interactively select ROI before processing",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help="Manually set ROI coordinates (x1 y1 x2 y2). Overrides --select-roi.",
    )

    args = parser.parse_args()
    if args.output == "driver_monitor/output/hand_on_wheel.mp4":
        args.output = str(workspace_root(create=True) / "artifacts" / "driver_monitor" / "hand_on_wheel.mp4")
    else:
        args.output = str(Path(args.output).expanduser())
    if args.artifacts_dir == "driver_monitor/output":
        args.artifacts_dir = str(workspace_root(create=True) / "artifacts" / "driver_monitor")
    else:
        args.artifacts_dir = str(Path(args.artifacts_dir).expanduser())

    if not args.no_video:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    
    # Handle ROI selection
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if args.roi:
        roi_x1, roi_y1, roi_x2, roi_y2 = args.roi
        print(f"Using manual ROI: {roi_x1}, {roi_y1}, {roi_x2}, {roi_y2}")
    elif args.select_roi:
        roi = select_roi_interactive(cap, artifacts_dir=args.artifacts_dir)
        if roi:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_sec = max(0.0, float(args.start_sec))
    duration_sec = max(0.0, float(args.duration_sec))
    start_frame = int(round(start_sec * fps))
    if total_frames > 0 and start_frame >= total_frames:
        raise ValueError(
            f"--start-sec ({start_sec:.3f}s) exceeds video duration "
            f"({total_frames / max(1e-6, fps):.3f}s)"
        )
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    max_proc_frames = 0
    if duration_sec > 0:
        max_proc_frames = max(1, int(round(duration_sec * fps)))
    if total_frames > 0:
        remain_from_start = max(0, total_frames - start_frame)
        if max_proc_frames > 0:
            max_proc_frames = min(max_proc_frames, remain_from_start)
        else:
            max_proc_frames = remain_from_start

    if args.sample_fps <= 0:
        stride = 1
    else:
        stride = max(1, int(round(fps / args.sample_fps)))

    iou_on_threshold = float(args.iou_on_threshold) if args.iou_on_threshold is not None else float(args.iou_threshold)
    iou_off_threshold = float(args.iou_off_threshold) if args.iou_off_threshold is not None else float(args.iou_threshold)
    if iou_on_threshold < iou_off_threshold:
        raise ValueError("--iou-on-threshold should be >= --iou-off-threshold")

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {args.output}")

    detector = str(args.detector).strip().lower()
    if detector == "groundingdino":
        resolved_config = resolve_groundingdino_config(args.config)
        resolved_weights = resolve_groundingdino_weights(args.weights)
        print(f"Using detector: groundingdino")
        print(f"Using GroundingDINO config: {resolved_config}")
        print(f"Using GroundingDINO weights: {resolved_weights}")
        try:
            from groundingdino.util.inference import Model
        except ImportError as exc:
            raise ImportError(
                "groundingdino is not installed. Run: pip install -r driver_monitor/requirements.txt"
            ) from exc

        model = Model(
            model_config_path=resolved_config,
            model_checkpoint_path=resolved_weights,
            device=args.device,
        )
        yolo_model = None
        yolo_hand_class_ids = set()
        yolo_wheel_class_ids = set()
    elif detector == "yolo":
        yolo_path = str(args.yolo_model or "").strip()
        if not yolo_path:
            raise ValueError("--yolo-model is required when --detector yolo")
        if not Path(yolo_path).exists():
            raise FileNotFoundError(yolo_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Install it in the current environment first."
            ) from exc
        yolo_hand_class_ids = parse_class_id_set(args.yolo_hand_class_ids)
        yolo_wheel_class_ids = parse_class_id_set(args.yolo_wheel_class_ids)
        if not yolo_hand_class_ids or not yolo_wheel_class_ids:
            raise ValueError("YOLO hand/wheel class-id sets must be non-empty")
        yolo_model = YOLO(yolo_path)
        model = None
        print(f"Using detector: yolo")
        print(f"Using YOLO model: {yolo_path}")
        print(
            f"YOLO class mapping: hand={sorted(yolo_hand_class_ids)} "
            f"wheel={sorted(yolo_wheel_class_ids)}"
        )
    else:
        raise ValueError(f"Unsupported --detector: {args.detector}")

    frame_idx = 0
    last_detections: List[Dict[str, Any]] = []
    last_iou = 0.0
    max_hand_conf = 0.0
    max_wheel_conf = 0.0
    raw_state = STATE_OFF
    stable_state = STATE_OFF
    last_reliable_sec = -1e9
    window_sec = max(0.0, float(args.decision_window_sec))
    decision_window: Deque[Tuple[float, str]] = deque()

    state_csv_file = None
    state_csv_writer = None
    if args.state_csv:
        os.makedirs(os.path.dirname(args.state_csv) or ".", exist_ok=True)
        state_csv_file = open(args.state_csv, "w", newline="", encoding="utf-8")
        state_csv_writer = csv.writer(state_csv_file)
        state_csv_writer.writerow(
            [
                "frame",
                "time_sec",
                "time_text",
                "video_time_sec",
                "video_frame",
                "raw_state",
                "stable_state",
                "raw_hand_on_wheel",
                "stable_hand_on_wheel",
                "iou_max",
                "max_hand_conf",
                "max_wheel_conf",
                "window_sec",
                "window_samples",
                "window_on_count",
                "window_off_count",
                "window_uncertain_count",
            ]
        )

    det_csv_file = None
    det_csv_writer = None
    if args.det_csv:
        os.makedirs(os.path.dirname(args.det_csv) or ".", exist_ok=True)
        det_csv_file = open(args.det_csv, "w", newline="", encoding="utf-8")
        det_csv_writer = csv.writer(det_csv_file)
        det_csv_writer.writerow(
            [
                "video_path",
                "frame",
                "video_frame",
                "time_sec",
                "video_time_sec",
                "roi_x1",
                "roi_y1",
                "roi_x2",
                "roi_y2",
                "sampled",
                "num_dets",
                "det_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

    while True:
        if max_proc_frames > 0 and frame_idx >= max_proc_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        current_time_sec = frame_idx / fps
        video_time_sec = start_sec + current_time_sec
        video_frame_idx = start_frame + frame_idx
        if args.max_seconds > 0 and current_time_sec >= args.max_seconds:
            break

        if frame_idx % stride == 0:
            # Crop image for prediction
            frame_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            filtered: List[Dict[str, Any]] = []
            # Skip empty crops (just in case)
            if frame_crop.size > 0:
                if detector == "groundingdino":
                    detections = model.predict_with_classes(
                        image=frame_crop,
                        classes=PROMPT_LIST,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                    )

                    # Map raw prompt indices to logical class indices
                    if detections.class_id is not None:
                        new_class_ids = []
                        for cid in detections.class_id:
                            if cid is not None and 0 <= cid < len(PROMPT_TO_CLASS):
                                new_class_ids.append(PROMPT_TO_CLASS[cid])
                            else:
                                new_class_ids.append(None)
                        detections.class_id = np.array(new_class_ids)

                    # Adjust boxes coordinates: Add ROI offset
                    if detections.xyxy is not None and len(detections.xyxy) > 0:
                        detections.xyxy += np.array([roi_x1, roi_y1, roi_x1, roi_y1])
                    filtered = build_detections(detections) if detections else []
                else:
                    filtered = build_detections_from_yolo(
                        frame_crop=frame_crop,
                        model=yolo_model,
                        device=args.device,
                        imgsz=int(args.yolo_imgsz),
                        conf=float(args.yolo_conf),
                        iou=float(args.yolo_iou),
                        hand_class_ids=yolo_hand_class_ids,
                        wheel_class_ids=yolo_wheel_class_ids,
                        roi_x1=int(roi_x1),
                        roi_y1=int(roi_y1),
                    )
            hand_boxes = np.array([d["box"] for d in filtered if d["class_id"] == 0], dtype=np.float32)
            wheel_boxes = np.array([d["box"] for d in filtered if d["class_id"] == 1], dtype=np.float32)
            hand_confs = [d["confidence"] for d in filtered if d["class_id"] == 0]
            wheel_confs = [d["confidence"] for d in filtered if d["class_id"] == 1]

            if det_csv_writer is not None:
                num_dets = len(filtered)
                if num_dets <= 0:
                    det_csv_writer.writerow(
                        [
                            args.video,
                            frame_idx,
                            video_frame_idx,
                            f"{current_time_sec:.6f}",
                            f"{video_time_sec:.6f}",
                            roi_x1,
                            roi_y1,
                            roi_x2,
                            roi_y2,
                            1,
                            0,
                            -1,
                            -1,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                else:
                    for di, det in enumerate(filtered):
                        x1, y1, x2, y2 = [int(v) for v in det["box"]]
                        cid = int(det["class_id"])
                        cname = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"class_{cid}"
                        det_csv_writer.writerow(
                            [
                                args.video,
                                frame_idx,
                                video_frame_idx,
                                f"{current_time_sec:.6f}",
                                f"{video_time_sec:.6f}",
                                roi_x1,
                                roi_y1,
                                roi_x2,
                                roi_y2,
                                1,
                                num_dets,
                                di,
                                cid,
                                cname,
                                f"{float(det['confidence']):.6f}",
                                x1,
                                y1,
                                x2,
                                y2,
                            ]
                        )

            iou_max = compute_iou(hand_boxes, wheel_boxes)
            last_detections = filtered
            last_iou = iou_max
            max_hand_conf = max(hand_confs) if hand_confs else 0.0
            max_wheel_conf = max(wheel_confs) if wheel_confs else 0.0

            raw_state, last_reliable_sec = update_raw_state(
                previous_state=raw_state,
                iou_max=iou_max,
                max_hand_conf=max_hand_conf,
                max_wheel_conf=max_wheel_conf,
                now_sec=current_time_sec,
                last_reliable_sec=last_reliable_sec,
                iou_on_threshold=iou_on_threshold,
                iou_off_threshold=iou_off_threshold,
                min_hand_conf=float(args.min_hand_conf),
                min_wheel_conf=float(args.min_wheel_conf),
                uncertain_grace_sec=float(args.uncertain_grace_sec),
                enable_uncertain=(not args.disable_uncertain),
            )

        if window_sec > 0:
            decision_window.append((current_time_sec, raw_state))
            cutoff = current_time_sec - window_sec
            while decision_window and decision_window[0][0] < cutoff:
                decision_window.popleft()
            stable_state, on_count, off_count, uncertain_count = vote_in_window(
                decision_window, stable_state
            )
        else:
            stable_state = raw_state
            on_count = int(raw_state == STATE_ON)
            off_count = int(raw_state == STATE_OFF)
            uncertain_count = int(raw_state == STATE_UNCERTAIN)

        # Draw ROI boundary for specific visualization
        if args.select_roi:
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 1)

        draw_detections(frame, last_detections)

        timestamp = format_timestamp(current_time_sec)
        video_timestamp = format_timestamp(video_time_sec)
        if stable_state == STATE_ON:
            status_text = "HAND ON WHEEL"
            status_color = (0, 255, 0)
        elif stable_state == STATE_OFF:
            status_text = "HAND OFF WHEEL"
            status_color = (0, 0, 255)
        else:
            status_text = "STATE UNCERTAIN"
            status_color = (0, 255, 255)

        cv2.putText(
            frame,
            f"t={timestamp}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"video_t={video_timestamp}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{status_text}  IoU={last_iou:.3f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )
        if window_sec > 0:
            cv2.putText(
                frame,
                (
                    f"RAW={raw_state}  WIN={window_sec:.1f}s  "
                    f"VOTE(on/off/unc)={on_count}/{off_count}/{uncertain_count}"
                ),
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            (
                f"thr_on/off={iou_on_threshold:.3f}/{iou_off_threshold:.3f} "
                f"conf(h/w)={max_hand_conf:.2f}/{max_wheel_conf:.2f}"
            ),
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(frame)

        if state_csv_writer is not None:
            raw_num = STATE_TO_NUM[raw_state]
            stable_num = STATE_TO_NUM[stable_state]
            state_csv_writer.writerow(
                [
                    frame_idx,
                    f"{current_time_sec:.6f}",
                    timestamp,
                    f"{video_time_sec:.6f}",
                    video_frame_idx,
                    raw_state,
                    stable_state,
                    raw_num,
                    stable_num,
                    f"{last_iou:.6f}",
                    f"{max_hand_conf:.6f}",
                    f"{max_wheel_conf:.6f}",
                    f"{window_sec:.3f}",
                    len(decision_window) if window_sec > 0 else 1,
                    on_count,
                    off_count,
                    uncertain_count,
                ]
            )

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if state_csv_file is not None:
        state_csv_file.close()
    if det_csv_file is not None:
        det_csv_file.close()

    print("Done.")
    print(f"Input frames (video total): {total_frames}")
    print(f"Segment start_sec={start_sec:.3f} duration_sec={duration_sec:.3f}")
    print(f"Processed frames: {frame_idx}")
    if writer is not None:
        print(f"Output video: {args.output}")
    else:
        print("Output video: <disabled by --no-video>")
    if args.state_csv:
        print(f"State CSV: {args.state_csv}")
    if args.det_csv:
        print(f"Det CSV: {args.det_csv}")


if __name__ == "__main__":
    main()
