import argparse
import csv
import os
from collections import Counter, deque
from typing import List, Dict, Any, Deque, Tuple

import cv2
import numpy as np
import torch

from groundingdino.util.inference import Model
from groundingdino.util.box_ops import box_iou


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


def compute_iou(hand_boxes: np.ndarray, wheel_boxes: np.ndarray) -> float:
    if hand_boxes.size == 0 or wheel_boxes.size == 0:
        return 0.0
    hand_tensor = torch.tensor(hand_boxes, dtype=torch.float32)
    wheel_tensor = torch.tensor(wheel_boxes, dtype=torch.float32)
    iou, _ = box_iou(hand_tensor, wheel_tensor)
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
    parser = argparse.ArgumentParser(description="Detect hand on steering wheel with GroundingDINO")
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
        "--artifacts-dir",
        default="driver_monitor/output",
        help="Directory for helper images (ROI guides/previews)",
    )
    parser.add_argument(
        "--config",
        default="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Path to model config",
    )
    parser.add_argument(
        "--weights",
        default="../GroundingDINO/weights/groundingdino_swint_ogc.pth",
        help="Path to model weights",
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

    if args.sample_fps <= 0:
        stride = 1
    else:
        stride = max(1, int(round(fps / args.sample_fps)))

    iou_on_threshold = float(args.iou_on_threshold) if args.iou_on_threshold is not None else float(args.iou_threshold)
    iou_off_threshold = float(args.iou_off_threshold) if args.iou_off_threshold is not None else float(args.iou_threshold)
    if iou_on_threshold < iou_off_threshold:
        raise ValueError("--iou-on-threshold should be >= --iou-off-threshold")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    model = Model(
        model_config_path=args.config,
        model_checkpoint_path=args.weights,
        device=args.device,
    )

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time_sec = frame_idx / fps
        if args.max_seconds > 0 and current_time_sec >= args.max_seconds:
            break

        if frame_idx % stride == 0:
            # Crop image for prediction
            frame_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Skip empty crops (just in case)
            if frame_crop.size == 0:
                detections = None
            else:
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
            hand_boxes = np.array([d["box"] for d in filtered if d["class_id"] == 0], dtype=np.float32)
            wheel_boxes = np.array([d["box"] for d in filtered if d["class_id"] == 1], dtype=np.float32)
            hand_confs = [d["confidence"] for d in filtered if d["class_id"] == 0]
            wheel_confs = [d["confidence"] for d in filtered if d["class_id"] == 1]

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

        writer.write(frame)

        if state_csv_writer is not None:
            raw_num = STATE_TO_NUM[raw_state]
            stable_num = STATE_TO_NUM[stable_state]
            state_csv_writer.writerow(
                [
                    frame_idx,
                    f"{current_time_sec:.6f}",
                    timestamp,
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
    writer.release()
    if state_csv_file is not None:
        state_csv_file.close()

    print("Done.")
    print(f"Input frames: {total_frames}")
    print(f"Output video: {args.output}")
    if args.state_csv:
        print(f"State CSV: {args.state_csv}")


if __name__ == "__main__":
    main()
