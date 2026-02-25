# Driver Monitoring + Gaze ONNX

本仓库包含两个独立模块，用于驾驶员状态监测与注视方向估计：

- `driver_monitor`：基于 MediaPipe / GroundingDINO 的驾驶员监测（注视跟踪、方向盘握持等）
- `gaze_onnx`：基于 ONNX 的注视方向估计与分档统计

所有生成物（视频、CSV、调试图、快照）会写入各自模块的 `output/` 或 `snaps*` 目录，已在 `.gitignore` 中排除。

---

## 目录结构

```
.
├─ driver_monitor/
│  ├─ README.md              # 模块说明（注视 + 手握方向盘）
│  ├─ gaze_tracking.py        # MediaPipe 眼球追踪与视线方向
│  ├─ hand_on_wheel.py         # GroundingDINO 检测手与方向盘
│  ├─ check_mp.py              # MediaPipe 环境检查
│  └─ output/                  # 运行输出（忽略提交）
├─ gaze_onnx/
│  ├─ gaze_state_onnx.py       # ONNX 注视方向估计与统计
│  ├─ download_models.py       # 模型下载辅助脚本（可选）
│  ├─ README.md                # 模块说明
│  └─ output/                  # 运行输出（忽略提交）
└─ README.md
```

---

## 环境依赖

建议使用 Python 3.8+。

基础依赖：

- `opencv-python`
- `numpy`
- `mediapipe`
- `onnxruntime`
- `torch`（仅 `hand_on_wheel.py` 使用）

如需运行 GroundingDINO 相关脚本，需要可用的 GroundingDINO 代码与权重（本仓库未包含）。

---

## 快速开始

### 1) MediaPipe 注视方向（driver_monitor）

```
python driver_monitor/gaze_tracking.py \
	--video /path/to/input.mp4 \
	--output driver_monitor/output/gaze_output.mp4
```

输出视频会叠加注视方向与视线向量。

### 2) 方向盘握持检测（driver_monitor）

```
python driver_monitor/hand_on_wheel.py \
	--video /path/to/input.mp4 \
	--output driver_monitor/output/hand_on_wheel.mp4 \
	--config /path/to/GroundingDINO_SwinT_OGC.py \
	--weights /path/to/groundingdino_swint_ogc.pth
```

运行后会生成辅助 ROI 参考图 `roi_helper.jpg` 与 ROI 预览图，按提示输入 ROI 坐标即可。

### 3) ONNX 注视方向估计（gaze_onnx）

```
python gaze_onnx/gaze_state_onnx.py \
	--video /path/to/input.mp4 \
	--scrfd /path/to/scrfd_person_2.5g.onnx \
	--l2cs /path/to/L2CSNet_gaze360.onnx \
	--roi 950 300 1650 690 \
	--out-video gaze_onnx/output/output_gaze.mp4 \
	--csv gaze_onnx/output/output_gaze.csv
```

更多参数与说明请查看 [gaze_onnx/README.md](gaze_onnx/README.md)。

---

## 输出说明

- `driver_monitor/output/`：注视与握持检测的视频结果
- `gaze_onnx/output/`：注视估计的视频、CSV、调试图、统计文件
- `docs/generalization_datasets.md`：跨人/跨车泛化训练与数据整合计划
- `docs/annotation_workflow.md`：从零开始的两车两人标注流程（含 ROI 标注与工具命令）

---

## 常见问题

- 若 MediaPipe 报错，可先运行 `check_mp.py` 进行环境检测。
- 若 ONNX 推理速度较慢，请优先使用 GPU 版 `onnxruntime-gpu`。
- GroundingDINO 权重与配置文件需自行准备并在命令行中指定。
