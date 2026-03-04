# autodri

面向真实驾驶场景的驾驶员状态研究仓库。

当前主线目标：
- 注视状态模型在跨车/跨人场景下的可泛化能力
- 新域少样本快速适配（few-shot）
- `hand-on-wheel` 的时间窗口稳态判别（用于降低逐帧抖动）

---

## 1. 仓库模块

- `gaze_onnx/`
  - 注视状态推理（`gaze_state_cls.py`）
  - 标注、训练、跨域评估与事件级聚合（`experiments/`）
- `driver_monitor/`
  - 手是否在方向盘上（`hand_on_wheel.py`）
  - 状态 CSV 评估分析（`analyze_state_csv.py`）
- `docs/`
  - 标注流程、数据方案、poster 计划

---

## 2. 推荐环境

```bash
conda activate adri
pip install -r driver_monitor/requirements.txt
```

快速检查：

```bash
python gaze_onnx/experiments/train_gaze_cls.py --help
python gaze_onnx/experiments/web_label_tool.py --help
python gaze_onnx/gaze_state_cls.py --help
```

---

## 3. 常用入口

### 3.1 单视频注视推理（使用 ONNX）

```bash
python gaze_onnx/gaze_state_cls.py \
  --video /path/to/input.mp4 \
  --roi 1900 660 3300 1400 \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video gaze_onnx/output/gaze_demo.mp4 \
  --csv gaze_onnx/output/gaze_demo.csv
```

将帧级结果聚合为 20s 事件级：

```bash
python gaze_onnx/experiments/aggregate_gaze_windows.py \
  --csv gaze_onnx/output/gaze_demo.csv \
  --out-csv gaze_onnx/output/gaze_demo.event20s.csv \
  --window-sec 20
```

### 3.2 新参与者数据流程（ROI -> 标注 -> few-shot）

完整流程请直接看：
- `gaze_onnx/experiments/README.md`
- `docs/annotation_workflow.md`

核心脚本链路：
1. `sync_natural_driving_smb.py`（拉取 pX 视频，可选）
2. `prepare_roi_label_pack.py`（生成 ROI 参考图与待填 manifest）
3. `build_domains_csv_from_roi_manifest.py`
4. `assign_dual_roi.py`（纠正 gaze/wheel 双窗口交换）
5. `build_domains_csv_from_dual_assignment.py`
6. `create_multidomain_annotation_pack.py` + `web_label_tool.py`
7. `build_fewshot_pack.py`（如 200-shot）
8. `prepare_cls_dataset_from_pack.py` + `train_gaze_cls.py`

### 3.3 hand-on-wheel（30s 稳态）

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/hand_on_wheel_30s.mp4 \
  --weights models/groundingdino_swint_ogc.pth \
  --roi 1900 660 3300 1400 \
  --decision-window-sec 30 \
  --state-csv driver_monitor/output/hand_on_wheel_30s_states.csv
```

分析窗口平滑收益：

```bash
python driver_monitor/analyze_state_csv.py \
  --csv driver_monitor/output/hand_on_wheel_30s_states.csv \
  --sweep-windows 0,3,5,30 \
  --sweep-out-csv driver_monitor/output/window_sweep.csv
```

---

## 4. 关键文档

- 总体标注与批量流程：`docs/annotation_workflow.md`
- 泛化数据方案：`docs/generalization_datasets.md`
- Poster 思路草稿：`docs/poster_plan.md`
- 注视实验执行手册：`gaze_onnx/experiments/README.md`
- hand-on-wheel 模块说明：`driver_monitor/README.md`

---

## 5. 输出与版本管理

默认输出目录：
- `gaze_onnx/output/`
- `driver_monitor/output/`
- `gaze_onnx/experiments/anno_*`
- `gaze_onnx/experiments/cls_dataset_*`
- `runs/classify/...`

大文件与数据目录已通过 `.gitignore` 管控；提交前建议确认：

```bash
git status --short
```

---

## 6. 常见问题

- `OSError: [Errno 98] Address already in use`
  - Web 标注端口冲突，换端口（例如 `--port 8011`）。
- ONNX 推理未使用 GPU
  - 通常是 CUDA/onnxruntime-gpu 动态库缺失，推理会回退 CPU。
- 双 ROI 自动分配出现 `assignment_uncertain=1`
  - 先人工检查预览图，再决定该视频的 ROI 映射。
