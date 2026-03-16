# autodri

面向真实驾驶场景的驾驶员状态研究仓库。

当前主线目标：
- 注视状态模型在跨车/跨人场景下的可泛化能力
- 新域少样本快速适配（few-shot）
- `hand-on-wheel` 的时间窗口稳态判别（用于降低逐帧抖动）

给标注同学（非研发）：
- 直接看 `docs/annotation_quickstart.md`
- 不需要看训练和评估章节

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
python -m pip install -r driver_monitor/requirements.txt
```

VCS 依赖说明（GroundingDINO）：
- `driver_monitor/requirements.txt` 使用 VCS 方式安装并固定到 commit。
- 不再要求本地克隆 `GroundingDINO/` 代码目录。
- 若已安装旧版本，可强制重装：

```bash
python -m pip install --upgrade --force-reinstall -r driver_monitor/requirements.txt
```

快速确认安装来源：

```bash
python - <<'PY'
import groundingdino
from pathlib import Path
print(Path(groundingdino.__file__).resolve())
PY
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
  --face-priority right_to_left_track \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video gaze_onnx/output/gaze_demo.mp4 \
  --csv gaze_onnx/output/gaze_demo.csv
```

只跑视频中的一个 20s 片段（例如从第 120s 开始）：

```bash
python gaze_onnx/gaze_state_cls.py \
  --video /path/to/input.mp4 \
  --start-sec 120 \
  --duration-sec 20 \
  --roi 1900 660 3300 1400 \
  --face-priority right_to_left_track \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video gaze_onnx/output/gaze_demo_seg_120_20.mp4 \
  --csv gaze_onnx/output/gaze_demo_seg_120_20.csv
```

多人脸场景说明：
- `gaze_state_cls.py` 默认使用 `--face-priority right_to_left_track`，在同一 ROI 内优先选择右侧人脸，并结合跟踪稳定性，减少副驾/后排干扰。
- 如需回退旧逻辑可显式设置 `--face-priority score_track`。

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

### 3.2.1 Domain 标注流程（可直接演示）

如果你今天要给同事演示“如何做 domain 标注”，按这 4 步即可：

1. 准备 domain 清单 CSV（最小字段）  
   `domain_id,video,roi_x1,roi_y1,roi_x2,roi_y2`  
   可选字段：`n_samples`（该视频抽帧数）

示例（`gaze_onnx/experiments/manifests/demo_two_domain.csv`）：

```csv
domain_id,video,roi_x1,roi_y1,roi_x2,roi_y2,n_samples
car1_person1,data/demo/videos/car1.mp4,1900,660,3300,1400,300
car2_person2,data/demo/videos/car2.mp4,1900,660,3300,1400,300
```

2. 生成标注包（ROI 裁剪后的图片）

```bash
python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
  --domains-csv gaze_onnx/experiments/manifests/demo_two_domain.csv \
  --out-dir gaze_onnx/experiments/anno_demo_v1 \
  --samples-per-domain 300 \
  --seed 42
```

3. 打开 Web 标注页面

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_demo_v1 \
  --port 8001
```

浏览器：`http://127.0.0.1:8001`

4. 交付标注结果  
   `gaze_onnx/experiments/anno_demo_v1/labels.csv`

标注快捷键：
- `1` Forward
- `2` Non-Forward
- `3` In-Car
- `4` Other
- `0` 清除

统一规则：
- ROI 对不上、无人、非驾驶员、无法判定，一律标 `Other`。

### 3.2.2 历史数据放在哪（路径速查）

两域历史数据（car1_person1 / car2_person2）：
- domain 清单：`gaze_onnx/experiments/manifests/two_domain_videos.csv`
- 早期版本清单：`gaze_onnx/experiments/manifests/two_domain_videos.v1.csv`
- 建议版本清单：`gaze_onnx/experiments/manifests/two_domain_videos.suggested.csv`
- 标注包：
  - `gaze_onnx/experiments/anno_two_domain_v1/`
  - `gaze_onnx/experiments/anno_two_domain_v2_same_roi/`
  - `gaze_onnx/experiments/anno_two_domain_v3_ratio_roi_run1/`
- 训练集：
  - `gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3/`
  - `gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2_genv3/`
- 训练输出：`runs/classify/gaze_onnx/experiments/runs_cls/`

p1 历史数据：
- 视频与分析目录：`data/natural_driving_p1/`
- p1 窗口指标：`data/natural_driving_p1/analysis/p1_window_metrics.20s.csv`
- p1 标注包：
  - `gaze_onnx/experiments/anno_p1_gaze_small_v1/`
  - `gaze_onnx/experiments/anno_p1_gaze_200shot_v1/`
- p1 训练集：`gaze_onnx/experiments/cls_dataset_p1_200shot_driveonly_v1/`

说明：
- 如果某些目录在当前机器不存在，通常表示该数据尚未同步到本机。
- SMB 拉取流程见 `gaze_onnx/experiments/README.md` 第 4 章和 `docs/annotation_workflow.md`。

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

`hand_on_wheel.py` 现支持双后端：
- `--detector groundingdino`（基线）
- `--detector yolo --yolo-model /path/to/best.pt`（轻量加速）

为后续小模型加速导出检测框伪标签：

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --roi 1900 660 3300 1400 \
  --weights models/groundingdino_swint_ogc.pth \
  --no-video \
  --det-csv driver_monitor/output/hand_on_wheel_dets.csv
```

### 3.4 p1 自动驾驶片段批处理（20s）

当你有 `自然驾驶_视频标注情况 - p1(1).csv` 时，推荐直接跑这条链路：

1. 解析时间段并生成 20s 窗口（自动去掉每段前后 60s）
2. 生成 segment 级推理计划
3. 批量推理 gaze + wheel（建议 `--no-video`，只留 CSV）
4. 汇总窗口指标（允许部分片段尚未推理完成）

完整命令见：`gaze_onnx/experiments/README.md` 的 `4.13 p1 自动驾驶片段批处理（20s 窗口指标）`。

---

## 4. 关键文档

- 标注员快速上手：`docs/annotation_quickstart.md`
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
