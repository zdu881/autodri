# gaze_onnx/experiments 操作手册（泛化优先）

本手册是当前项目的执行基线，目标是把流程固化为可复用的四步：
1. ROI 确定与纠错
2. 小样本标注
3. few-shot 微调
4. 帧级 + 事件级评估

适用场景：
- 一车一人视角采集（后续可扩展到更多参与者）
- 视频中可能存在 gaze/wheel 两个窗口互换
- 研究关注驾驶状态片段（优先 `Forward / Non-Forward / In-Car`）

---

## 1. 当前研究口径

- 主要任务：驾驶状态判别（3 类）
  - `Forward`
  - `Non-Forward`
  - `In-Car`
- 辅助类：`Other`
  - 包含不在驾驶位、看不到有效人脸、无法判别等情况
  - 在训练时可按任务选择是否纳入
- 评估口径：
  - 帧级指标：用于看模型瞬时能力
  - 事件级指标：20s 非重叠窗口多数投票，更贴近真实监测应用

---

## 2. 环境准备

推荐统一使用 `adri` 环境：

```bash
conda activate adri
```

最小检查：

```bash
python --version
python gaze_onnx/experiments/train_gaze_cls.py --help
python gaze_onnx/experiments/web_label_tool.py --help
```

---

## 3. 目录约定

建议按参与者组织：

- 原始视频：`data/natural_driving/pX/剪辑好的视频/`
- ROI 标注参考：`gaze_onnx/experiments/roi_refs/pX_label_pack/`
- 任务清单：`gaze_onnx/experiments/manifests/pX_*.csv`
- 标注包：`gaze_onnx/experiments/anno_pX_*`
- 训练集：`gaze_onnx/experiments/cls_dataset_pX_*`
- 训练输出：`runs/classify/gaze_onnx/experiments/runs_cls/...`
- ONNX 导出：`models/*.onnx`

历史数据速查（本项目已有）：
- 两域清单：`gaze_onnx/experiments/manifests/two_domain_videos*.csv`
- 两域标注包：`gaze_onnx/experiments/anno_two_domain_v*`
- 两域训练集：`gaze_onnx/experiments/cls_dataset_two_domain_*`
- p1 标注包：`gaze_onnx/experiments/anno_p1_gaze_*`
- p1 训练集：`gaze_onnx/experiments/cls_dataset_p1_*`
- p1 分析结果：`data/natural_driving_p1/analysis/`

说明：若目录不存在，通常是当前机器尚未同步对应数据。

---

## 4. 端到端流程（新参与者 pX）

### 4.1 同步原始视频（可选，SMB）

```bash
export SMB_PASSWORD='你的密码'

python gaze_onnx/experiments/sync_natural_driving_smb.py \
  --user nyz \
  --password-env SMB_PASSWORD \
  --participant p1 \
  --out-root data/natural_driving
```

说明：该脚本是增量同步，不会重复下载同大小文件。

### 4.2 生成 ROI 标注参考包

```bash
python gaze_onnx/experiments/prepare_roi_label_pack.py \
  --videos-root data/natural_driving/p1/剪辑好的视频 \
  --out-dir gaze_onnx/experiments/roi_refs/p1_label_pack \
  --glob '*.mp4' \
  --sample-position first \
  --grid-step 220
```

输出：
- `roi_label_manifest.csv`（待填写 ROI 坐标）
- `refs/*__grid.jpg`（网格参考图）
- `invalid_or_unreadable_videos.txt`

### 4.3 将 ROI 清单转为抽样清单

在 `roi_label_manifest.csv` 填完 `roi_x1,roi_y1,roi_x2,roi_y2` 后：

```bash
python gaze_onnx/experiments/build_domains_csv_from_roi_manifest.py \
  --roi-manifest gaze_onnx/experiments/roi_refs/p1_label_pack/roi_label_manifest.csv \
  --out-csv gaze_onnx/experiments/manifests/p1_domains.csv \
  --domain-id p1 \
  --samples-per-video 150
```

### 4.4 双窗口自动纠错（gaze/wheel 交换检测）

固定候选区域：
- ROI-A: `0,0,1900,1100`
- ROI-B: `1900,660,3300,1400`

```bash
python gaze_onnx/experiments/assign_dual_roi.py \
  --videos-csv gaze_onnx/experiments/manifests/p1_domains.csv \
  --roi-a 0 0 1900 1100 \
  --roi-b 1900 660 3300 1400 \
  --base-gaze-roi A \
  --samples 64 \
  --assignment-csv data/natural_driving_p1/p1_dual_roi_assignment.csv \
  --preview-dir gaze_onnx/experiments/output/p1_dual_roi_previews
```

关键字段：
- `gaze_roi`, `wheel_roi`：最终 ROI
- `swapped=1`：相对基准映射发生交换
- `assignment_uncertain=1`：建议人工复核

### 4.5 从纠错结果构建 gaze 任务清单

常规（gaze 用 `gaze_roi`）：

```bash
python gaze_onnx/experiments/build_domains_csv_from_dual_assignment.py \
  --assignment-csv data/natural_driving_p1/p1_dual_roi_assignment.csv \
  --task gaze \
  --domain-id p1 \
  --samples-per-video 12 \
  --samples-per-video-uncertain 20 \
  --include-uncertain \
  --require-status-ok \
  --out-csv gaze_onnx/experiments/manifests/p1_gaze_domains.csv
```

若某批次语义反向（实践中可能出现），可临时反向：

```bash
python gaze_onnx/experiments/build_domains_csv_from_dual_assignment.py \
  --assignment-csv data/natural_driving_p1/p1_dual_roi_assignment.csv \
  --task wheel \
  --domain-id p1 \
  --samples-per-video 12 \
  --samples-per-video-uncertain 20 \
  --include-uncertain \
  --require-status-ok \
  --out-csv gaze_onnx/experiments/manifests/p1_gaze_domains_inverted.csv
```

### 4.6 生成标注包（推荐 `seek`）

```bash
python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
  --domains-csv gaze_onnx/experiments/manifests/p1_gaze_domains.csv \
  --out-dir gaze_onnx/experiments/anno_p1_gaze_small_v1 \
  --seed 42 \
  --read-mode seek
```

说明：
- `seek`：稀疏抽帧更快，适合长视频
- `scan`：逐帧扫描，兼容性更保守

### 4.7 Web 标注

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_p1_gaze_small_v1 \
  --host 0.0.0.0 \
  --port 8001
```

若报 `Address already in use`，换端口：

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_p1_gaze_small_v1 \
  --port 8011
```

### 4.8 构建 200-shot 子包（可选）

```bash
python gaze_onnx/experiments/build_fewshot_pack.py \
  --src-pack gaze_onnx/experiments/anno_p1_gaze_small_v1 \
  --out-pack gaze_onnx/experiments/anno_p1_gaze_200shot_v1 \
  --num-samples 200 \
  --sample-mode by_video_uniform \
  --keep-labeled
```

### 4.9 生成训练数据集（驾驶状态三类）

```bash
python gaze_onnx/experiments/prepare_cls_dataset_from_pack.py \
  --samples-dir gaze_onnx/experiments/anno_p1_gaze_200shot_v1 \
  --out-dir gaze_onnx/experiments/cls_dataset_p1_200shot_driveonly_v1 \
  --split-mode random \
  --val-ratio 0.2 \
  --labels Forward Non-Forward In-Car \
  --augment-minority \
  --target-train-map "Forward=100,In-Car=100,Non-Forward=100" \
  --copy-mode hardlink
```

### 4.10 few-shot 微调

推荐优先基于已有泛化基模微调：

```bash
conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_p1_200shot_driveonly_v1 \
  --mode train \
  --model runs/classify/gaze_onnx/experiments/runs_cls/gaze_final_two_domain_genv2_cpu/weights/best.pt \
  --epochs 60 \
  --batch 16 \
  --device 1 \
  --aug-preset genv3 \
  --name gaze_p1_200shot_driveonly_ft_v1_gpu1
```

### 4.11 导出 ONNX

```bash
conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_p1_200shot_driveonly_v1 \
  --mode export \
  --weights runs/classify/gaze_onnx/experiments/runs_cls/gaze_p1_200shot_driveonly_ft_v1_gpu1/weights/best.pt
```

可发布为：
- `models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx`

### 4.12 单视频全量推理 + 20s 事件级聚合

```bash
conda run -n adri python gaze_onnx/gaze_state_cls.py \
  --video "data/natural_driving_p1/p1_剪辑好的视频/第三批/10.31 085825/12月13日(1).mp4" \
  --roi 1900 660 3300 1400 \
  --face-priority right_to_left_track \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video data/natural_driving_p1/infer_onevideo/p1_demo_gaze_full.mp4 \
  --csv data/natural_driving_p1/infer_onevideo/p1_demo_gaze_full.csv
```

若只研究视频中的某一段（例如从 600s 开始截取 20s）：

```bash
conda run -n adri python gaze_onnx/gaze_state_cls.py \
  --video "data/natural_driving_p1/p1_剪辑好的视频/第三批/10.31 085825/12月13日(1).mp4" \
  --start-sec 600 \
  --duration-sec 20 \
  --roi 1900 660 3300 1400 \
  --face-priority right_to_left_track \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video data/natural_driving_p1/infer_onevideo/p1_demo_gaze_seg_600_20.mp4 \
  --csv data/natural_driving_p1/infer_onevideo/p1_demo_gaze_seg_600_20.csv
```

说明：CSV 中 `Timestamp/FrameID` 为片段内相对值；`Video_Timestamp/Video_FrameID` 为原视频绝对位置。
多人脸场景默认使用 `right_to_left_track`，优先右侧人脸并结合跟踪稳定；如需回退可手工改为 `score_track`。

```bash
python gaze_onnx/experiments/aggregate_gaze_windows.py \
  --csv data/natural_driving_p1/infer_onevideo/p1_demo_gaze_full.csv \
  --out-csv data/natural_driving_p1/infer_onevideo/p1_demo_gaze_full.event20s.csv \
  --window-sec 20
```

### 4.13 p1 自动驾驶片段批处理（20s 窗口指标）

目标：从 `自然驾驶_视频标注情况 - p1(1).csv` 自动生成“去头尾 60s 后的 20s 窗口”，批量跑 gaze + wheel 推理，并提取窗口指标。

规则（已确认）：
- 窗口长度固定 `20s`
- `off-path = Non-Forward + In-Car`
- 每段 AD 区间先裁掉前后各 `60s`
- 小部分 ROI 对不上/无人片段保留为缺失状态，不强行计算

1) 从排班 CSV 解析区间并生成 20s 窗口：

```bash
python gaze_onnx/experiments/build_p1_schedule_windows.py \
  --schedule-csv "自然驾驶_视频标注情况 - p1(1).csv" \
  --videos-root data/natural_driving_p1/p1_剪辑好的视频 \
  --window-sec 20 \
  --trim-sec 60 \
  --segments-out data/natural_driving_p1/analysis/p1_segments.parsed.csv \
  --windows-out data/natural_driving_p1/analysis/p1_windows.20s.csv
```

2) 生成按 `segment_uid` 的推理计划（推荐）：

```bash
python gaze_onnx/experiments/build_p1_infer_plan.py \
  --windows-csv data/natural_driving_p1/analysis/p1_windows.20s.csv \
  --group-by segment \
  --out-dir data/natural_driving_p1/infer_p1_windows/segment_mode \
  --plan-csv data/natural_driving_p1/analysis/p1_infer_plan.segment.csv \
  --gaze-map-csv data/natural_driving_p1/analysis/p1_gaze_map.segment.csv \
  --wheel-map-csv data/natural_driving_p1/analysis/p1_wheel_map.segment.csv
```

3) 批量推理（只产出 CSV，不写 mp4，避免磁盘占用）：

```bash
/data/home/sim6g/anaconda3/envs/adri/bin/python gaze_onnx/experiments/run_p1_infer_plan.py \
  --plan-csv data/natural_driving_p1/analysis/p1_infer_plan.segment.csv \
  --run-gaze --run-wheel \
  --skip-existing \
  --no-video \
  --python-bin /data/home/sim6g/anaconda3/envs/adri/bin/python \
  --wheel-device cuda
```

若要同步导出 wheel 检测框（用于 YOLO 训练伪标签）：

```bash
/data/home/sim6g/anaconda3/envs/adri/bin/python gaze_onnx/experiments/run_p1_infer_plan.py \
  --plan-csv data/natural_driving_p1/analysis/p1_infer_plan.segment.csv \
  --run-wheel \
  --skip-existing \
  --no-video \
  --python-bin /data/home/sim6g/anaconda3/envs/adri/bin/python \
  --wheel-device cuda \
  --wheel-det-csv-dir data/natural_driving_p1/analysis/wheel_det_csv
```

4) 计算 20s 窗口指标（允许部分片段尚未推理完成）：

```bash
python gaze_onnx/experiments/compute_p1_window_metrics.py \
  --windows-csv data/natural_driving_p1/analysis/p1_windows.20s.csv \
  --gaze-map-csv data/natural_driving_p1/analysis/p1_gaze_map.segment.csv \
  --wheel-map-csv data/natural_driving_p1/analysis/p1_wheel_map.segment.csv \
  --out-csv data/natural_driving_p1/analysis/p1_window_metrics.20s.csv
```

输出 `status` 说明：
- `ok`：该 20s 窗口指标已成功计算
- `missing_csv_map`：窗口未找到映射 CSV（计划/映射文件问题）
- `missing_gaze_file` / `missing_wheel_file` / `missing_gaze_wheel_file`：该段推理文件尚未生成
- `csv_load_error`：CSV 存在但字段不符合预期，需要检查该段输出格式

---

## 5. 跨域统一评估（帧级 + 事件级）

```bash
python gaze_onnx/experiments/cross_domain_eval.py \
  --eval-item "car1_to_car2|gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2_genv3|runs/classify/gaze_onnx/experiments/runs_cls/gaze_holdout_car2_genv3_cpu3/weights/best.pt" \
  --eval-item "car2_to_car1|gaze_onnx/experiments/cls_dataset_two_domain_holdout_car1_genv3|runs/classify/gaze_onnx/experiments/runs_cls/gaze_holdout_car1_genv3_cpu/weights/best.pt" \
  --event-window-sec 20 \
  --out-csv gaze_onnx/experiments/runs_cls/cross_domain_eval_genv3.csv
```

建议投稿时至少报告：
- 帧级：`accuracy` + `macro-F1`
- 事件级：窗口多数投票后的 `accuracy` + `macro-F1`
- 类别级召回：特别关注 `Non-Forward` 与 `In-Car`

---

## 6. Hand-on-wheel 项目流程（30s 稳定状态）

手放方向盘项目建议采用“帧级判定 + 窗口多数投票”的同构思路，减少状态抖动。
依赖安装采用 VCS：

```bash
python -m pip install -r driver_monitor/requirements.txt
```

当前 `driver_monitor/hand_on_wheel.py` 支持两种检测后端：
- `--detector groundingdino`（默认基线）
- `--detector yolo --yolo-model /path/to/best.pt`（轻量加速）

基础推理：

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/hand_on_wheel.mp4 \
  --weights models/groundingdino_swint_ogc.pth \
  --roi 1900 660 3300 1400 \
  --state-csv driver_monitor/output/hand_on_wheel_states.csv
```

30s 窗口稳定输出（推荐用于报告）：

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/hand_on_wheel_30s.mp4 \
  --weights models/groundingdino_swint_ogc.pth \
  --roi 1900 660 3300 1400 \
  --decision-window-sec 30 \
  --state-csv driver_monitor/output/hand_on_wheel_30s_states.csv
```

结果分析：

```bash
python driver_monitor/analyze_state_csv.py \
  --csv driver_monitor/output/hand_on_wheel_30s_states.csv \
  --sweep-windows 0,3,5,30 \
  --sweep-out-csv driver_monitor/output/window_sweep.csv
```

为后续轻量 YOLO 加速准备伪标签（导出检测框）：

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --roi 1900 660 3300 1400 \
  --weights models/groundingdino_swint_ogc.pth \
  --no-video \
  --state-csv driver_monitor/output/hand_on_wheel_states.csv \
  --det-csv driver_monitor/output/hand_on_wheel_dets.csv
```

将 det-csv 转为 YOLO 检测数据集：

```bash
python driver_monitor/build_wheel_yolo_dataset.py \
  --det-csv "driver_monitor/output/*.dets.csv" \
  --out-dir driver_monitor/output/wheel_yolo_ds \
  --use-roi-crop \
  --include-negatives \
  --neg-keep-prob 0.2
```

---

## 7. 标注规范（必须统一）

- `Forward`：主要注视前方道路
- `Non-Forward`：明显偏离前方（侧看、低头等）
- `In-Car`：看车内区域（中控/仪表/车内后视镜等）
- `Other`：不在驾驶位、无有效人脸、无法判别、非目标驾驶员、ROI 对不上导致的无人区域
- `Unknown`：仅用于标注中间态，不建议作为训练类

驾驶状态三类训练时，通常过滤 `Other/Unknown`。

---

## 8. 常见问题

### 8.1 Web 标注端口占用

报错：`OSError: [Errno 98] Address already in use`

处理：
1. 换端口（例如 `--port 8011`）
2. 或结束占用进程后复用原端口

### 8.2 ONNXRuntime GPU 回退 CPU

如果缺少 CUDA 动态库，推理会自动回退 CPU。训练可继续使用 PyTorch GPU，不影响训练本身。

### 8.3 双 ROI 自动分配低置信

出现 `assignment_uncertain=1` 时，不要直接批量跑全流程，先看 `preview_dir` 里的叠框图，人工确认后再继续。

---

## 9. 已落地脚本索引

- ROI 与抽样：
  - `prepare_roi_label_pack.py`
  - `build_domains_csv_from_roi_manifest.py`
  - `create_multidomain_annotation_pack.py`
- 双窗口纠错：
  - `assign_dual_roi.py`
  - `build_domains_csv_from_dual_assignment.py`
- 标注与 few-shot：
  - `web_label_tool.py`
  - `build_fewshot_pack.py`
- 训练与导出：
  - `prepare_cls_dataset_from_pack.py`
  - `train_gaze_cls.py`
- 评估：
  - `aggregate_gaze_windows.py`
  - `cross_domain_eval.py`
