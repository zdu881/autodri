# gaze_onnx 实验：判断是模型精度问题还是划分策略问题

你现在遇到的典型现象（中段大量 In-Car、开头看中控却 Forward）可能来自两类根因：

- **模型输出问题**：L2CS 产出的 pitch/yaw 本身不可靠（域偏移、裁剪/光照、姿态极端等），导致无论怎么改阈值都难救。
- **策略/校准问题**：pitch/yaw 还行，但 **ref 校准/漂移**、阈值、去抖、规则形状不合理，导致可通过策略显著提升。

这里提供一个最小闭环：

1) **从全量结果中“分层抽样”导出少量帧**（时间段×预测类别×边界样本）
2) **人工三分类标注**（Forward / Non-Forward / In-Car）
3) **自动评估 + 阈值消融**：看只改阈值能提升多少（策略空间），以及错误是否集中在边界附近（更像策略）

---

## 1) CSV 时序诊断（不需要人工标注）

先看 ref 是否漂移、以及 In-Car 触发条件是否“饱和”。

```bash
python gaze_onnx/experiments/analyze_csv.py \
  --pred-csv gaze_onnx/output/output_gaze_smooth4_full.csv \
  --window-sec 10
```

重点看每个时间窗的：
- `refP/refY` 是否慢慢漂移
- `dp50/dp90` 是否整体偏移
- `trig(InCar)` 是否接近 100%（这通常意味着阈值或 ref 出问题）

---

## 2) 抽样导出待标注帧

建议从**你最不满意的那份 full 输出**开始（比如 `output_gaze_smooth4_full.*`）。

```bash
python gaze_onnx/experiments/sample_frames.py \
  --video gaze_onnx/output/output_gaze_smooth4_full.mp4 \
  --pred-csv gaze_onnx/output/output_gaze_smooth4_full.csv \
  --out-dir gaze_onnx/experiments/samples_smooth4_full \
  --n-total 360 --time-bins 6 --boundary-frac 0.35
```

产物：
- `manifest.csv`：样本清单
- 一堆 `*.png`：带基本信息面板的帧

---

## 3) 人工标注

```bash
python gaze_onnx/experiments/label_tool.py \
  --samples-dir gaze_onnx/experiments/samples_smooth4_full
```

按键：
- `1`/`f` = Forward
- `2`/`n` = Non-Forward
- `3`/`i` = In-Car
- `0`/`u` = Unknown（评估时会跳过）
- `b` = 回退一条
- `q` = 退出

会生成/更新：
- `labels.csv`

建议标注量：
- 先做 200~400 帧（通常就能判断方向）

---

## 4) 评估 + 阈值消融（关键）

```bash
python gaze_onnx/experiments/eval_labels.py \
  --pred-csv gaze_onnx/output/output_gaze_smooth4_full.csv \
  --labels gaze_onnx/experiments/samples_smooth4_full/labels.csv
```

你会看到：
- 当前策略的 accuracy / confusion matrix
- 每类 precision/recall
- **threshold grid search** 的最佳 accuracy
- “strategy headroom（只改阈值能提升多少）”

如何解读：
- 如果 **headroom 很大（例如 +10%~+30%）**：说明模型角度信息可用，主要是策略/阈值/去抖形状问题。
- 如果 **headroom 很小（例如 <+3%~+5%）**：说明在现有 delta 上“阈值已救不了”，更像模型输出本身或 ref 校准路径有系统性偏差（需要改 ref 门控/改 crop/改模型）。

---

## 推荐你先跑哪两步？

1) `analyze_csv.py`（立刻能告诉你 ref/触发是否异常）
2) 抽样 360 帧 + 标注 200 帧，然后跑 `eval_labels.py`

如果你愿意，我也可以基于你标注出来的 `labels.csv`：
- 自动给出一套“更稳”的阈值建议
- 以及指出最离谱的错例帧（方便你快速看是模型瞎了还是规则瞎了）

---

## 新增：从零标注两域数据

当没有现成预测 CSV 时，使用以下脚本：

1) 生成 ROI 参考图（带网格坐标）：

```bash
python gaze_onnx/experiments/make_roi_reference.py \
  --video "6月1日.mp4" \
  --out-dir gaze_onnx/experiments/roi_refs/car1_person1
```

2) 生成多域标注包（直接从视频+ROI抽样）：

```bash
python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
  --domains-csv gaze_onnx/experiments/manifests/two_domain_videos.v1.csv \
  --out-dir gaze_onnx/experiments/anno_two_domain_v1 \
  --samples-per-domain 600
```

3) Web 标注（支持 `Other`）：

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_two_domain_v1 \
  --port 8000
```

---

## 新增：抗明暗 + 抗车/人域差异训练

你已经完成两域标注后，推荐按下面流程训练：

1) 先做跨域验证（留一域）评估泛化能力

```bash
python gaze_onnx/experiments/prepare_cls_dataset_from_pack.py \
  --samples-dir gaze_onnx/experiments/anno_two_domain_v3_ratio_roi_run1 \
  --out-dir gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2 \
  --split-mode domain_holdout \
  --val-domain car2_person2 \
  --augment-minority \
  --target-train-per-class 550
```

2) 用更强增强训练分类模型（`robust` 预设）

```bash
python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2 \
  --mode train \
  --model yolov8n-cls.pt \
  --epochs 80 \
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --aug-preset robust \
  --name gaze_two_domain_holdout_car2
```

3) 评估：

```bash
python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_two_domain_holdout_car2 \
  --mode eval \
  --weights gaze_onnx/experiments/runs_cls/gaze_two_domain_holdout_car2/weights/best.pt
```

说明：
- `domain_holdout` 会把一个域完整留作验证，避免“同车同人泄漏”。
- `robust` 预设包含更强颜色/亮度/几何扰动（`randaugment + mixup + cutmix + erasing`）。
- 如果要训练最终上线模型，可再用 `domain_stratified` 生成训练集，让两域都进入训练。
