# 从零开始标注流程（两车两人，面向泛化）

目标：构建可泛化到任意车/任意人的车内注视判别模型。

当前标签体系（4类）：
- `Forward`
- `Non-Forward`
- `In-Car`
- `Other`（不在车内/无法判定/非目标驾驶员）

---

## 0) 当前已生成的数据

已为两域生成 ROI 参考图（带坐标网格和分区编号）：
- `gaze_onnx/experiments/roi_refs/car1_person1/`
- `gaze_onnx/experiments/roi_refs/car2_person2/`

已生成第一版待标注包（ROI 裁剪图）：
- `gaze_onnx/experiments/anno_two_domain_v1/images/`
- `gaze_onnx/experiments/anno_two_domain_v1/manifest.csv`

当前样本数：
- `car1_person1`: 600
- `car2_person2`: 528

---

## 1) 先给 ROI 粗坐标（你来提供）

请基于 `*_grid.jpg` 给每个域一个大概 ROI：
- 格式：`x1 y1 x2 y2`
- 例如：`950 300 1650 690`

你只需要回复我两行：
- `car1_person1: x1 y1 x2 y2`
- `car2_person2: x1 y1 x2 y2`

我会据此重建标注包（只保留有效驾驶员区域）。

---

## 2) 生成/重建两域标注包（命令）

域配置文件：
- `gaze_onnx/experiments/manifests/two_domain_videos.v1.csv`

重建标注包：
```bash
python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
  --domains-csv gaze_onnx/experiments/manifests/two_domain_videos.v1.csv \
  --out-dir gaze_onnx/experiments/anno_two_domain_v1 \
  --samples-per-domain 600 \
  --seed 42
```

---

## 3) 开始人工标注（推荐 Web）

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_two_domain_v1 \
  --port 8000
```

浏览器打开：
- `http://127.0.0.1:8000`

快捷键：
- `1` Forward
- `2` Non-Forward
- `3` In-Car
- `4` Other
- `0` Unknown/清除
- `n` 下一张
- `b` 上一张

输出标签文件：
- `gaze_onnx/experiments/anno_two_domain_v1/labels.csv`

---

## 4) 标注质检建议（泛化关键）

至少保证每个域都包含：
- 白天/夜晚
- 头部轻微遮挡
- 大角度侧头
- 短时低头（中控/仪表）
- `Other` 典型场景（空座/乘客干扰/无法判定）

并保持每个域四类都尽量有样本，不要只在单域出现某一类。

---

## 5) 训练准备与训练

当前 `prepare_cls_dataset.py` 支持按 `labels.csv` 生成分类数据集。

如果你要直接从当前标注包训练，我会在下一步帮你补一个
“从 `anno_two_domain_v1/labels.csv` 一键转训练集”的脚本，避免你手动拼装。

---

## 6) 泛化评估建议（必须）

至少做三种评估：
1. 训练域混合随机划分（基础）
2. 留一域验证（train car1_person1 -> test car2_person2，反之亦然）
3. 新域测试（未来新增车/人数据）

最终以 `macro-F1` 和每类召回率为主指标，不只看 overall accuracy。
