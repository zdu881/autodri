# 从零开始标注流程（两车两人，面向泛化）

如果你只是负责“图片打标签”，请优先看：
- `docs/annotation_quickstart.md`

目标：构建可泛化到任意车/任意人的车内注视判别模型。

当前标签体系（4类）：
- `Forward`
- `Non-Forward`
- `In-Car`
- `Other`（不在车内/无法判定/非目标驾驶员）

特殊说明（标注必须统一）：
- 如果图片是“无人区域”（ROI 对不上、没包含到驾驶员），统一标 `Other`。

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

---

## 7) 固化批量流程（p1 / p2 / p3 ...）

下面这套流程用于后续大量参与者数据，避免每次手工重复。

### 步骤 A：从 SMB 拉取指定 pX 视频（不挂载）

脚本：`gaze_onnx/experiments/sync_natural_driving_smb.py`

```bash
export SMB_PASSWORD='你的密码'

python gaze_onnx/experiments/sync_natural_driving_smb.py \
  --user nyz \
  --password-env SMB_PASSWORD \
  --participant p1 \
  --out-root data/natural_driving
```

输出目录（示例）：
- `data/natural_driving/p1/剪辑好的视频/`

脚本会自动：
1. 递归遍历远端 `p1/剪辑好的视频`
2. 按文件大小跳过已下载文件（增量同步）
3. 下载完成后输出远端/本地一致性统计（数量、总字节、缺失、大小不一致）

### 步骤 B：生成 ROI 标注包（你先标模型区域）

脚本：`gaze_onnx/experiments/prepare_roi_label_pack.py`

```bash
python gaze_onnx/experiments/prepare_roi_label_pack.py \
  --videos-root data/natural_driving/p1/剪辑好的视频 \
  --out-dir gaze_onnx/experiments/roi_refs/p1_label_pack \
  --glob '*.mp4' \
  --sample-position first \
  --grid-step 220
```

生成内容：
- `roi_label_manifest.csv`：待填写 ROI 坐标（`roi_x1,roi_y1,roi_x2,roi_y2`）
- `refs/*__grid.jpg`：带坐标网格的参考图
- `invalid_or_unreadable_videos.txt`：损坏/占位视频清单（如 0B mp4）

### 步骤 C：ROI 清单转 domains.csv

你填完 `roi_label_manifest.csv` 后执行：

```bash
python gaze_onnx/experiments/build_domains_csv_from_roi_manifest.py \
  --roi-manifest gaze_onnx/experiments/roi_refs/p1_label_pack/roi_label_manifest.csv \
  --out-csv gaze_onnx/experiments/manifests/p1_domains.csv \
  --domain-id p1 \
  --samples-per-video 150
```

### 步骤 D：抽样生成小样本标注包（你标小样本）

```bash
python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
  --domains-csv gaze_onnx/experiments/manifests/p1_domains.csv \
  --out-dir gaze_onnx/experiments/anno_p1_v1 \
  --seed 42
```

启动 Web 标注：

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_p1_v1 \
  --port 8000
```

### 步骤 E：模型推理辅助标注 / 训练

标注完成后，后续走现有训练与评估流程：
1. 构建分类训练集（`prepare_cls_dataset_from_pack.py`）
2. 训练（`train_gaze_cls.py`）
3. 跨域评估（`cross_domain_eval.py`，帧级 + 事件级）

---

## 8) 双区域视频的自动纠错（gaze/wheel ROI 交换）

如果剪辑人员把两个窗口位置放反，可用固定双 ROI 自动分配脚本先做纠错。

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
  --assignment-csv gaze_onnx/experiments/output/p1_dual_roi_assignment.csv \
  --preview-dir gaze_onnx/experiments/output/p1_dual_roi_previews
```

关键输出字段：
- `gaze_roi` / `wheel_roi`：当前视频最终使用的 ROI
- `swapped=1`：说明该视频相对于默认映射（A->gaze）发生了交换
- `assignment_uncertain=1`：低置信，建议人工快速复核预览图
- 默认会跳过坏视频并继续；若希望出现错误即返回非 0，可加 `--fail-on-error`

### 推荐目录约定（按参与者）

1. 原始视频：`data/natural_driving/pX/剪辑好的视频/`
2. ROI 标注包：`gaze_onnx/experiments/roi_refs/pX_label_pack/`
3. 域配置：`gaze_onnx/experiments/manifests/pX_domains.csv`
4. 小样本标注包：`gaze_onnx/experiments/anno_pX_v1/`
