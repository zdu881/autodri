# autodri

这是本仓库唯一的主 README。

如果你是第一次接触这个项目，请只看这一份文档，不要再去找别的 README。

这个仓库主要做三件事：
- `gaze`：识别驾驶员注视位置
- `hand-on-wheel`：识别手是否扶方向盘
- `few-shot`：新车新人的少样本适配训练

当前研究目标很明确：
- 模型尽量不受明暗变化影响
- 模型尽量不受车辆差异和人员差异影响
- 对新车新人的新视频，只做少量标注后快速适配

---

## 1. 你属于哪一类使用者

### A. 只负责标注的人

你只需要看这几节：
- `2. 第一次使用前要知道什么`
- `3. 标签怎么打`
- `4. 如何打开标注网页`
- `5. 标注结果保存在哪里`
- `11. 常见问题`

你不需要懂训练，也不需要改代码。

### B. 负责准备新被试标注包的人

你需要看：
- `6. 文件和目录应该怎么放`
- `7. 如何上线一个新的被试标注`

### C. 负责训练和导出模型的人

你需要看：
- `8. 如何做 200-shot few-shot 训练`
- `9. 如何导出 ONNX 并做推理`

---

## 2. 第一次使用前要知道什么

### 2.1 推荐环境

默认使用现成的 Conda 环境：

```bash
conda activate adri
```

如果这是新机器，至少需要安装项目依赖：

```bash
python -m pip install -r driver_monitor/requirements.txt
python -m pip install ultralytics onnxruntime opencv-python openpyxl pandas
```

说明：
- `driver_monitor/requirements.txt` 已经通过 VCS 安装 `GroundingDINO`
- 不需要再单独克隆一个 `GroundingDINO/` 仓库
- 如果 VCS 包异常，可以强制重装

```bash
python -m pip install --upgrade --force-reinstall -r driver_monitor/requirements.txt
```

检查 `groundingdino` 实际从哪里导入：

```bash
python - <<'PY'
import groundingdino
from pathlib import Path
print(Path(groundingdino.__file__).resolve())
PY
```

### 2.2 你必须知道：模型文件不在 Git 里

`models/` 和 `data/` 默认都在 `.gitignore` 中。

这意味着：
- 仓库代码可以 `git clone`
- 但模型文件不会自动出现
- 原始视频数据也不会自动出现

如果没有模型，推理和 ROI 自动分配都会失败。

### 2.3 最常用的模型文件位置

推荐把文件放在下面这些路径：

- `models/scrfd_person_2.5g.onnx`
- `models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx`
- `models/groundingdino_swint_ogc.pth`

说明：
- 第一个是人脸检测模型
- 第二个是 gaze 分类模型
- 第三个是 wheel 项目用的 GroundingDINO 权重

快速自检：

```bash
python - <<'PY'
from pathlib import Path
checks = {
    "scrfd": Path("models/scrfd_person_2.5g.onnx"),
    "gaze_cls": Path("models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx"),
    "wheel_weight_models": Path("models/groundingdino_swint_ogc.pth"),
    "wheel_weight_legacy": Path("GroundingDINO/weights/groundingdino_swint_ogc.pth"),
}
for name, path in checks.items():
    print(f"{name:22s} {'OK' if path.exists() else 'MISSING'}  {path}")
PY
```

---

## 3. 标签怎么打

Web 标注工具里目前有 5 个状态：
- `Forward`
- `Non-Forward`
- `In-Car`
- `Other`
- `Unknown`

它们的含义如下。

### 3.1 `Forward`

驾驶员看向前方道路。

快捷键：
- `1`

### 3.2 `Non-Forward`

驾驶员没有看前方道路，但仍然是驾驶相关视线。

常见例子：
- 看中控
- 看后视镜
- 看侧方

快捷键：
- `2`

### 3.3 `In-Car`

驾驶员看向车内其它区域。

常见例子：
- 看副驾
- 看车内物品
- 看手上的东西

快捷键：
- `3`

### 3.4 `Other`

这是非常重要的类。

只要图片不属于我们真正关心的驾驶员 gaze 状态，就优先打 `Other`。

典型情况：
- ROI 对错了，截图里根本不是驾驶员区域
- 图里没有人脸
- 驾驶员不在驾驶位
- 画面里主要是副驾或后排乘客
- 图像严重异常，根本不是有效驾驶员样本
- 该帧不在我们的研究兴趣内

快捷键：
- `4`
- `o`

一句话规则：
- “不是有效驾驶员 gaze 样本”，优先打 `Other`

### 3.5 `Unknown`

`Unknown` 不是研究目标类，它更像“暂时无法判断”。

只有在下面这种情况才建议使用：
- 画面看起来像驾驶员区域
- 但你真的无法判断是 `Forward / Non-Forward / In-Car`

快捷键：
- `0`

一句话规则：
- “是驾驶员区域，但我实在判断不出来”，才打 `Unknown`

### 3.6 `Other` 和 `Unknown` 的关系

这是最容易混淆的地方。

正确理解：
- `Other`：这张图本身就不属于我们要研究的有效驾驶 gaze 样本
- `Unknown`：这张图可能是有效驾驶 gaze 样本，但人眼也判断不清

请优先使用 `Other`，不要把“没人、错 ROI、副驾、后排、驾驶员离位”打成 `Unknown`。

---

## 4. 如何打开标注网页

### 4.1 如果维护人员已经把网页开好了

最简单。

直接在浏览器打开他给你的网址，例如：

- `http://10.30.2.11:8001`

你不需要会命令行。

### 4.2 如果你需要自己在服务器上启动标注网页

进入仓库根目录后运行：

```bash
conda activate adri
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_p2_gaze_same_as_p1_v1 \
  --host 0.0.0.0 \
  --port 8001
```

然后浏览器打开：

- 本机打开：`http://127.0.0.1:8001`
- 局域网打开：`http://你的服务器IP:8001`

### 4.3 标注时常用快捷键

- `1` = `Forward`
- `2` = `Non-Forward`
- `3` = `In-Car`
- `4` = `Other`
- `o` = `Other`
- `0` = `Unknown`
- `n` = 下一张
- `b` = 上一张

### 4.4 标注页面关闭了怎么办

不用慌。

`labels.csv` 是持续保存的。

重新打开同一个标注包，再启动同一条命令即可继续标，不会从头开始。

---

## 5. 标注结果保存在哪里

每一个标注包目录通常长这样：

```text
gaze_onnx/experiments/anno_pX_gaze_same_as_p1_v1/
├── images/
├── manifest.csv
└── labels.csv
```

含义：
- `images/`：待标注图片
- `manifest.csv`：图片清单
- `labels.csv`：人工标注结果

最重要的是：
- 你的劳动成果就在 `labels.csv`

例如：
- `gaze_onnx/experiments/anno_p2_gaze_same_as_p1_v1/labels.csv`

切换到下一个被试前，建议备份：

```bash
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p gaze_onnx/experiments/label_backups/$stamp
cp gaze_onnx/experiments/anno_p2_gaze_same_as_p1_v1/labels.csv \
  gaze_onnx/experiments/label_backups/$stamp/
```

本仓库目前也统一把历史备份放在：
- `gaze_onnx/experiments/label_backups/`

---

## 6. 文件和目录应该怎么放

请尽量按下面的方式放，不要自己发明新目录。

### 6.1 仓库核心目录

- `gaze_onnx/`
  - gaze 推理主脚本：`gaze_onnx/gaze_state_cls.py`
  - 标注、few-shot、评估脚本：`gaze_onnx/experiments/`
- `driver_monitor/`
  - hand-on-wheel：`driver_monitor/hand_on_wheel.py`
  - 眼睛 EAR demo：`driver_monitor/eye_state_ear.py`
- `models/`
  - 放各种本地模型权重
- `data/`
  - 放本地视频数据
- `docs/`
  - 放论文和补充说明

### 6.2 原始视频推荐目录

自然驾驶视频统一放这里：

- `data/natural_driving/pX/剪辑好的视频/`

例如：
- `data/natural_driving/p2/剪辑好的视频/`
- `data/natural_driving/p6/剪辑好的视频/`

### 6.3 标注包目录

建议命名：

- `gaze_onnx/experiments/anno_pX_gaze_same_as_p1_v1`

### 6.4 ROI 分配和分析目录

建议放在：

- `data/natural_driving/pX/analysis/`

常见文件：
- `pX_dual_roi_assignment.csv`
- `pX_dual_roi_previews/`

### 6.5 manifest 目录

建议放在：

- `gaze_onnx/experiments/manifests/`

例如：
- `p6_target_videos.v1.csv`
- `p6_gaze_domains.same_as_p1_v1.csv`

---

## 7. 如何上线一个新的被试标注

这一节给“准备新被试”的人看。

下面用 `p6` 举例。

### 7.1 第一步：从 SMB 拉视频

如果本地还没有该被试视频：

```bash
export SMB_PASSWORD='你的密码'

python gaze_onnx/experiments/sync_natural_driving_smb.py \
  --user nyz \
  --password-env SMB_PASSWORD \
  --participant p6 \
  --out-root data/natural_driving
```

脚本特点：
- 不需要手动挂载网络盘
- 会跳过已经完整下载的文件
- 会把视频放到 `data/natural_driving/p6/剪辑好的视频/`

### 7.2 第二步：从 Excel 生成本次真正要处理的视频清单

我们现在不是“一个被试所有视频都标”，而是只标 Excel 指定的目标视频。

用下面这个脚本生成清单：

```bash
python gaze_onnx/experiments/build_participant_video_manifest_from_xlsx.py \
  --xlsx "自然驾驶_视频标注情况 (3).xlsx" \
  --sheet p6 \
  --videos-root "data/natural_driving/p6/剪辑好的视频" \
  --out-csv gaze_onnx/experiments/manifests/p6_target_videos.v1.csv
```

这个脚本会做三件事：
- 读取 Excel 中 `p6` 工作表第一列的视频文件夹名称
- 在本地视频目录中寻找对应文件夹
- 为每个目标文件夹挑选一个实际 `.mp4` 文件写入 CSV

它还会自动处理我们已经遇到过的命名差异：
- `20241121 090208-092608` 这类 Excel 日期格式
- `11.21 090208-092608` 这类本地目录格式
- 逗号和点混用的问题
- 同一文件夹内多个 `.mp4` 时，优先选择非空且更大的文件

### 7.3 第三步：自动判别 gaze / wheel 双窗口位置

我们当前固定有两个候选区域：

- ROI-A：`0,0,1900,1100`
- ROI-B：`1900,660,3300,1400`

由于剪辑人员有时会把 gaze 和 wheel 左右放反，所以不能直接假设每个视频都一样。

先跑自动分配：

```bash
python gaze_onnx/experiments/assign_dual_roi.py \
  --videos-csv gaze_onnx/experiments/manifests/p6_target_videos.v1.csv \
  --roi-a 0 0 1900 1100 \
  --roi-b 1900 660 3300 1400 \
  --base-gaze-roi B \
  --samples 32 \
  --assignment-csv data/natural_driving/p6/analysis/p6_dual_roi_assignment.csv \
  --preview-dir data/natural_driving/p6/analysis/p6_dual_roi_previews
```

### 7.4 第四步：按“同一被试内部多数 ROI”统一 gaze 区域

这是本项目现在的固定规则。

原因：
- 同一个被试内部，绝大多数视频的左右布局应该一致
- 个别视频如果被剪反，应该服从该被试内部的多数结果

生成 gaze 标注域清单：

```bash
python gaze_onnx/experiments/build_domains_csv_from_dual_assignment.py \
  --assignment-csv data/natural_driving/p6/analysis/p6_dual_roi_assignment.csv \
  --task gaze \
  --domain-id p6 \
  --samples-per-video 24 \
  --samples-per-video-uncertain 32 \
  --include-uncertain \
  --require-status-ok \
  --participant-majority-roi \
  --out-csv gaze_onnx/experiments/manifests/p6_gaze_domains.same_as_p1_v1.csv
```

### 7.5 第五步：生成标注包

```bash
python gaze_onnx/experiments/create_multidomain_annotation_pack.py \
  --domains-csv gaze_onnx/experiments/manifests/p6_gaze_domains.same_as_p1_v1.csv \
  --out-dir gaze_onnx/experiments/anno_p6_gaze_same_as_p1_v1 \
  --seed 42 \
  --read-mode seek
```

### 7.6 第六步：上线网页

```bash
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir gaze_onnx/experiments/anno_p6_gaze_same_as_p1_v1 \
  --host 0.0.0.0 \
  --port 8001
```

### 7.7 切换到下一个被试前，一定要先备份旧标签

例如当前网页还挂着 `p2`：

```bash
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p gaze_onnx/experiments/label_backups/$stamp/anno_p2_gaze_same_as_p1_v1
cp gaze_onnx/experiments/anno_p2_gaze_same_as_p1_v1/labels.csv \
  gaze_onnx/experiments/label_backups/$stamp/anno_p2_gaze_same_as_p1_v1/
cp gaze_onnx/experiments/anno_p2_gaze_same_as_p1_v1/manifest.csv \
  gaze_onnx/experiments/label_backups/$stamp/anno_p2_gaze_same_as_p1_v1/
```

然后再停旧服务、开新服务。

---

## 8. 如何做 200-shot few-shot 训练

这一节给训练人员看。

我们的当前研究重点是驾驶状态三分类：
- `Forward`
- `Non-Forward`
- `In-Car`

说明：
- `Other` 很重要，标注阶段必须保留
- 但当前很多 few-shot 训练集会优先只用上面 3 个驾驶状态类
- `Unknown` 一般不进入训练集

### 8.1 从一个完整标注包里抽 200 张 few-shot 包

```bash
python gaze_onnx/experiments/build_fewshot_pack.py \
  --src-pack gaze_onnx/experiments/anno_p6_gaze_same_as_p1_v1 \
  --out-pack gaze_onnx/experiments/anno_p6_gaze_200shot_v1 \
  --num-samples 200 \
  --sample-mode by_video_uniform \
  --keep-labeled
```

### 8.2 生成分类训练集

```bash
python gaze_onnx/experiments/prepare_cls_dataset_from_pack.py \
  --samples-dir gaze_onnx/experiments/anno_p6_gaze_200shot_v1 \
  --out-dir gaze_onnx/experiments/cls_dataset_p6_200shot_driveonly_v1 \
  --split-mode random \
  --val-ratio 0.2 \
  --labels Forward Non-Forward In-Car \
  --augment-minority \
  --target-train-map "Forward=100,In-Car=100,Non-Forward=100" \
  --copy-mode hardlink
```

### 8.3 开始训练

推荐基于已有较强基模继续 few-shot 微调：

```bash
conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_p6_200shot_driveonly_v1 \
  --mode train \
  --model runs/classify/gaze_onnx/experiments/runs_cls/gaze_final_two_domain_genv2_cpu/weights/best.pt \
  --epochs 60 \
  --batch 16 \
  --device 0 \
  --aug-preset genv3 \
  --name gaze_p6_200shot_driveonly_ft_v1_gpu0
```

说明：
- `genv3` 是当前偏泛化的增强配置
- 少样本适配时，通常优先在已有泛化基模上微调

### 8.4 训练完成后做验证

```bash
conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_p6_200shot_driveonly_v1 \
  --mode eval \
  --weights runs/classify/gaze_onnx/experiments/runs_cls/gaze_p6_200shot_driveonly_ft_v1_gpu0/weights/best.pt \
  --device 0
```

---

## 9. 如何导出 ONNX 并做推理

### 9.1 导出 ONNX

```bash
conda run -n adri python gaze_onnx/experiments/train_gaze_cls.py \
  --data gaze_onnx/experiments/cls_dataset_p6_200shot_driveonly_v1 \
  --mode export \
  --weights runs/classify/gaze_onnx/experiments/runs_cls/gaze_p6_200shot_driveonly_ft_v1_gpu0/weights/best.pt
```

导出完成后，把生成的 `.onnx` 放到：

- `models/`

例如：
- `models/gaze_cls_p6_200shot_driveonly_ft_v1.onnx`

### 9.2 对单个视频做 gaze 推理

```bash
python gaze_onnx/gaze_state_cls.py \
  --video /path/to/input.mp4 \
  --scrfd models/scrfd_person_2.5g.onnx \
  --roi 1900 660 3300 1400 \
  --face-priority right_to_left_track \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video gaze_onnx/output/gaze_demo.mp4 \
  --csv gaze_onnx/output/gaze_demo.csv
```

### 9.3 只跑视频中的一个片段

例如从 `120s` 开始跑 `20s`：

```bash
python gaze_onnx/gaze_state_cls.py \
  --video /path/to/input.mp4 \
  --start-sec 120 \
  --duration-sec 20 \
  --scrfd models/scrfd_person_2.5g.onnx \
  --roi 1900 660 3300 1400 \
  --face-priority right_to_left_track \
  --cls-model models/gaze_cls_p1_200shot_driveonly_ft_v1.onnx \
  --out-video gaze_onnx/output/gaze_demo_seg_120_20.mp4 \
  --csv gaze_onnx/output/gaze_demo_seg_120_20.csv
```

### 9.4 生成 20s 事件级聚合结果

```bash
python gaze_onnx/experiments/aggregate_gaze_windows.py \
  --csv gaze_onnx/output/gaze_demo.csv \
  --out-csv gaze_onnx/output/gaze_demo.event20s.csv \
  --window-sec 20
```

### 9.5 多人脸场景如何减少副驾和后排干扰

当前推荐配置：

- `--face-priority right_to_left_track`

原因：
- 在一个 ROI 内优先关注偏右的人脸
- 再结合时间跟踪稳定性
- 用来压制副驾和后排干扰

---

## 10. hand-on-wheel 项目怎么跑

### 10.1 基础运行

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --output driver_monitor/output/hand_on_wheel.mp4 \
  --weights models/groundingdino_swint_ogc.pth \
  --select-roi
```

### 10.2 固定 ROI 运行

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --roi 950 300 1650 690 \
  --weights models/groundingdino_swint_ogc.pth \
  --output driver_monitor/output/hand_on_wheel_fixed_roi.mp4 \
  --state-csv driver_monitor/output/hand_on_wheel_states.csv
```

### 10.3 加时间窗口多数投票

例如使用 `30s` 稳态窗口：

```bash
python driver_monitor/hand_on_wheel.py \
  --video /path/to/input.mp4 \
  --roi 950 300 1650 690 \
  --weights models/groundingdino_swint_ogc.pth \
  --decision-window-sec 30 \
  --output driver_monitor/output/hand_on_wheel_30s.mp4 \
  --state-csv driver_monitor/output/hand_on_wheel_30s_states.csv
```

这个功能是为了减少逐帧抖动，更接近真实驾驶监测应用。

### 10.4 EAR 闭眼 demo

仓库里还有一个眼睛状态脚本：

- `driver_monitor/eye_state_ear.py`

它目前更适合作为 demo 或原型，不是当前主线训练流程的一部分。

---

## 11. 常见问题

### 11.1 浏览器打不开标注页面

先确认服务有没有启动：

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

如果没有输出，说明服务没起来。

### 11.2 报错 `Address already in use`

说明端口被占用了。

查看谁占着：

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

如果确定要切换服务，可以结束旧进程，或者换端口，例如 `8011`。

### 11.3 标注一半网页关了，数据会不会丢

通常不会。

因为每次点击标签后，都会更新对应标注包里的：
- `labels.csv`

恢复方法：
- 重新启动同一个标注包
- 再打开网页继续

### 11.4 图片里完全没人，或者明显不是驾驶员区域，该怎么打

打：
- `Other`

不要打：
- `Unknown`

### 11.5 `Unknown` 是不是等于以前说的 `unclear`

可以近似这么理解，但我们现在统一用 `Unknown` 这个词。

同时请记住：
- 不是有效驾驶员区域，用 `Other`
- 真的是驾驶员区域但看不清，才用 `Unknown`

### 11.6 同一个被试内部，ROI 是不是每个视频都可以不一样

当前规则不是这样。

当前规则是：
- 先自动分配每个视频的 ROI
- 再在同一个被试内部，使用“多数 ROI”统一 gaze 区域

### 11.7 为什么要保留 `Other`

因为现实数据里会有这些情况：
- 驾驶员不在位
- 画面取错区域
- 乘客占主导
- 无法纳入研究兴趣

如果不保留 `Other`，训练数据会被严重污染。

### 11.8 为什么训练时很多时候只用三类

因为当前研究兴趣主要是驾驶状态：
- `Forward`
- `Non-Forward`
- `In-Car`

但这不代表 `Other` 不重要。

`Other` 在数据清洗和推理阶段仍然非常重要。

---

## 12. 建议的最小工作流

如果你今天只想把事情做完，不想研究细节，就按下面执行。

### 12.1 只做标注

1. 拿到维护人员给你的网址
2. 打开网页
3. 用 `1/2/3/4/0` 打标签
4. 标完告诉维护人员 `labels.csv` 已更新

### 12.2 准备一个新被试

1. 拉视频
2. 从 Excel 生成目标视频清单
3. 自动做双 ROI 分配
4. 用被试内部多数 ROI 生成 gaze domains
5. 生成标注包
6. 备份旧 `labels.csv`
7. 挂网页

### 12.3 做 few-shot 训练

1. 从完整标注包抽 `200-shot`
2. 生成三分类训练集
3. 基于已有泛化基模微调
4. 导出 ONNX
5. 跑单视频验证

---

## 13. 现在这个仓库里最常用的脚本

- `gaze_onnx/experiments/web_label_tool.py`
- `gaze_onnx/experiments/build_participant_video_manifest_from_xlsx.py`
- `gaze_onnx/experiments/assign_dual_roi.py`
- `gaze_onnx/experiments/build_domains_csv_from_dual_assignment.py`
- `gaze_onnx/experiments/create_multidomain_annotation_pack.py`
- `gaze_onnx/experiments/build_fewshot_pack.py`
- `gaze_onnx/experiments/prepare_cls_dataset_from_pack.py`
- `gaze_onnx/experiments/train_gaze_cls.py`
- `gaze_onnx/gaze_state_cls.py`
- `driver_monitor/hand_on_wheel.py`

如果你不知道从哪里开始，默认先从这份 README 找对应章节，不要先去翻源码。
