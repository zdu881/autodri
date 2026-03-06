# 标注快速上手（小白版）

这份文档只给“做标注的人”使用，不涉及训练和模型开发。

目标：把图片标到 `labels.csv`，交回给研发同学。

---

## 1. 你只需要知道的三件事

1. 每张图要打一个标签：`Forward / Non-Forward / In-Car / Other`
2. 每次点击标签会自动保存到 `labels.csv`
3. 进度到 `Progress: 已标数/总数` 即可结束

必须注意（本项目特有）：
- 有一小部分图片会出现“无人区域”（ROI 对不上、没框到驾驶员）。
- 这种情况统一标 `Other`，不要标 `Unknown`。

---

## 2. 启动标注页面

在仓库根目录执行：

```bash
conda activate adri
PACK_DIR=gaze_onnx/experiments/anno_p1_gaze_200shot_v1
python gaze_onnx/experiments/web_label_tool.py \
  --samples-dir "$PACK_DIR" \
  --port 8001
```

只需要改一处：`PACK_DIR=...`（标注包目录）。

然后在浏览器打开：
- 本机：`http://127.0.0.1:8001`
- 如果是远程服务器：先做端口转发再打开本机地址  
  `ssh -L 8001:127.0.0.1:8001 <user>@<server>`

---

## 3. 标签定义（按这个标准）

- `Forward`：看前方道路。
- `In-Car`：看车内区域（中控、仪表、车内后视镜等）。
- `Non-Forward`：明显不是看前方，且不属于 `In-Car`（例如看侧窗/侧后方向）。
- `Other`：不在驾驶位、看不到有效人脸、不是目标驾驶员、无法判断。
- `Other` 还包括：ROI 对不上导致的无人区域（画面里没有驾驶员）。

如果你不确定：
- 优先用 `Other`，不要硬猜。

---

## 4. 快捷键（建议全程用键盘）

- `1` -> `Forward`
- `2` -> `Non-Forward`
- `3` -> `In-Car`
- `4` -> `Other`
- `0` -> `Unknown/Clear`（清除本张标签）
- `n` -> 下一张
- `b` -> 上一张

---

## 5. 标注结束后交付什么

只需要交付这个文件：
- `.../labels.csv`（就在 `--samples-dir` 目录下）

可选自检（看还有没有未标）：

```bash
PACK_DIR=gaze_onnx/experiments/anno_p1_gaze_200shot_v1
python - <<'PY'
import csv, os
from pathlib import Path
pack_dir = os.environ["PACK_DIR"]
p = Path(pack_dir) / "labels.csv"
rows = list(csv.DictReader(p.open("r", encoding="utf-8")))
done = sum(1 for r in rows if (r.get("label") or "").strip())
print(f"labeled={done} total={len(rows)} unlabeled={len(rows)-done}")
PY
```

---

## 6. 常见问题

- 页面打不开：
  - 先确认终端里标注服务在运行（有 `Serving labels for ...` 日志）
  - 端口被占用就换一个，例如 `--port 8011`
- 图片不显示：
  - 通常是 `samples-dir` 路径不对，确认目录下有 `manifest.csv` 和 `images/`
- 误标了：
  - 按 `b` 回上一张重标，或按 `0` 清除后重标
