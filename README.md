# autodri

## 1. 工作区准备

- 代码仓默认只保留源码、轻量文档和最小样例。
- 把非代码资源放到 `${AUTODRI_WORKSPACE}`。如果不设置环境变量，默认工作区是仓库同级目录 `autodri_workspace/`。
- 工作区固定目录：
  - `data/`
  - `models/`
  - `artifacts/`
  - `archive/`
  - `sources/`

常用布局：

```text
autodri/
autodri_workspace/
  data/
  models/
  artifacts/
  archive/
  sources/
```

## 2. 支持的规范入口

规范命令统一使用：

```bash
python -m autodri.cli.<name>
```

当前支持面：

- `gaze_state_cls`
- `hand_on_wheel`
- `web_label_tool`
- `build_participant_video_manifest_from_xlsx`
- `assign_dual_roi`
- `build_domains_csv_from_dual_assignment`
- `create_multidomain_annotation_pack`
- `build_fewshot_pack`
- `prepare_cls_dataset_from_pack`
- `train_gaze_cls`
- `run_p1_infer_plan`
- `run_domains_gaze_infer`
- `build_p1_schedule_windows`
- `compute_p1_window_metrics`
- `build_all_participants_window_metrics`
- `build_participants_results_summary`

详情见 [supported_workflows.md](docs/supported_workflows.md)。

## 3. 旧入口映射

旧脚本路径仍可运行，但现在只是兼容 wrapper：

- `gaze_onnx/gaze_state_cls.py` -> `python -m autodri.cli.gaze_state_cls`
- `driver_monitor/hand_on_wheel.py` -> `python -m autodri.cli.hand_on_wheel`
- `gaze_onnx/experiments/web_label_tool.py` -> `python -m autodri.cli.web_label_tool`
- `gaze_onnx/experiments/build_participant_video_manifest_from_xlsx.py` -> `python -m autodri.cli.build_participant_video_manifest_from_xlsx`
- `gaze_onnx/experiments/assign_dual_roi.py` -> `python -m autodri.cli.assign_dual_roi`
- `gaze_onnx/experiments/build_domains_csv_from_dual_assignment.py` -> `python -m autodri.cli.build_domains_csv_from_dual_assignment`
- `gaze_onnx/experiments/create_multidomain_annotation_pack.py` -> `python -m autodri.cli.create_multidomain_annotation_pack`
- `gaze_onnx/experiments/build_fewshot_pack.py` -> `python -m autodri.cli.build_fewshot_pack`
- `gaze_onnx/experiments/prepare_cls_dataset_from_pack.py` -> `python -m autodri.cli.prepare_cls_dataset_from_pack`
- `gaze_onnx/experiments/train_gaze_cls.py` -> `python -m autodri.cli.train_gaze_cls`
- `gaze_onnx/experiments/run_p1_infer_plan.py` -> `python -m autodri.cli.run_p1_infer_plan`
- `gaze_onnx/experiments/run_domains_gaze_infer.py` -> `python -m autodri.cli.run_domains_gaze_infer`
- `gaze_onnx/experiments/build_p1_schedule_windows.py` -> `python -m autodri.cli.build_p1_schedule_windows`
- `gaze_onnx/experiments/compute_p1_window_metrics.py` -> `python -m autodri.cli.compute_p1_window_metrics`
- `gaze_onnx/experiments/build_all_participants_window_metrics.py` -> `python -m autodri.cli.build_all_participants_window_metrics`
- `gaze_onnx/experiments/build_participants_results_summary.py` -> `python -m autodri.cli.build_participants_results_summary`

## 4. Legacy 说明

- 数据、模型、报表和历史标注备份不再是仓库源码的一部分。
- 脚本默认优先从 `${AUTODRI_WORKSPACE}` 解析资源；必要时才回退到旧 repo 路径，并打印弃用警告。
- 需要归档的历史内容见 [legacy_inventory.md](docs/legacy_inventory.md)。
