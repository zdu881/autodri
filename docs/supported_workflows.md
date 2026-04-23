# Supported Workflows

Canonical entrypoints use `python -m autodri.cli.<name>`.

## Core Runtime

- `python -m autodri.cli.gaze_state_cls`
- `python -m autodri.cli.hand_on_wheel`

## Data Prep And Annotation

- `python -m autodri.cli.web_label_tool`
- `python -m autodri.cli.build_participant_video_manifest_from_xlsx`
- `python -m autodri.cli.assign_dual_roi`
- `python -m autodri.cli.build_domains_csv_from_dual_assignment`
- `python -m autodri.cli.create_multidomain_annotation_pack`
- `python -m autodri.cli.build_fewshot_pack`
- `python -m autodri.cli.prepare_cls_dataset_from_pack`
- `python -m autodri.cli.train_gaze_cls`

## Inference And Metrics

- `python -m autodri.cli.run_p1_infer_plan`
- `python -m autodri.cli.run_domains_gaze_infer`
- `python -m autodri.cli.build_p1_schedule_windows`
- `python -m autodri.cli.compute_p1_window_metrics`
- `python -m autodri.cli.build_all_participants_window_metrics`
- `python -m autodri.cli.build_participants_results_summary`

