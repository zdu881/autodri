# Generalization Dataset Plan

Goal: train a gaze model that generalizes across drivers and vehicles.

## Priority public datasets

1. DMD (Driver Monitoring Dataset)
- Multi-modal in-cabin data (RGB/IR/Depth) with driver-monitoring tasks.
- Useful for gaze + hands-on-wheel related transfer.

2. 100-Driver
- Large multi-driver, multi-vehicle, multi-view dataset.
- Best fit for cross-driver / cross-vehicle validation.

3. Drive&Act
- Rich in-cabin action labels, multi-view setup.
- Useful for behavior priors and long-tail events.

## Split strategy for robust validation

Use three independent evaluations:

- Cross-driver: train drivers A, test unseen drivers B.
- Cross-vehicle: train on vehicle set A, test on unseen vehicle set B.
- Cross-domain: train on internal + one public dataset, test on another public dataset.

## Label mapping to current classes

Map labels to:
- `Forward`
- `In-Car`
- `Non-Forward`
- `Other`

Keep a mapping file per source dataset under `data/manifests/label_maps/`.

## Internal data integration (future)

When your new data arrives (new person + new car):

1. Convert to frame manifest CSV with fields:
- `video`
- `frame_id`
- `timestamp`
- `roi_x1,roi_y1,roi_x2,roi_y2`
- `label`
- `annotator`

2. Run same QA checks on all datasets:
- class distribution
- per-video label continuity
- uncertain frames list

3. Retrain with mixed sampling:
- balance by class and by source domain (internal/public)
- avoid one domain dominating batches

## Current recommendation

Before adding new external data, finish `Other` class stabilization on current pipeline, then lock
an evaluation protocol. This prevents metric drift when new domains are merged.
