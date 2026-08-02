# AutoUI Figure Redraw Brief

This document summarizes the figures currently referenced by `autoui.tex` so a designer can redraw them with a consistent publication style. The figures support a method paper, not the downstream trust/NOA analysis in `sample.tex`.

## Global Requirements

- Target paper: `autoui.tex`, ACM `acmart` two-column WIP format.
- Preserve figure numbers and conceptual roles:
  - Figure 1: workflow overview.
  - Figure 2: evaluation/evidence checkpoints.
  - Figure 3: few-shot adaptation curve.
  - Figure 4: hand-state distillation schematic.
- Use clear, formal academic styling. The current hand-drawn look is only a placeholder.
- Keep all numerical values exactly as listed below unless the paper text is updated.
- Text should remain readable when included at `\linewidth` in a two-column ACM paper.
- Use color consistently:
  - Gaze/AOI branch: blue/green.
  - Hand/wheel branch: orange/teal.
  - Warnings or scope limits: red or muted gray.
- Avoid decorative clutter. The figures should make the method and evidence trail easier to scan.

## Figure 1: Human-in-the-loop Workflow

Current file: `lable_workflow.png`

Referenced in `autoui.tex`:
- Section: `Workflow Design`.
- Figure label: `fig:workflow`.
- Text linkage: lines around the figure state that the workflow starts from target videos and analysis segments, then resolves videos, reviews driver ROIs, builds annotation packs, collects AOI labels, trains participant-specific classifiers, exports deployment models, runs inference, stabilizes predictions, estimates hand-on-wheel states, optionally distills teacher outputs, and aggregates behavior windows.

Core message:
- This figure is the pipeline map for converting naturalistic in-cabin video into reviewable AOI glance and hand-on-wheel variables.
- It should emphasize auditability: ROI provenance, frame semantics, temporal stability, hand-state review, and behavior aggregation are separate checkpoints.

Logical structure:
- Left-to-right process with four major columns:
  1. Inputs.
  2. AOI Models.
  3. Inference.
  4. Windows.
- Within columns, show vertical steps; arrows should show both within-column progression and cross-column progression.

Required content:
- Title or visible header: "Human-in-the-loop workflow".
- Optional subtitle: "reviewable checkpoints from ROI to behavior windows".
- Column headers:
  - `1 Inputs`
  - `2 AOI Models`
  - `3 Inference`
  - `4 Windows`
- Step labels and text:
  - A: Study spreadsheet + videos.
  - B: Driver ROI review.
  - C: Sample extraction.
  - D: Four AOI labels.
  - E: AOI calibration.
  - F: Backbone comparison.
  - G: ONNX parity checks.
  - H: Segment-level gaze inference.
  - I: GroundingDINO hand/wheel evidence.
  - J: Temporal state rules.
  - K: Gaze + hand-state alignment.
  - L: Window-level Behavior Metrics.
  - M: Participant summaries.

Connection to paper:
- Supports the `Workflow Design` section and the caption claim that the method keeps ROI provenance, frame semantics, temporal stability, hand-state review, and behavior aggregation as separate reviewable stages.
- This is the figure a downstream paper can cite when saying AOI and hand-state features were produced by a documented human-in-the-loop annotation workflow.

Redraw notes:
- Consider using a clean flowchart with four vertical swimlanes.
- Make the two branches visible:
  - AOI/gaze branch: ROI -> AOI labels -> calibration -> gaze inference -> temporal rules.
  - Hand/wheel branch: GroundingDINO evidence -> hand-state rules -> alignment.
- If space allows, mark human checkpoints with a small reviewer icon or "review" badge: Driver ROI review, Four AOI labels, Hand/wheel evidence.
- Avoid overloading step L; use "Window-level metrics" if the full phrase is too long.

## Figure 2: Evidence Checkpoints

Current file: `figures/evidence_checkpoints.png`

Referenced in `autoui.tex`:
- Section: `Preliminary Evaluation`.
- Figure label: `fig:evidence_checkpoints`.
- Text linkage: the paragraph before the figure states that the evaluation asks whether the workflow produces reviewable measurement artifacts, not whether a single end-to-end score solves driver monitoring. It also states the evidence is finalized by May 24, 2026, with processing coverage finalized on May 25, 2026 for the 15-participant analysis set.

Core message:
- This figure summarizes the evidence trail across distinct failure points in the measurement workflow.
- The central logic is not "one model is accurate"; it is "each major source of measurement error has a separate checkpoint."

Logical structure:
- Six checkpoint panels/cards arranged in a 2-column by 3-row grid.
- Each panel should include:
  - Checkpoint name.
  - Headline number.
  - One or two supporting details.
- The panels should read as independent but related evidence layers.

Required content and values:
- Header: "Evidence checkpoints".
- Optional subtitle: "artifact trail for the current WIP claim".
- Panel 1: ROI validation.
  - Main value: `15 participants`.
  - Details: `24/24 high-risk reviewed`; `40/40 consistency check`.
  - Interpretation: production ROI set has completed visual audit for the analysis participant set.
- Panel 2: AOI semantics.
  - Main value: `280 LOPO jobs`.
  - Details: `11/14 non-YOLO NI`; `five seeds per split`.
  - Interpretation: non-YOLO family-best comparators are often non-inferior to YOLOv8s-cls under held-out participant testing.
- Panel 3: Few-shot.
  - Main value: `25-200 labels`.
  - Details: `ResNet NI in 8/12`; `three no-leak panels`.
  - Interpretation: label budget and participant adaptation are first-order design variables.
- Panel 4: Temporal rules.
  - Main value: `-84.91% switches`.
  - Details: `279.82 to 42.22/min`; `60 two-context segments`.
  - Interpretation: temporal stabilization materially changes event structure, not just visual smoothness.
- Panel 5: Hand-state review.
  - Main value: `18/24 + 21/21`.
  - Details: `two review passes`; `UNCERTAIN retained`.
  - Interpretation: first pass revealed errors; corrected second pass supports final-state plausibility but still needs broader independent review.
- Panel 6: Deployment.
  - Main value: `280/280 LOPO exports`.
  - Details: `top-1 parity >= .99`; `ONNX check passed`.
  - Interpretation: trained AOI models can be exported and checked for deployment parity.

Connection to paper:
- Supports the whole `Preliminary Evaluation` section.
- Each panel maps to a later subsection:
  - ROI validation -> `ROI and Processing Coverage`.
  - AOI semantics -> `AOI Generalization`.
  - Few-shot -> `Few-Shot Participant Adaptation`.
  - Temporal rules -> `Temporal Stability`.
  - Hand-state review -> `Hand-on-Wheel Review`.
  - Deployment -> `Deployment Checks`.

Redraw notes:
- Use a dashboard-like evidence matrix rather than a decorative infographic.
- Keep headline numbers large; keep details compact.
- Consider adding a small subtitle under the title: "Six separate checks for six possible measurement failures."
- Use consistent iconography only if it remains subtle.

## Figure 3: Few-shot Adaptation Curve

Current file: `figures/fewshot_curve.png`

Referenced in `autoui.tex`:
- Section: `Few-Shot Participant Adaptation`.
- Figure label: `fig:fewshot_curve`.
- Text linkage: the paragraph states that the experiment used three participant-specific adaptation panels; budgets 25, 50, 100, and 200 labels; five seeds per budget; frozen participant-specific test sets; ResNet50 gains rapidly from 25 to 100 labels; YOLOv8s-cls catches up and exceeds ResNet50 at 200 labels; ResNet50 is non-inferior to YOLOv8s-cls in 8/12 participant-budget cells.

Core message:
- Participant-specific label budget changes model performance.
- ResNet50 improves quickly through 100 labels; YOLOv8s-cls is better at 200 labels in the averaged curve.
- The curve motivates participant adaptation as a methodological design variable, not a fixed preprocessing detail.

Logical structure:
- Line chart.
- X-axis: participant-specific label budget.
- Y-axis: macro-F1.
- Two model series:
  - ResNet50.
  - YOLOv8s-cls.
- Annotate each point with its value.
- Include a short interpretive callout: "ResNet gains through 100 labels; YOLO leads at 200".

Required axes:
- X-axis title: `participant-specific labels`.
- X tick values: `25`, `50`, `100`, `200`.
- Y-axis title: `macro-F1`.
- Y tick values currently shown: `0.35`, `0.45`, `0.55`, `0.65`, `0.75`.
- Suggested y-axis range: approximately `0.30` to `0.75`.

Required data points:
- ResNet50:
  - 25 labels: `0.441`.
  - 50 labels: `0.517`.
  - 100 labels: `0.644`.
  - 200 labels: `0.662`.
- YOLOv8s-cls:
  - 25 labels: `0.326`.
  - 50 labels: `0.425`.
  - 100 labels: `0.633`.
  - 200 labels: `0.712`.

Required legend:
- `ResNet50`.
- `YOLOv8s-cls`.

Required caption meaning:
- "Few-shot no-leak adaptation curve over three participant-specific adaptation panels (five seeds per participant-budget cell)."

Connection to paper:
- Supports the claim that label budget and participant adaptation are first-order design variables.
- Provides visual support for the text claim:
  - ResNet50 rises from `0.441` to `0.644` between 25 and 100 labels.
  - YOLOv8s-cls rises from `0.326` to `0.712` between 25 and 200 labels.
  - At 100 labels, the two are close: ResNet50 `0.644`, YOLOv8s-cls `0.633`.
  - At 200 labels, YOLOv8s-cls leads: `0.712` vs. `0.662`.

Redraw notes:
- Use a clean academic line plot with marker points and direct labels.
- Keep `0.662` below the final blue point if direct labels are retained; the earlier hand-drawn version had overlap issues.
- If the chart becomes crowded, use endpoint labels plus a compact data table inset, but do not omit exact values.
- Consider adding a light gray band or annotation for "100 labels" to emphasize where ResNet's rapid gains level off.

## Figure 4: Hand-state Distillation Schematic

Current file: `figures/distillation_schematic.png`

Referenced in `autoui.tex`:
- Section: `Hand/Wheel Distillation Check`.
- Figure label: `fig:distillation_schematic`.
- Text linkage: the preceding paragraph states that two initial GroundingDINO runtime probes, 60 s and 140 s, were converted into 5,000 ROI crops labeled by the teacher pipeline's stable ON/OFF/UNCERTAIN state. A deterministic frame-hash split held out 1,035 validation crops. The revised text also adds an ON-enriched multi-context clean time-block check with 300 validation crops per state, stable/raw agreement, 0.5 s separation from stable-state transitions, and reliable contact evidence for ON crops.
- The following paragraph/table state the independent saved-student check: 1,022/1,035 matches, 0.987 agreement, 13 total mismatches, 6.342 s for 5,000 crops, 1.268 ms per crop, teacher probes required 172.752 s, and 27.2x speedup. The ON-enriched clean time-block check reports 900 balanced held-out crops, 0.923 overall agreement, and ON precision/recall/F1 of 0.969/0.930/0.949. OFF remains the weakest class, so present the student as a support audit, not as a production-ready replacement.

Core message:
- GroundingDINO is used as a slow teacher for selected probes.
- Teacher stable states are distilled into a compact student model.
- The student is an acceleration checkpoint, not a validated production replacement.

Logical structure:
- Four left-to-right stages:
  1. Teacher-labeled probes.
  2. Stable state labels.
  3. Student training.
  4. Held-out check.
- Then a bottom row of three summary boxes:
  - Agreement.
  - Runtime.
  - Boundary/scope limitation.

Required stage content:
- Header: `Hand-state distillation check`.
- Optional subtitle: `GroundingDINO teacher labels compact YOLOv8s-cls student states`.
- Stage 1: `Teacher-labeled probes`.
  - `2 initial probes`.
  - `60 s + 140 s`.
  - `GroundingDINO boxes`.
- Stage 2: `Stable state labels`.
  - `ON / OFF / UNCERTAIN`.
  - `teacher pipeline states`.
  - `5,000 ROI crops`.
- Stage 3: `Student training`.
  - `YOLOv8s-cls`.
  - `balanced clean crops`.
  - `time-block validation`.
- Stage 4: `Held-out check`.
  - `ON support: 300 crops`.
  - `ON P/R: .969 / .930`.
  - `27.2x speedup`.

Required summary boxes:
- Agreement:
  - `0.987 initial`.
  - `0.923 clean held-out`.
- Runtime:
  - `6.342 s / 5,000 crops`.
  - `1.268 ms per crop`.
- Boundary:
  - `support audit`.
  - `teacher-state check`.
  - `not replacement`.
- Bottom note:
  - `Use teacher cost once on selected clips; test student only as an acceleration checkpoint`.

Additional numerical details from text/table that may be added as small notes:
- Validation agreement: `1,022/1,035`.
- ON-enriched clean held-out agreement: `0.923`.
- ON-enriched support: `300` held-out crops for each state.
- ON-enriched ON precision/recall/F1: `0.969 / 0.930 / 0.949`.
- ON-enriched OFF precision/recall/F1: `0.894 / 0.897 / 0.895`.
- ON-enriched UNCERTAIN precision/recall/F1: `0.910 / 0.943 / 0.926`.
- Initial split confusion counts:
  - `531 OFF -> OFF`.
  - `12 ON -> ON`.
  - `479 UNCERTAIN -> UNCERTAIN`.
  - `13 total mismatches`.
- ON-enriched clean held-out confusion counts:
  - `269 OFF -> OFF`.
  - `279 ON -> ON`.
  - `283 UNCERTAIN -> UNCERTAIN`.
  - `69 total mismatches`.
- Teacher wall time for the two probes: `172.752 s`.
- Student runtime: `6.342 s`.
- Speedup: `27.2x`.
- Student: `YOLOv8s-cls` for the ON-enriched check.

Connection to paper:
- Supports the revised distillation paragraph and Table `tab:wheel_distill`.
- Reinforces the limitation that the student is evaluated by teacher-state agreement, so it is an acceleration checkpoint rather than a replacement for full human-labeled production validation.

Redraw notes:
- Use a formal pipeline schematic, not a performance chart.
- Use different visual treatments for teacher, labels, student, and validation:
  - Teacher: detector/box icon or slow model block.
  - Labels: ON/OFF/UNCERTAIN state chips.
  - Student: compact classifier block.
  - Held-out check: checkmark plus runtime/speed.
- Make the limitation visually explicit, e.g. a red or gray "scope limit" box.
- Do not imply the student has been validated against human labels; the agreement is with GroundingDINO teacher-state outputs.

## Figure-to-Text Consistency Checklist

Before finalizing redesigned figures, verify:

- `autoui.tex` still includes all four figure files with the expected labels.
- Figure 1 includes all workflow stages from input videos to participant summaries.
- Figure 2 keeps all six checkpoint values exactly:
  - `15 participants`.
  - `280 LOPO jobs`.
  - `25-200 labels`.
  - `-84.91% switches`.
  - `18/24 + 21/21`.
  - `280/280 LOPO exports`.
- Figure 3 uses the exact eight macro-F1 values listed above.
- Figure 4 clearly says teacher-state agreement, not human-label agreement.
- Captions should remain consistent with the rewritten visuals.
