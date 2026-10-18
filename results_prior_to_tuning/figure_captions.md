# Figure Captions

All figures: single validation sequence seq_02_30sec_a (744 frames, 50 tracks, night).
Single run per setting, seed 42. Thresholds 0.5 (except where noted). Not statistically established.

**fig01_hota_vs_steps.** HOTA (%) against optimizer steps for Arm 1 (real-only fine-tuning, blue circles)
and Arm 2 (real + photometric augmentation, vermilion triangles), with Phase 5 baseline shown as a grey
dashed horizontal reference (HOTA 50.51). Stars mark the best-HOTA checkpoint per arm (Arm 1 ckpt_3
at step 10,112; Arm 2 ckpt_1 at step 20,224). Dotted vertical lines indicate the learning-rate drop
for each arm (step 10,112 for Arm 1; step 40,448 for Arm 2). The double-headed arrow connects the
equal-step pair Arm 1 ckpt_3 and Arm 2 ckpt_0, both at step 10,112.

**fig02_deta_assa_vs_steps.** DetA (top) and AssA (bottom) against optimizer steps for Arm 1 and Arm 2,
with Phase 5 reference lines. Both metrics follow a non-monotonic trajectory; AssA does not recover
to the Phase 5 level for either arm.

**fig03_fp_fn_vs_steps.** False positives (top-left), false negatives (top-right), recall (bottom-left),
and precision (bottom-right) against optimizer steps for Arm 1 and Arm 2, with Phase 5 reference lines.
FN decreases monotonically for both arms; FP decreases for Arm 1 in early epochs then rises, and
increases monotonically for Arm 2. Source: TrackEval CLEAR counts.

**fig04_detection_outcomes.** Per-frame detection-outcome summary for Phase 5 (grey), Arm 1 ckpt_3 (blue),
and Arm 2 ckpt_1 (vermilion). Each panel shows TP, FP, FN, and a greyed TN cell (labelled
"TN not defined — single-class detection"). Precision and recall are derived from TP, FP, FN.
Source: per-frame greedy IoU>=0.5 matching, which agrees exactly with TrackEval CLEAR counts for these runs.

**fig05_candidate_comparison.** Grouped bar chart of HOTA, DetA, AssA, and IDF1 (left panel) and
MOTA (right panel, separate axis to accommodate negative values) for the three candidates: Phase 5 ckpt_4,
Arm 1 ckpt_3, and Arm 2 ckpt_1. Actual values are labelled on each bar.

**fig06_identity_metrics.** AssA, IDF1, and ID-switch count (IDSW) for the three candidates. Gold edges
mark the best value per metric. The best-HOTA checkpoint does not coincide with the lowest-IDSW checkpoint
for Arm 1 (ckpt_3 IDSW=9; ckpt_4 IDSW=7).

**fig07_epochwise_summary.** HOTA by completed epoch for Arm 1 (blue) and Arm 2 (vermilion) as grouped bars,
with Phase 5 reference. Gold-edged bars are the best-HOTA checkpoint per arm. The table below the bars
lists DetA, AssA, MOTA, and IDF1 for each epoch.

**fig08_qualitative.** Sample frames 298 and 447 of seq_02_30sec_a for Phase 5, Arm 1 ckpt_3, and
Arm 2 ckpt_1. GT boxes in green; tracker boxes in the arm's assigned colour. No region polygon applied.
