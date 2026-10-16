# MeMOTR on IOTD — Tuning Execution Walkthrough

A plain-language, append-only record of how the hyperparameter tuning of MeMOTR on the IOTD vehicle data was done,
what was found and why each decision was made. Written so it can be read without the terminal logs.

**Rules for this file.** Entries are only ever appended. A mistake is fixed by a new dated "Correction" entry that
points to the earlier entry. Every number carries its source file. Labels used:

- **Verified** — read from the named file or TrackEval summary during this review (2026-10-16).
- **Reported from earlier execution** — stated in an earlier Claude Code run's report; source file named but not
  re-read in this review.
- **Inferred** — reasoned from verified facts, not measured.
- **Not yet verified** — no supporting file found.

Times are local, India Standard Time (IST, UTC+05:30).

---

## Part A. Terms used in this record

| Term | Plain meaning |
|---|---|
| **MOT / tracker** | Multi-object tracking: detect every vehicle in every frame and keep the same ID on the same vehicle over time. MeMOTR is a transformer tracker that keeps a memory per tracked vehicle. |
| **Checkpoint** | The model weights saved at a point in training. Here one checkpoint is saved after every epoch: `checkpoint_0` … `checkpoint_4`. |
| **Epoch / step** | One epoch = one pass over the training data. One step = one optimizer update. Arm 1 has 2,528 steps per epoch; Arm 2 has 10,112. |
| **Seed** | The random-number starting value. Same data and settings with a different seed gives a slightly different model; the spread across seeds shows how much of a difference is just luck. |
| **GT (ground truth)** | The hand-made CVAT annotations: the correct box and ID of every vehicle. |
| **TP / FP / FN** | True positive: a predicted box that matches a real vehicle. False positive: a predicted box with no real vehicle (a false alarm). False negative: a real vehicle the tracker missed. |
| **Precision / Recall** | Precision = share of predicted boxes that are correct (TP / (TP+FP)). Recall = share of real vehicles found (TP / (TP+FN)). |
| **DetA** | Detection accuracy: how well boxes are found, ignoring IDs. |
| **AssA** | Association accuracy: how well the same ID stays on the same vehicle over time. |
| **HOTA** | Higher Order Tracking Accuracy: the main score; roughly the geometric mean of DetA and AssA, so it rewards both finding vehicles and keeping their IDs. |
| **MOTA** | An older score: 1 − (FP + FN + ID switches) / number of GT boxes. Can be negative when there are many false positives. |
| **IDF1** | How often a vehicle carries its correct ID, as an F1 score over the whole sequence. |
| **ID switch (IDSW)** | The tracker's ID for one real vehicle changes. Fewer is better. |
| **Detection threshold** | Minimum confidence for a detection to start a new track. |
| **Tracking threshold** | Minimum confidence for an existing track to count as "seen" in this frame. Below it, the track is "missed" for that frame. |
| **Miss tolerance** | How many consecutive missed frames a track may have before its ID is ended. |
| **Output score filter** | Upstream MeMOTR only writes a track's box to the results file if that frame's confidence is above `RESULT_SCORE_THRESH` (0.5) and the box is larger than 100 px². |
| **C1 / C2** | The project's two proposed additions. C1: perspective-aware memory update (PositionMLP). C2: scene-conditioned detection threshold (ThresholdAdapter). |
| **S0 scoring** | Scoring against the original, unmodified GT of the validation sequence (SHA256 `17ebe278…d3472`). |

---

## Part B. Historical chronology (before this review)

The main validation sequence throughout is `seq_02_30sec_a` (744 frames, 50 tracks, night). The two training
sequences are `seq_01_1min` (1,599 frames, night) and `seq_03_30sec_b` (931 frames, dusk, portrait). Real annotated
frames: 3,274; real training frames: 2,530. **Verified** — `archive/pre_tuning_baseline_v1/tables/table06_dataset_summary.csv`.

### B1. Phase 5 baseline
- **What:** MeMOTR (DanceTrack pre-training) fine-tuned on UA-DETRAC + VisDrone vehicles; never saw IOTD.
  Its `checkpoint_4` is the starting point for every IOTD run.
- **Result (old, unfiltered output):** HOTA 50.51, DetA 43.88, AssA 58.49, MOTA 29.10, IDF1 68.84, IDSW 1,
  FP 2,117, FN 847. **Verified** — `archive/pre_tuning_baseline_v1/data/results_registry_v1.csv` (row Phase5 ckpt_4, 0.5).

### B2. Arm 1 (real IOTD frames) and Arm 2 (real + augmented)
- **What changed:** Arm 1 fine-tuned Phase 5 on the 2,530 real training frames; Arm 2 on those plus 7,590
  photometrically augmented copies (exposure gain, gamma, hue/saturation, JPEG quality, noise; 3 copies of each training
  sequence). **Augmented frames are additional training samples, not additional real annotated frames** — their labels
  are copies of the real labels. **Verified** — `tools/augment_photometric_v2.py` header, `data/IOTD_train_aug/` (6 folders,
  3 × 1,599 + 3 × 931 frames).
- **What stayed the same:** backbone frozen (LR 0), LR 2e-5 dropping to 2e-6 for the last epoch, 5 epochs, seed 42.
- **Results per checkpoint (old, unfiltered):** Arm 1 HOTA 45.72 / 49.30 / 48.87 / **50.47** / 49.58; Arm 2
  40.51 / **48.17** / 44.77 / 43.41 / 42.84. By the written rule (highest HOTA of the five checkpoints at threshold
  0.5) Arm 1's best was `checkpoint_3` and Arm 2's best was `checkpoint_1`. At equal steps (10,112) Arm 1 ckpt_3
  beat Arm 2 ckpt_0 by 9.96 HOTA. **Verified** — `results_registry_v1.csv`.
- **Meaning:** Augmentation did not help under that protocol; Arm 1 roughly matched Phase 5 HOTA while finding more
  vehicles (recall 92.71 % vs 79.75 %) but with more false alarms and worse ID keeping (AssA 54.51 vs 58.49).

### B3. Seed repeats (42, 1234, 2024)
- **Why:** one run per setting cannot tell a real improvement from luck.
- **What:** Arm 1's configuration retrained with seeds 1234 and 2024 (Tuning Stage 1, 2026-10-15/16). Seed 42 is the
  original Arm 1.
- **Result (old, unfiltered):** HOTA at checkpoint_3 = 50.47 / 48.39 / 47.96 → mean 48.94, SD 1.34; across the five
  checkpoints the seed SD was 0.60–1.41 and the range up to 2.67 HOTA. The best checkpoint differed per seed
  (ckpt_3, ckpt_2, ckpt_4). **Verified** — `results/tuning_stage1/stage1_results.csv`, `results/tuning_stage1/stage1_report.md`.
- **Meaning:** Arm 1's 50.47 was the luckiest of the three seeds; a difference of 1–2 HOTA is within run-to-run noise.
  This led to the rule: a change counts only if it beats the seed mean by more than two seed SD.

### B4. Training variants `assoc_lr0` and `lr_half`
- **What changed:** `assoc_lr0` froze the query updater (the memory-update module that carries track identity
  between frames; LR 0). `lr_half` halved every non-zero learning rate.
- **Why:** Arm 1 had better detection but worse association than Phase 5; the question was whether the association
  module's drift caused it, or whether smaller updates in general would help.
- **Result (old, unfiltered):** `assoc_lr0` HOTA 50.45 / 40.84 / 46.70 / 47.90 / 46.78; `lr_half` 51.19 / 42.32 / 48.15 /
  44.91 / 45.94. Each beat the seed band only at checkpoint_0; both collapsed at checkpoint_1 (IDSW 43 and 45).
  Neither passed the rule (needs > 1 checkpoint). Freezing the query updater (verified exactly unchanged weights) did
  not restore Phase 5's AssA. **Verified** — `stage1_results.csv`, `stage1_report.md`.
- **Meaning:** both were **negative results** — useful because they rule out two explanations, not successes.

### B5. Original seven-setting inference sweep (Arm 1 ckpt_3)
- **What:** det/track threshold 0.5–0.7 and miss tolerance 20–40, using `tools/run_inference_mot.py`.
- **Result (old, unfiltered):** S_A (0.5/0.5/30) 50.47; S_D (0.5/0.5/20) **54.51**; S_F (0.6/0.6/20) 55.45 but IDSW
  rose. S_D was then "recommended". **Verified** — `results/inference_sweep/sweep_results.csv`.
- **Why it could not be trusted:** while reading the code it was found that the project's output script wrote a box
  for every live track, even tracks that were currently missed and had very low confidence. A shorter miss tolerance
  happened to delete some of those junk boxes. See B6.

### B6. Checking upstream MeMOTR and the missing output filter
- **Finding:** upstream `MeMOTR/submit_engine.py:94-95` filters the output every frame: keep a track only if its
  current-frame score > `RESULT_SCORE_THRESH` (0.5 in `configs/train_iotd_real.yaml:38` and every upstream config) and its
  box area > 100 px². `tools/run_inference_mot.py:184-204` had no such filter (lowest written score was 0.0028).
  **Verified** — code re-read in this review (Entry E003).
- **What this is:** a correction of the implementation to match upstream MeMOTR — **not** a research contribution and
  not a tuned improvement.

### B7. Filtered runner, equivalence check, smoke test
- **What:** `tools/run_inference_mot_filtered.py` = copy of the original with only the two filters added; the original
  file was not changed (SHA256 `7fa7b110…952dc4`).
- **Checks:** with filters switched off it reproduced the original S_A predictions byte for byte; with filters on,
  S_A kept 4,535 of the same 6,506 boxes (all unchanged), removed 1,971 by score and 0 by area.
  **Verified** — `results_after_tuning/logs/00_equivalence_filters_off.txt`, `01_S_A.txt`.

### B8. Corrected seven-setting sweep and corrected seed check
- **Result (corrected):** S_A 64.12, S_B 62.10, S_C 60.19, S_D 63.89, S_E 62.77, S_F 64.25, S_G 59.83.
  Seeds 42/1234/2024 at checkpoint_3: S_A 64.12 / 61.07 / 59.74 (mean 61.64 ± 2.25); S_D 63.89 / 61.09 / 59.73
  (61.57 ± 2.12); S_D − S_A = −0.23 / +0.02 / −0.01. **Verified in this review** — Entry E002.
- **Decision:** keep S_A. S_D's earlier +4 HOTA disappeared once the output was filtered; S_F's 64.25 is one run, +0.12
  over S_A with more ID switches (6 vs 3), far below the corrected bar (66.14).
- **Failure recorded:** the seed 2024 / S_D run crashed once (native exit `0xC0000409`); the retry succeeded. The failed
  attempt is kept at `results_after_tuning/seed_check/seed2024_S_D_failed_attempt1/`.

### B9. C2 (scene-conditioned threshold) — old, unfiltered evaluation
- **Result (old, unfiltered, Phase 5 base):** seq_01 HOTA 27.44 off → 7.71 on (FP 2,518 → 168,975; IDSW 1 → 104);
  seq_02 50.51 → 45.76 (IDSW 1 → 10); seq_03 29.82 → 30.72 (IDSW 2 → 2). The adapter's threshold rose with
  brightness (mean τ 0.311 / 0.332 / 0.404). **Verified** — `results/c2_eval/c2_results.csv`.
- **Status:** historical, unfiltered — must not be mixed with corrected results. The adapter was trained on
  UA-DETRAC/VisDrone, not IOTD.

### B10. Preservation and documentation
- MeMOTR submodule changes stored as patches 004–008 (verified to reproduce the working copy), pushed in commits
  `35597cf`, `4821fbc`; Python version in REPRODUCE.md fixed in `7079b4d`. **Verified** — `git log`.
- Frozen archive `archive/pre_tuning_baseline_v1/` (+ zip) created read-only with a SHA256 manifest; verified
  changed/missing/added = 0. **Verified** — `check/logs/archive_*.txt`; re-verified in Entry E0xx below.
- Note: the archive's numbers are all from the **old, unfiltered** output procedure.

### B11. Outstanding at the start of this review
1. Phase 5, Arm 1, Arm 2 and the Stage 1 variants have never been scored with the corrected output filter, so no
   final model ranking can be claimed yet.
2. C2 has never been evaluated with the corrected filter.
3. C1 has never been calibrated or evaluated.

---

## Part C. Execution entries (this review, 2026-10-16)

### E001 — Inspect existing results and paths — Completed (2026-10-16 14:57:34 +0530 IST)
- **Question:** where are the existing results, which checkpoints exist, which output folder is active?
- **Why:** to reuse verified outputs and avoid overwriting or duplicating anything.
- **What was done:** listed `results_after_tuning/` (exists, active), `results/after_tuning/` (does **not** exist — no
  parallel tree created), `results_prior_to_tuning/` (exists, untouched); listed checkpoints: Phase 5, Arm 1, Arm 2,
  assoc_lr0, lr_half, seed1 (`results/tuning_stage1/seed1_r2/`), seed2 (`results/train_iotd_real_seed2/`) — five
  checkpoints each. Compared `configs/train_iotd_real.yaml` and `configs/train_iotd_real_aug.yaml`: they differ only in
  OUTPUTS_DIR, INTERIM_LABEL and DATASET_DIRS (training data), so inference behaviour is identical.
- **Outputs:** none (read-only). **Status:** Completed.
- **Next:** verify Stage 1 and the earlier corrected numbers from raw files (E002).

### E002 — Verify Stage 1 completion and earlier corrected figures from raw files — Started (2026-10-16 14:57:34 +0530 IST)
- **Question:** did all four Stage 1 jobs really finish with usable evaluations, and are the corrected S_A/S_D and
  seed numbers quoted in the prompt exactly what the saved TrackEval summaries say?
- **Why:** numbers in this review must come from files, not from earlier chat output.
- **What was done:** wrote and ran `tools/final_review_verify.py` (read-only). First attempt **failed** with a script bug
  (the frozen registry names the summary column `source_path`, not `summary_path`); log kept as
  `final_review/logs/E002_verify_failed_attempt1.txt`. Fixed the column lookup and re-ran; valid output is
  `final_review/logs/E002_verify.txt`.
- **Results (Verified):**
  - Stage 1 — seed 1234 (`results/tuning_stage1/seed1_r2/`), seed 2024 (`results/train_iotd_real_seed2/`),
    `assoc_lr0`, `lr_half`: each has 5/5 checkpoints, 5/5 S0 evaluation rows in `stage1_results.csv`, 0 mismatches
    against the TrackEval summaries, status "done" in `status.json`. Backbone unchanged in all four; query updater
    unchanged (max |Δ| = 0) in `assoc_lr0` only. Arm 1 seed 42: 5/5 rows in the frozen registry, 0 mismatches.
  - Seed spread (old unfiltered output, S0, threshold 0.5) per checkpoint: mean ± SD (range) = 46.78 ± 1.41 (2.67),
    48.94 ± 0.83 (1.53), 49.40 ± 0.60 (1.18), 48.94 ± 1.34 (2.51), 48.99 ± 0.99 (1.75).
  - Best checkpoints (old unfiltered output): Arm 1 ckpt_3 50.47; seed1 ckpt_2 50.05; seed2 ckpt_4 49.54; assoc_lr0
    ckpt_0 50.45; lr_half ckpt_0 51.19; Arm 2 ckpt_1 48.17.
  - Corrected figures quoted in the prompt — S_A 64.12, S_D 63.89, S_F 64.25; seeds 42/1234/2024 S_A 64.12/61.07/59.74,
    S_D 63.89/61.09/59.73, diffs −0.23/+0.02/−0.01, means 61.64 ± 2.25 vs 61.57 ± 2.12, mean diff −0.07 — all match the
    TrackEval summaries (0 mismatches in 7 + 6 rows). They are now **Verified**.
- **Meaning:** the Stage 1 experiments are complete and usable; the corrected S_A-vs-S_D evidence is exactly as reported.
- **Comparability:** the Stage 1 and best-checkpoint numbers above are **old unfiltered protocol**; the S_A/S_D/seed
  numbers are **corrected protocol**. They must not be compared with each other.
- **Status:** Completed (after one failed attempt). **Next:** confirm the filter code (E003).

### E003 — Confirm the upstream output filter in code — Completed (2026-10-16 14:58:11 +0530 IST)
- **Question:** exactly what does upstream MeMOTR filter, using which score and which configured value?
- **What was done:** re-read `MeMOTR/submit_engine.py`, `MeMOTR/models/runtime_tracker.py`, `MeMOTR/models/utils.py`,
  `tools/run_inference_mot.py`, `tools/run_inference_mot_filtered.py` (diff in `results_after_tuning/logs/runner_diff.txt`).
- **Result (Verified):** `submit_engine.py:94` `filter_by_score(tracks, thresh=RESULT_SCORE_THRESH)` keeps tracks with
  `max(scores) > thresh` (`:122-125`); `:95` `filter_by_area` keeps area `> 100` px² (`:127-131`). `scores` =
  `sigmoid(logits)` of the current frame (`runtime_tracker.py:61`, `utils.py:171-172`). `RESULT_SCORE_THRESH: 0.5` in
  `configs/train_iotd_real.yaml:38` and `configs/train_iotd_real_aug.yaml` (identical except data keys). The filtered
  runner applies exactly these two filters; the original `tools/run_inference_mot.py` is unchanged
  (SHA256 `7fa7b110…952dc4`).
- **Meaning:** the corrected runner matches upstream MeMOTR's output step; it is an implementation fix, not a tuned
  improvement. **Next:** re-score all candidate models with it (E004).

### E004 — IOTD hash check before inference — Completed (2026-10-16 14:59:34 +0530 IST)
- **Question:** are the original IOTD images and annotations unchanged before new inference starts?
- **What was done:** `tools/sweep_hashcheck.py` against `data/IOTD/check/hashes_before.json`.
- **Result (Verified):** 3,283 files, changed=0, missing=0, added=0. Log: `final_review/logs/E004_hash_before.txt`.

### E005 — Plan and launch the corrected comparison batch — Started (2026-10-16 14:59:34 +0530 IST)
- **Question:** under one consistent corrected procedure, which checkpoint of each model is best, how do the models
  compare, and what does C2 do?
- **Why inference is needed:** none of Phase 5, Arm 2, assoc_lr0, lr_half or Arm 1 checkpoints 0/1/2/4 has ever been
  scored with the corrected output filter, and the best-checkpoint rule needs all five checkpoints of each model under
  the same procedure. No training is involved.
- **Plan (25 inference jobs, no training):** Phase 5 ckpt_4; Arm 1 ckpt_0/1/2/4 (ckpt_3 **reused** from
  `results_after_tuning/sweep/S_A` — same runner, checkpoint, thresholds, GT and TrackEval protocol, verified in E002);
  Arm 2, assoc_lr0, lr_half ckpt_0–4; C2 off/on on all three sequences with Phase 5 ckpt_4 (seq_02 C2-off reuses the
  Phase 5 job, identical settings). Inference configs: Phase 5 / Arm 1 / Stage 1 variants `configs/train_iotd_real.yaml`,
  Arm 2 `configs/train_iotd_real_aug.yaml` (as in the original evaluations; identical for inference).
  GT: `results/eval_iotd_val/gt_eval` for seq_02 model comparison, `results/c2_eval/gt_eval` for C2 (class-1
  evaluation copies; originals unchanged).
- **How:** detached driver `tools/final_review_batch.py` (hidden window) runs `tools/final_review_run.py` per job; each
  job checks for other Python processes and GPU memory first, refuses to overwrite, and the driver appends its own
  Started / Completed / Failed entries below. Before launch: no Python process running, GPU 647 MiB of 24,564 MiB used.

### E006 — Corrected inference `Phase5_ckpt_4` — Started (2026-10-16 14:59:35 +0530)
- **Question:** How does Phase5 checkpoint_4 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\phase5_interim_baseline\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E007 — Inspect C1 implementation and calibration status — Completed (2026-10-16 15:00:20 +0530 IST)
- **Question:** is C1 (perspective-aware memory update) implemented, calibrated and evaluated?
- **What was done (read-only):** read `modules/position_decay.py`, `MeMOTR/models/query_updater.py` (patch 007),
  `tools/calibrate_perspective.py`, `tests/test_modules.py`, `tests/test_zero_regression.py`, `results/teacher_summary.md`;
  searched the project for calibration outputs and for any config or run with C1 enabled.
- **Findings (Verified by code inspection):**
  - Implemented: `PerspectiveZoneEncoder` and `PositionMLP` (λ in [0.002, 0.050]) in `modules/position_decay.py`; wired into
    the query updater behind `use_c1` (config `USE_C1`, default False) — `MeMOTR/models/query_updater.py`; unit tests exist.
  - Not calibrated: zone boundaries are still the placeholders `BOUNDARY_FAR_MID = 0.30`, `BOUNDARY_MID_NEAR = 0.70`
    (`modules/position_decay.py:29-31, 78-80`); `tools/calibrate_perspective.py` exists but no calibration output file was
    found.
  - Not trained or evaluated: every config has `USE_C1: False`; no run with C1 enabled exists; `results/teacher_summary.md:81`
    states the PositionMLP is not trained.
- **Conclusion:** C1 has been implemented (code, config flag and unit tests), but is not yet calibrated, trained or
  evaluated; **no performance-improvement claim is made.** No training was started.

### E008 — Corrected inference `Phase5_ckpt_4` — Completed (2026-10-16 15:01:37 +0530) — follows E006
- **What was done:** ran `tools/final_review_run.py Phase5_ckpt_4` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 57.742 DetA 49.873 AssA 67.304 MOTA 51.291 IDF1 75.18 IDSW 0 FP 940 FN 1097 Prec 76.646 Rec 73.769.
- **Output filter counts:** {'candidates': 5452, 'removed_score': 1427, 'removed_area': 0, 'kept': 4025, 'min_kept_score': 0.5007460117340088}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Phase5_ckpt_4`; log `results_after_tuning\final_review\logs\Phase5_ckpt_4.txt`.

### E009 — Corrected inference `Arm1_ckpt_0` — Started (2026-10-16 15:01:37 +0530)
- **Question:** How does Arm1 checkpoint_0 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real\checkpoint_0.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E010 — Corrected inference `Arm1_ckpt_0` — Completed (2026-10-16 15:03:36 +0530) — follows E009
- **What was done:** ran `tools/final_review_run.py Arm1_ckpt_0` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 58.857 DetA 55.196 AssA 63.196 MOTA 59.732 IDF1 77.151 IDSW 12 FP 908 FN 764 Prec 79.011 Rec 81.731.
- **Output filter counts:** {'candidates': 6427, 'removed_score': 2101, 'removed_area': 0, 'kept': 4326, 'min_kept_score': 0.5000206232070923}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm1_ckpt_0`; log `results_after_tuning\final_review\logs\Arm1_ckpt_0.txt`.

### E011 — Corrected inference `Arm1_ckpt_1` — Started (2026-10-16 15:03:36 +0530)
- **Question:** How does Arm1 checkpoint_1 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real\checkpoint_1.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E012 — Corrected inference `Arm1_ckpt_1` — Completed (2026-10-16 15:05:35 +0530) — follows E011
- **What was done:** ran `tools/final_review_run.py Arm1_ckpt_1` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 62.484 DetA 58.701 AssA 66.958 MOTA 67.241 IDF1 82.47 IDSW 7 FP 718 FN 645 Prec 83.126 Rec 84.577.
- **Output filter counts:** {'candidates': 6268, 'removed_score': 2013, 'removed_area': 0, 'kept': 4255, 'min_kept_score': 0.5000467300415039}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm1_ckpt_1`; log `results_after_tuning\final_review\logs\Arm1_ckpt_1.txt`.

### E013 — Corrected inference `Arm1_ckpt_2` — Started (2026-10-16 15:05:35 +0530)
- **Question:** How does Arm1 checkpoint_2 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real\checkpoint_2.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E014 — Corrected inference `Arm1_ckpt_2` — Completed (2026-10-16 15:07:33 +0530) — follows E013
- **What was done:** ran `tools/final_review_run.py Arm1_ckpt_2` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 62.788 DetA 60.095 AssA 65.935 MOTA 68.675 IDF1 82.572 IDSW 7 FP 839 FN 464 Prec 81.589 Rec 88.905.
- **Output filter counts:** {'candidates': 6462, 'removed_score': 1905, 'removed_area': 0, 'kept': 4557, 'min_kept_score': 0.5000054836273193}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm1_ckpt_2`; log `results_after_tuning\final_review\logs\Arm1_ckpt_2.txt`.

### E015 — Corrected inference `Arm1_ckpt_4` — Started (2026-10-16 15:07:33 +0530)
- **Question:** How does Arm1 checkpoint_4 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E016 — Corrected inference `Arm1_ckpt_4` — Completed (2026-10-16 15:09:32 +0530) — follows E015
- **What was done:** ran `tools/final_review_run.py Arm1_ckpt_4` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 63.545 DetA 60.144 AssA 67.433 MOTA 67.48 IDF1 84.246 IDSW 3 FP 962 FN 395 Prec 79.743 Rec 90.555.
- **Output filter counts:** {'candidates': 6760, 'removed_score': 2011, 'removed_area': 0, 'kept': 4749, 'min_kept_score': 0.5001175403594971}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm1_ckpt_4`; log `results_after_tuning\final_review\logs\Arm1_ckpt_4.txt`.

### E017 — Corrected inference `Arm2_ckpt_0` — Started (2026-10-16 15:09:32 +0530)
- **Question:** How does Arm2 checkpoint_0 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real_aug\checkpoint_0.pth`, config `configs\train_iotd_real_aug.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E018 — Corrected inference `Arm2_ckpt_0` — Completed (2026-10-16 15:11:31 +0530) — follows E017
- **What was done:** ran `tools/final_review_run.py Arm2_ckpt_0` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 52.02 DetA 45.711 AssA 59.356 MOTA 50.167 IDF1 68.37 IDSW 6 FP 379 FN 1699 Prec 86.758 Rec 59.374.
- **Output filter counts:** {'candidates': 4619, 'removed_score': 1757, 'removed_area': 0, 'kept': 2862, 'min_kept_score': 0.500468373298645}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm2_ckpt_0`; log `results_after_tuning\final_review\logs\Arm2_ckpt_0.txt`.

### E019 — Corrected inference `Arm2_ckpt_1` — Started (2026-10-16 15:11:31 +0530)
- **Question:** How does Arm2 checkpoint_1 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real_aug\checkpoint_1.pth`, config `configs\train_iotd_real_aug.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E020 — Corrected inference `Arm2_ckpt_1` — Completed (2026-10-16 15:13:29 +0530) — follows E019
- **What was done:** ran `tools/final_review_run.py Arm2_ckpt_1` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 61.607 DetA 58.097 AssA 65.513 MOTA 67.432 IDF1 82.832 IDSW 3 FP 601 FN 758 Prec 85.068 Rec 81.875.
- **Output filter counts:** {'candidates': 6124, 'removed_score': 2099, 'removed_area': 0, 'kept': 4025, 'min_kept_score': 0.500312089920044}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm2_ckpt_1`; log `results_after_tuning\final_review\logs\Arm2_ckpt_1.txt`.

### E021 — Corrected inference `Arm2_ckpt_2` — Started (2026-10-16 15:13:29 +0530)
- **Question:** How does Arm2 checkpoint_2 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real_aug\checkpoint_2.pth`, config `configs\train_iotd_real_aug.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E022 — Corrected inference `Arm2_ckpt_2` — Completed (2026-10-16 15:15:29 +0530) — follows E021
- **What was done:** ran `tools/final_review_run.py Arm2_ckpt_2` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 54.943 DetA 51.384 AssA 58.941 MOTA 47.8 IDF1 77.987 IDSW 1 FP 1897 FN 285 Prec 67.259 Rec 93.185.
- **Output filter counts:** {'candidates': 7697, 'removed_score': 1903, 'removed_area': 0, 'kept': 5794, 'min_kept_score': 0.5002169609069824}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm2_ckpt_2`; log `results_after_tuning\final_review\logs\Arm2_ckpt_2.txt`.

### E023 — Corrected inference `Arm2_ckpt_3` — Started (2026-10-16 15:15:29 +0530)
- **Question:** How does Arm2 checkpoint_3 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real_aug\checkpoint_3.pth`, config `configs\train_iotd_real_aug.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E024 — Corrected inference `Arm2_ckpt_3` — Completed (2026-10-16 15:17:29 +0530) — follows E023
- **What was done:** ran `tools/final_review_run.py Arm2_ckpt_3` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 54.01 DetA 49.782 AssA 58.748 MOTA 41.057 IDF1 76.064 IDSW 1 FP 2268 FN 196 Prec 63.735 Rec 95.313.
- **Output filter counts:** {'candidates': 8287, 'removed_score': 2033, 'removed_area': 0, 'kept': 6254, 'min_kept_score': 0.5000201463699341}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm2_ckpt_3`; log `results_after_tuning\final_review\logs\Arm2_ckpt_3.txt`.

### E025 — Corrected inference `Arm2_ckpt_4` — Started (2026-10-16 15:17:29 +0530)
- **Question:** How does Arm2 checkpoint_4 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\train_iotd_real_aug\checkpoint_4.pth`, config `configs\train_iotd_real_aug.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E026 — Corrected inference `Arm2_ckpt_4` — Completed (2026-10-16 15:19:39 +0530) — follows E025
- **What was done:** ran `tools/final_review_run.py Arm2_ckpt_4` (attempt 2); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 53.099 DetA 48.528 AssA 58.251 MOTA 38.761 IDF1 75.222 IDSW 1 FP 2276 FN 284 Prec 63.136 Rec 93.209.
- **Output filter counts:** {'candidates': 8263, 'removed_score': 2089, 'removed_area': 0, 'kept': 6174, 'min_kept_score': 0.5000259876251221}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\Arm2_ckpt_4`; log `results_after_tuning\final_review\logs\Arm2_ckpt_4.txt`.
- **Deviation:** first attempt failed; failed output kept as `results_after_tuning\final_review\runs\Arm2_ckpt_4_failed_attempt1`.

### E027 — Corrected inference `assoc_lr0_ckpt_0` — Started (2026-10-16 15:19:39 +0530)
- **Question:** How does assoc_lr0 checkpoint_0 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\assoc_lr0\checkpoint_0.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E028 — Corrected inference `assoc_lr0_ckpt_0` — Completed (2026-10-16 15:21:38 +0530) — follows E027
- **What was done:** ran `tools/final_review_run.py assoc_lr0_ckpt_0` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 62.763 DetA 59.374 AssA 66.62 MOTA 64.275 IDF1 82.467 IDSW 5 FP 1134 FN 355 Prec 77.142 Rec 91.511.
- **Output filter counts:** {'candidates': 6711, 'removed_score': 1750, 'removed_area': 0, 'kept': 4961, 'min_kept_score': 0.5001639127731323}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\assoc_lr0_ckpt_0`; log `results_after_tuning\final_review\logs\assoc_lr0_ckpt_0.txt`.

### E029 — Corrected inference `assoc_lr0_ckpt_1` — Started (2026-10-16 15:21:38 +0530)
- **Question:** How does assoc_lr0 checkpoint_1 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\assoc_lr0\checkpoint_1.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E030 — Corrected inference `assoc_lr0_ckpt_1` — Completed (2026-10-16 15:23:37 +0530) — follows E029
- **What was done:** ran `tools/final_review_run.py assoc_lr0_ckpt_1` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 49.371 DetA 42.267 AssA 58.041 MOTA 46.581 IDF1 62.785 IDSW 21 FP 282 FN 1931 Prec 88.867 Rec 53.826.
- **Output filter counts:** {'candidates': 4915, 'removed_score': 2382, 'removed_area': 0, 'kept': 2533, 'min_kept_score': 0.5002040863037109}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\assoc_lr0_ckpt_1`; log `results_after_tuning\final_review\logs\assoc_lr0_ckpt_1.txt`.

### E031 — Corrected inference `assoc_lr0_ckpt_2` — Started (2026-10-16 15:23:37 +0530)
- **Question:** How does assoc_lr0 checkpoint_2 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\assoc_lr0\checkpoint_2.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E032 — Corrected inference `assoc_lr0_ckpt_2` — Completed (2026-10-16 15:25:36 +0530) — follows E031
- **What was done:** ran `tools/final_review_run.py assoc_lr0_ckpt_2` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 59.177 DetA 56.158 AssA 62.749 MOTA 59.35 IDF1 79.017 IDSW 8 FP 1282 FN 410 Prec 74.634 Rec 90.196.
- **Output filter counts:** {'candidates': 6997, 'removed_score': 1943, 'removed_area': 0, 'kept': 5054, 'min_kept_score': 0.50008225440979}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\assoc_lr0_ckpt_2`; log `results_after_tuning\final_review\logs\assoc_lr0_ckpt_2.txt`.

### E033 — Corrected inference `assoc_lr0_ckpt_3` — Started (2026-10-16 15:25:36 +0530)
- **Question:** How does assoc_lr0 checkpoint_3 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\assoc_lr0\checkpoint_3.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E034 — Corrected inference `assoc_lr0_ckpt_3` — Completed (2026-10-16 15:27:36 +0530) — follows E033
- **What was done:** ran `tools/final_review_run.py assoc_lr0_ckpt_3` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 60.378 DetA 55.275 AssA 66.266 MOTA 55.045 IDF1 79.428 IDSW 4 FP 1510 FN 366 Prec 71.649 Rec 91.248.
- **Output filter counts:** {'candidates': 7339, 'removed_score': 2013, 'removed_area': 0, 'kept': 5326, 'min_kept_score': 0.5000243186950684}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\assoc_lr0_ckpt_3`; log `results_after_tuning\final_review\logs\assoc_lr0_ckpt_3.txt`.

### E035 — Corrected inference `assoc_lr0_ckpt_4` — Started (2026-10-16 15:27:36 +0530)
- **Question:** How does assoc_lr0 checkpoint_4 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\assoc_lr0\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E036 — Corrected inference `assoc_lr0_ckpt_4` — Completed (2026-10-16 15:29:35 +0530) — follows E035
- **What was done:** ran `tools/final_review_run.py assoc_lr0_ckpt_4` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 58.599 DetA 55.112 AssA 62.632 MOTA 55.835 IDF1 78.064 IDSW 6 FP 1450 FN 391 Prec 72.334 Rec 90.65.
- **Output filter counts:** {'candidates': 7288, 'removed_score': 2047, 'removed_area': 0, 'kept': 5241, 'min_kept_score': 0.5001055002212524}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\assoc_lr0_ckpt_4`; log `results_after_tuning\final_review\logs\assoc_lr0_ckpt_4.txt`.

### E037 — Corrected inference `lr_half_ckpt_0` — Started (2026-10-16 15:29:35 +0530)
- **Question:** How does lr_half checkpoint_0 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\lr_half\checkpoint_0.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E038 — Corrected inference `lr_half_ckpt_0` — Completed (2026-10-16 15:31:34 +0530) — follows E037
- **What was done:** ran `tools/final_review_run.py lr_half_ckpt_0` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 64.258 DetA 59.954 AssA 69.23 MOTA 69.201 IDF1 83.675 IDSW 4 FP 607 FN 677 Prec 85.238 Rec 83.812.
- **Output filter counts:** {'candidates': 5933, 'removed_score': 1821, 'removed_area': 0, 'kept': 4112, 'min_kept_score': 0.5000892877578735}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\lr_half_ckpt_0`; log `results_after_tuning\final_review\logs\lr_half_ckpt_0.txt`.

### E039 — Corrected inference `lr_half_ckpt_1` — Started (2026-10-16 15:31:34 +0530)
- **Question:** How does lr_half checkpoint_1 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\lr_half\checkpoint_1.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E040 — Corrected inference `lr_half_ckpt_1` — Completed (2026-10-16 15:33:34 +0530) — follows E039
- **What was done:** ran `tools/final_review_run.py lr_half_ckpt_1` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 51.647 DetA 46.968 AssA 57.144 MOTA 52.224 IDF1 67.821 IDSW 21 FP 388 FN 1589 Prec 86.984 Rec 62.004.
- **Output filter counts:** {'candidates': 5550, 'removed_score': 2569, 'removed_area': 0, 'kept': 2981, 'min_kept_score': 0.5002235174179077}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\lr_half_ckpt_1`; log `results_after_tuning\final_review\logs\lr_half_ckpt_1.txt`.

### E041 — Corrected inference `lr_half_ckpt_2` — Started (2026-10-16 15:33:34 +0530)
- **Question:** How does lr_half checkpoint_2 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\lr_half\checkpoint_2.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E042 — Corrected inference `lr_half_ckpt_2` — Completed (2026-10-16 15:35:33 +0530) — follows E041
- **What was done:** ran `tools/final_review_run.py lr_half_ckpt_2` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 60.985 DetA 56.402 AssA 66.281 MOTA 65.662 IDF1 80.865 IDSW 7 FP 601 FN 828 Prec 84.804 Rec 80.201.
- **Output filter counts:** {'candidates': 5788, 'removed_score': 1833, 'removed_area': 0, 'kept': 3955, 'min_kept_score': 0.5001596212387085}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\lr_half_ckpt_2`; log `results_after_tuning\final_review\logs\lr_half_ckpt_2.txt`.

### E043 — Corrected inference `lr_half_ckpt_3` — Started (2026-10-16 15:35:33 +0530)
- **Question:** How does lr_half checkpoint_3 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\lr_half\checkpoint_3.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E044 — Corrected inference `lr_half_ckpt_3` — Completed (2026-10-16 15:37:31 +0530) — follows E043
- **What was done:** ran `tools/final_review_run.py lr_half_ckpt_3` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 56.148 DetA 51.372 AssA 61.698 MOTA 57.078 IDF1 73.629 IDSW 13 FP 539 FN 1243 Prec 84.503 Rec 70.277.
- **Output filter counts:** {'candidates': 5713, 'removed_score': 2235, 'removed_area': 0, 'kept': 3478, 'min_kept_score': 0.5002777576446533}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\lr_half_ckpt_3`; log `results_after_tuning\final_review\logs\lr_half_ckpt_3.txt`.

### E045 — Corrected inference `lr_half_ckpt_4` — Started (2026-10-16 15:37:31 +0530)
- **Question:** How does lr_half checkpoint_4 score on seq_02_30sec_a under the corrected output filter?
- **Why:** Every model must be scored with the same corrected procedure before a best checkpoint or a best model can be named; the old scores were from unfiltered output.
- **Inputs:** checkpoint `results\tuning_stage1\lr_half\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\eval_iotd_val\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E046 — Corrected inference `lr_half_ckpt_4` — Completed (2026-10-16 15:39:31 +0530) — follows E045
- **What was done:** ran `tools/final_review_run.py lr_half_ckpt_4` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 58.978 DetA 53.589 AssA 65.223 MOTA 60.45 IDF1 77.131 IDSW 8 FP 594 FN 1052 Prec 84.049 Rec 74.845.
- **Output filter counts:** {'candidates': 5932, 'removed_score': 2208, 'removed_area': 0, 'kept': 3724, 'min_kept_score': 0.5002797842025757}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\lr_half_ckpt_4`; log `results_after_tuning\final_review\logs\lr_half_ckpt_4.txt`.

### E047 — Corrected inference `C2_seq_01_1min_off` — Started (2026-10-16 15:39:31 +0530)
- **Question:** With the corrected output filter, does C2 (scene-conditioned detection threshold) change tracking on seq_01_1min when C2 is off?
- **Why:** Earlier C2 numbers were from the unfiltered output and cannot be compared with corrected results.
- **Inputs:** checkpoint `results\phase5_interim_baseline\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_01_1min`, GT `results\c2_eval\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E048 — Corrected inference `C2_seq_01_1min_off` — Completed (2026-10-16 15:43:34 +0530) — follows E047
- **What was done:** ran `tools/final_review_run.py C2_seq_01_1min_off` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 28.867 DetA 16.689 AssA 50.022 MOTA 3.3033 IDF1 34.656 IDSW 0 FP 1445 FN 4790 Prec 53.432 Rec 25.713.
- **Output filter counts:** {'candidates': 4399, 'removed_score': 1296, 'removed_area': 0, 'kept': 3103, 'min_kept_score': 0.5000004768371582}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\C2_seq_01_1min_off`; log `results_after_tuning\final_review\logs\C2_seq_01_1min_off.txt`.

### E049 — Corrected inference `C2_seq_01_1min_on` — Started (2026-10-16 15:43:34 +0530)
- **Question:** With the corrected output filter, does C2 (scene-conditioned detection threshold) change tracking on seq_01_1min when C2 is on?
- **Why:** Earlier C2 numbers were from the unfiltered output and cannot be compared with corrected results.
- **Inputs:** checkpoint `results\phase5_interim_baseline\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_01_1min`, GT `results\c2_eval\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 on, C1 off, TrackEval DO_PREPROC=False.

### E050 — Corrected inference `C2_seq_01_1min_on` — Completed (2026-10-16 15:48:56 +0530) — follows E049
- **What was done:** ran `tools/final_review_run.py C2_seq_01_1min_on` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 27.816 DetA 14.6 AssA 53.172 MOTA -41.331 IDF1 29.565 IDSW 139 FP 4611 FN 4363 Prec 31.138 Rec 32.336.
- **Output filter counts:** {'candidates': 172346, 'removed_score': 165650, 'removed_area': 0, 'kept': 6696, 'min_kept_score': 0.5000070333480835}
- **C2 threshold:** tau over 1599 frames: mean=0.3106 min=0.3068 max=0.3201
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\C2_seq_01_1min_on`; log `results_after_tuning\final_review\logs\C2_seq_01_1min_on.txt`.

### E051 — Corrected inference `C2_seq_02_30sec_a_on` — Started (2026-10-16 15:48:57 +0530)
- **Question:** With the corrected output filter, does C2 (scene-conditioned detection threshold) change tracking on seq_02_30sec_a when C2 is on?
- **Why:** Earlier C2 numbers were from the unfiltered output and cannot be compared with corrected results.
- **Inputs:** checkpoint `results\phase5_interim_baseline\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_02_30sec_a`, GT `results\c2_eval\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 on, C1 off, TrackEval DO_PREPROC=False.

### E052 — Corrected inference `C2_seq_02_30sec_a_on` — Completed (2026-10-16 15:51:11 +0530) — follows E051
- **What was done:** ran `tools/final_review_run.py C2_seq_02_30sec_a_on` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 56.419 DetA 49.168 AssA 65.341 MOTA 46.365 IDF1 73.835 IDSW 7 FP 1442 FN 794 Prec 70.145 Rec 81.014.
- **Output filter counts:** {'candidates': 7654, 'removed_score': 2824, 'removed_area': 0, 'kept': 4830, 'min_kept_score': 0.5004688501358032}
- **C2 threshold:** tau over 744 frames: mean=0.3324 min=0.3271 max=0.3399
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\C2_seq_02_30sec_a_on`; log `results_after_tuning\final_review\logs\C2_seq_02_30sec_a_on.txt`.

### E053 — Corrected inference `C2_seq_03_30sec_b_off` — Started (2026-10-16 15:51:11 +0530)
- **Question:** With the corrected output filter, does C2 (scene-conditioned detection threshold) change tracking on seq_03_30sec_b when C2 is off?
- **Why:** Earlier C2 numbers were from the unfiltered output and cannot be compared with corrected results.
- **Inputs:** checkpoint `results\phase5_interim_baseline\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_03_30sec_b`, GT `results\c2_eval\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 off, C1 off, TrackEval DO_PREPROC=False.

### E054 — Corrected inference `C2_seq_03_30sec_b_off` — Completed (2026-10-16 15:53:39 +0530) — follows E053
- **What was done:** ran `tools/final_review_run.py C2_seq_03_30sec_b_off` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 26.247 DetA 18.385 AssA 37.536 MOTA 23.386 IDF1 37.537 IDSW 2 FP 402 FN 16104 Prec 93.122 Rec 25.261.
- **Output filter counts:** {'candidates': 6898, 'removed_score': 1053, 'removed_area': 0, 'kept': 5845, 'min_kept_score': 0.5000920295715332}
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\C2_seq_03_30sec_b_off`; log `results_after_tuning\final_review\logs\C2_seq_03_30sec_b_off.txt`.

### E055 — Corrected inference `C2_seq_03_30sec_b_on` — Started (2026-10-16 15:53:39 +0530)
- **Question:** With the corrected output filter, does C2 (scene-conditioned detection threshold) change tracking on seq_03_30sec_b when C2 is on?
- **Why:** Earlier C2 numbers were from the unfiltered output and cannot be compared with corrected results.
- **Inputs:** checkpoint `results\phase5_interim_baseline\checkpoint_4.pth`, config `configs\train_iotd_real.yaml`, sequence `seq_03_30sec_b`, GT `results\c2_eval\gt_eval` (class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and area > 100 px (`tools/run_inference_mot_filtered.py`), C2 on, C1 off, TrackEval DO_PREPROC=False.

### E056 — Corrected inference `C2_seq_03_30sec_b_on` — Completed (2026-10-16 15:56:38 +0530) — follows E055
- **What was done:** ran `tools/final_review_run.py C2_seq_03_30sec_b_on` (attempt 1); inference with the filtered runner, then TrackEval.
- **Result (Verified, TrackEval summary):** HOTA 27.038 DetA 19.242 AssA 38.082 MOTA 24.635 IDF1 39.188 IDSW 2 FP 420 FN 15817 Prec 93.171 Rec 26.593.
- **Output filter counts:** {'candidates': 7345, 'removed_score': 1195, 'removed_area': 0, 'kept': 6150, 'min_kept_score': 0.5000083446502686}
- **C2 threshold:** tau over 931 frames: mean=0.4045 min=0.3551 max=0.4518
- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).
- **Outputs:** `results_after_tuning\final_review\runs\C2_seq_03_30sec_b_on`; log `results_after_tuning\final_review\logs\C2_seq_03_30sec_b_on.txt`.

### E057 — Corrected comparison batch finished; integrity checks after inference — Completed (2026-10-16 15:59:52 +0530 IST) — closes E005
- **What happened:** all 25 jobs completed (`final_review/logs/batch_status.txt`). One failure: `Arm2_ckpt_4` attempt 1
  crashed natively (exit `0xC0000409`, no Python traceback) after 11 s; the automatic retry succeeded. The failed output is
  kept at `final_review/runs/Arm2_ckpt_4_failed_attempt1/` and is **not** used; the valid output is `final_review/runs/Arm2_ckpt_4/`.
- **IOTD hash check after inference (Verified):** 3,283 files, changed=0, missing=0, added=0 (`final_review/logs/hash_after.txt`).
- **Frozen archive (Verified):** 55 archive files and 66 source files unchanged, zip read-only (`final_review/logs/archive_verify.txt`).

### E058 — Build tables, dashboard and manifest — Completed (2026-10-16 15:59:52 +0530 IST)
- **What was done:** `tools/final_review_build.py` (read-only on inputs, refuses to overwrite) → `final_review/corrected_model_comparison.csv`
  (21 rows), `historical_vs_corrected.csv`, `corrected_c2_comparison.csv` (6 rows), `historical_c2_unfiltered.csv`,
  `selected_settings.csv`, `final_dashboard.png`, `run_manifest.json`. Log: `final_review/logs/build.txt`.
- **Key results (Verified, corrected procedure, seq_02, det/track 0.5, miss 30):** best checkpoint per model — lr_half ckpt_0
  64.26, Arm 1 ckpt_3 64.12, assoc_lr0 ckpt_0 62.76, Arm 2 ckpt_1 61.61, Phase 5 ckpt_4 57.74. C2 off → on HOTA: seq_01
  28.87 → 27.82, seq_02 57.74 → 56.42, seq_03 26.25 → 27.04; IDSW seq_01 0 → 139, seq_02 0 → 7.
- **What it means:**
  - **What did we change?** Only the output filter (to match upstream), applied to every model.
  - **What stayed the same?** Weights, data, GT, thresholds, TrackEval.
  - **What did the results show?** Fine-tuning on IOTD clearly beats Phase 5 (+6.38 HOTA; all three Arm 1 seeds above Phase 5);
    augmentation and the two learning-rate variants give no reliable gain; the old "AssA loss after fine-tuning" disappears
    (68.21 vs 67.30); C2 hurts the night sequences.
  - **Why keep / reject?** Arm 1 ckpt_3 kept (standard configuration, seed-checked); lr_half's +0.14 is one unstable run.
  - **Limitation:** one validation sequence; one run per row except Arm 1 ckpt_3.
- **Comparability:** corrected rows are comparable with each other only; historical rows are shown beside them with a warning.

### E059 — Write final documents and check every reported number — Completed (2026-10-16 15:59:52 +0530 IST)
- **What was done:** wrote `final_review/selected_settings_and_rationale.md`, `final_review_report.md`, `review_talking_points.md`.
  Cross-checked all 270 values in the report's two main tables against the CSVs: 0 mismatches (`final_review/logs/qc_numbers.txt`).
  Viewed `final_dashboard.png` (opens, not blank, corrected/historical labelled).
- **Decision recorded:** Arm 1 checkpoint_3; det 0.5, track 0.5, miss 30; filtered runner (score > 0.5, area > 100 px);
  C1 and C2 disabled.
- **Not pushed to GitHub** (rule 11): nothing has been committed; approval is needed first.

### E060 — Correction / catch-up: record failed attempts as their own rows — Completed (2026-10-16 16:00:08 +0530 IST)
- **Why:** the two failed attempts were described inside their Completed entries (E002 and E026) but had no separate
  `Failed` row in `execution_log.csv`. They are added now as rows E060a and E060b; nothing earlier was edited.
  - E060a — E002 verification, attempt 1 (2026-10-16 ~14:58): **Failed** — `KeyError: 'summary_path'` (registry column is
    `source_path`); evidence `final_review/logs/E002_verify_failed_attempt1.txt`; superseded by the valid run in E002.
  - E060b — `Arm2_ckpt_4` inference, attempt 1 (2026-10-16 15:17): **Failed** — native crash exit `0xC0000409` at the start of
    inference; partial output `final_review/runs/Arm2_ckpt_4_failed_attempt1/` (not used); superseded by E026.
