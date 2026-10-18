# Hyperparameter Tuning Readiness Report

**Written:** 11 October 2026
**Scripts:** `tools/tune_partA_inventory.py`, `tools/tune_partB_audit.py`
**Pre-run hash:** PASS (3,283 files, changed=0, missing=0)
**Post-run hash:** PASS (3,283 files, changed=0, missing=0)

---

## 1. Experiment Inventory (Part A)

### 1.1 Phase 5

| Item | Value | Status |
|---|---|---|
| File | `results/phase5_interim_baseline/checkpoint_4.pth` | **VERIFIED** (603 MB) |
| Description | UA-DETRAC + VisDrone fine-tune of DanceTrack-pretrained MeMOTR, no IOTD data | **VERIFIED** |

### 1.2 Arm 1 — all claims verified

| Claimed item | Verified value | Status |
|---|---|---|
| Config | `configs/train_iotd_real.yaml` | VERIFIED |
| Start weights | `results/phase5_interim_baseline/checkpoint_4.pth` | VERIFIED (PRETRAINED_MODEL L87; RESUME empty) |
| Mode | Weights-only (not resume) | VERIFIED |
| Sequences | IOTD_train_real: seq_01_1min + seq_03_30sec_b | VERIFIED |
| Steps per epoch | 2,528 | VERIFIED (log: Iter=2528/2528) |
| Epochs | 5 | VERIFIED (EPOCHS: 5) |
| Total optimizer steps | 12,640 | VERIFIED |
| LR | 2e-5 epochs 0-3, 2e-6 epoch 4 | VERIFIED (LR_DROP_MILESTONES:[4], rate=0.1) |
| LR backbone | 0.0 (frozen) | VERIFIED (LR_BACKBONE: 0.0) |
| Batch size | 1 | VERIFIED |
| Random seed | 42 | VERIFIED (SEED: 42; train_engine.py L36: set_seed) |
| Clip length | 2 frames (constant throughout all 5 epochs) | VERIFIED (SAMPLE_STEPS:[6,10,14] never triggered; stage=0 always) |
| Measured time | 28 min/epoch | VERIFIED ([Epoch: N, Total Time: 28min] in log) |
| Total wall time | ~140 min (2h 20m) | VERIFIED |
| Seconds per step | 0.664 s/step | VERIFIED (28x60/2528) |

**Independent-seed repeat: NONE** found in results/.

### 1.3 Arm 2 — all claims verified

| Claimed item | Verified value | Status |
|---|---|---|
| Config | `configs/train_iotd_real_aug.yaml` | VERIFIED |
| Sequences | IOTD_train_real + IOTD_train_aug (6 augmented copies) | VERIFIED |
| Steps per epoch | 10,112 | VERIFIED (log: Iter=10112/10112) |
| Total steps | 50,560 | VERIFIED |
| Time/epoch | 112-119 min | VERIFIED (log: epochs 0-4: 119/112/111/111/112 min) |
| Total wall time | ~565 min (~9h 25m) | VERIFIED |
| Seconds per step | 0.665 s/step | VERIFIED |
| Random seed | **42 (same as Arm 1)** | VERIFIED — Arm 1 and Arm 2 used the same seed; they are NOT independent-seed repeats |

### 1.4 Verified metrics (from trackeval_*/pedestrian_summary.txt)

| Run | Steps | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5 | 0 | 50.51 | 43.88 | 58.49 | 29.10 | 68.84 | 1 | 2,117 | 847 |
| arm1 ckpt_3 | 10,112 | 50.47 | 46.96 | 54.51 | 29.63 | 70.23 | 9 | 2,629 | 305 |
| arm1 ckpt_4 | 12,640 | 49.58 | 45.70 | 54.04 | 24.94 | 69.66 | 7 | 2,855 | 277 |
| arm2 ckpt_0 | 10,112 | 40.51 | 36.49 | 45.15 | 16.28 | 56.63 | 10 | 1,964 | 1,527 |
| arm2 ckpt_1 | 20,224 | 48.17 | 45.54 | 51.14 | 29.84 | 68.64 | 12 | 2,432 | 490 |
| arm2 ckpt_4 | 50,560 | 42.84 | 38.32 | 48.01 | -6.58 | 63.72 | 6 | 4,266 | 185 |

> All numbers match on-disk summaries. No contradictions found.

### 1.5 All result folders (verified on disk)

| Folder | Contents | Produced by |
|---|---|---|
| `results/phase5_interim_baseline/` | 5x .pth + 48 tracker .txt files | Phase 5 training |
| `results/train_iotd_real/` | 5x .pth + train/log.txt | Arm 1 training |
| `results/train_iotd_real_aug/` | 5x .pth + train/log.txt | Arm 2 training |
| `results/eval_iotd_val/{phase5,arm1,arm2}/` | 2 .txt each | run_inference_mot.py (thresh=0.5) |
| `results/eval_iotd_val/arm{1,2}_ckpt{0-3}/` | 2 .txt each | Part A diagnostic inference |
| `results/eval_iotd_val/trackeval_*/` | TrackEval outputs | run_trackeval.py |
| `results/eval_iotd_val/diag/thresh_0p7/`, `thresh_0p8/` | thresh=0.7, 0.8 + ROI-filtered | Part C diagnostic |
| `results/eval_iotd_val/diag/roi/` | ROI-filtered outputs | Part B diagnostic |
| `results/eval_iotd_train_audit/phase5/` | seq_01 (4,399 rows), seq_03 (6,898 rows) | Part B1 inference |
| `results/phase6_c2_interim/` | 5x .pth C2 adapter + tracker val | Phase 6 C2 training |

**No independent-seed repeat of any run exists in results/.**

---

## 2. Annotation Audit (Part B)

### B1 — Candidate generator inference (Phase 5 only; unbiased for training seqs)

| Sequence | Frames | Detections |
|---|---:|---:|
| seq_01_1min | 1,599 | 4,399 |
| seq_03_30sec_b | 931 | 6,898 |

**VERIFIED** — `check/logs/tune_B1_inference.txt`

### B2 — Candidate tracks (>=15 frames, >=80% max-IoU < 0.2)

| Sequence | Trackers used | Total tracks | Candidates | Merged unique |
|---|---|---:|---:|---:|
| seq_02_30sec_a | phase5 + arm1 + arm2 | 50 / 69 / 68 | 9 / 14 / 18 | **10** |
| seq_01_1min | phase5 | 35 | 15 | **5** |
| seq_03_30sec_b | phase5 | 34 | 0 | **0** |

### B3 — Triage by motion and position

Majority GT direction: all sequences show clear dominant direction (DOWN for seq_01 and seq_02, UP for seq_03 — all 50/37/54 tracks respectively).

**seq_02_30sec_a candidates:**

| Candidate | Frames | Class | Action | Runs | Dir |
|---|---:|---|---|---:|---|
| seq_02_30sec_a_007 | 76 | LIKELY_MISSING | REVIEW_ADD | 2 | DOWN |
| seq_02_30sec_a_005 | 39 | LIKELY_MISSING | REVIEW_ADD | 3 | DOWN |
| seq_02_30sec_a_006 | 31 | LIKELY_MISSING | REVIEW_ADD | 1 | DOWN |
| seq_02_30sec_a_009 | 30 | LIKELY_MISSING | REVIEW_ADD | 1 | DOWN |
| seq_02_30sec_a_004 | 93 | UNCERTAIN | REVIEW_UNCERTAIN | 1 | UP |
| seq_02_30sec_a_002 | 67 | UNCERTAIN | REVIEW_UNCERTAIN | 3 | UP |
| seq_02_30sec_a_008 | 66 | UNCERTAIN | REVIEW_UNCERTAIN | 2 | UP |
| seq_02_30sec_a_000 | 56 | UNCERTAIN | REVIEW_UNCERTAIN | 1 | UP |
| seq_02_30sec_a_001 | 47 | UNCERTAIN | REVIEW_UNCERTAIN | 1 | DOWN |
| seq_02_30sec_a_003 | 32 | UNCERTAIN | REVIEW_UNCERTAIN | 1 | UP |

**seq_01_1min candidates — all LIKELY_OPPOSITE (opposite-carriageway):**

| Candidate | Frames | Class | Action |
|---|---:|---|---|
| seq_01_1min_001 | 254 | LIKELY_OPPOSITE | REVIEW_EXCLUDE_BY_POLICY |
| seq_01_1min_000 | 154 | LIKELY_OPPOSITE | REVIEW_EXCLUDE_BY_POLICY |
| seq_01_1min_002 | 55 | LIKELY_OPPOSITE | REVIEW_EXCLUDE_BY_POLICY |
| seq_01_1min_004 | 45 | LIKELY_OPPOSITE | REVIEW_EXCLUDE_BY_POLICY |
| seq_01_1min_003 | 30 | LIKELY_OPPOSITE | REVIEW_EXCLUDE_BY_POLICY |

**seq_03_30sec_b:** 0 candidates (dense GT — 55 tracks, 21,547 rows — leaves no unmatched persistent predictions).

### B4 — Boundary analysis

**seq_01_1min:** GT bottom-edge min stable at 706–728 px across all 100-frame windows. All 5 candidates move opposite to GT majority and fall outside the GT lateral span (cx 770–1068 px). **Conclusion: fixed policy boundary enforced by annotation. LIKELY_OPPOSITE vehicles are consistently excluded.**

**seq_02_30sec_a:** GT bottom-edge min varies 267–322 px across windows (GT extends across nearly the full frame height). The 4 LIKELY_MISSING candidates are in the same vertical and lateral range as annotated GT. **Conclusion: incomplete annotation, not a policy boundary.** GT and candidate positions interleave — there is no fixed cutoff distinguishing annotated from unannotated vehicles.

**seq_03_30sec_b:** No candidates.

### B5 — Output files

| File | Path |
|---|---|
| Proposals (15 rows) | `check/annotation_audit/proposals.csv` |
| Summary | `check/annotation_audit/summary.md` |
| Evidence images (seq_02) | `check/annotation_audit/seq_02_30sec_a/cand_*.jpg` |
| Evidence images (seq_01) | `check/annotation_audit/seq_01_1min/cand_*.jpg` |
| B4 overview — seq_02 | `check/annotation_audit/overview_seq_02_30sec_a.jpg` |
| B4 overview — seq_01 | `check/annotation_audit/overview_seq_01_1min.jpg` |
| B4 overview — seq_03 | `check/annotation_audit/overview_seq_03_30sec_b.jpg` |
| CVAT import (4 REVIEW_ADD tracks) | `check/annotation_audit/cvat_import/seq_02_30sec_a_review_add.txt` |

---

## GATE 1 — STOP: awaiting user decision

**B4 conclusion summary:**

| Sequence | LIKELY_MISSING | LIKELY_OPPOSITE | UNCERTAIN |
|---|---:|---:|---:|
| seq_02_30sec_a | **4** | 0 | 6 |
| seq_01_1min | 0 | **5** | 0 |
| seq_03_30sec_b | 0 | 0 | 0 |

The 4 LIKELY_MISSING candidates in seq_02 were detected by 2–3 independent runs, move in the same direction as GT, and lie inside the annotated lane span. This is evidence of incomplete annotation.

**User must reply with one of:**
- **X, policy only** — exclusion rule on tracker output (GT unchanged); continue with Parts C, D, E, F
- **Y, correct annotations** — user re-annotates in CVAT; stop here. All Arm 1 + Arm 2 checkpoints become invalid if training GT is changed. Phase 5 is unaffected. Seed repeats in Part D must not be launched on old labels.
- **Z, both**

---

## 3. Re-scored predictions (Part C)

**Status: NOT RUN — awaiting GATE 1.**

---

## 4. Seed Repeats (Part D)

**Status: configs prepared, NOT LAUNCHED — awaiting GATE 2.**

### 4.1 What set_seed controls

`train_engine.py` L36: `set_seed(config["SEED"])` — seeds `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`. **VERIFIED.**

`torch.backends.cudnn.allow_tf32 = False` (`main.py` L97). `deterministic` and `benchmark` flags NOT set — residual non-determinism possible within a run.

Between runs with different seeds: DataLoader shuffle order, clip-start selection, any random augmentation transforms change. Weight initialisation is unaffected (weights-only load from phase5).

### 4.2 Prepared configs (verified on disk, 7,159 bytes each)

**configs/train_iotd_real_seed1.yaml** — changes vs train_iotd_real.yaml:
```diff
-OUTPUTS_DIR: D:/MajorProject-MeMOTR/results/train_iotd_real
+OUTPUTS_DIR: D:/MajorProject-MeMOTR/results/train_iotd_real_seed1
-SEED: 42
+SEED: 1234
```

**configs/train_iotd_real_seed2.yaml** — changes vs train_iotd_real.yaml:
```diff
-OUTPUTS_DIR: D:/MajorProject-MeMOTR/results/train_iotd_real
+OUTPUTS_DIR: D:/MajorProject-MeMOTR/results/train_iotd_real_seed2
-SEED: 42
+SEED: 2024
```

### 4.3 Cost estimate

| Item | Value |
|---|---|
| Steps per repeat | 12,640 |
| Measured s/step | 0.664 s (VERIFIED) |
| Wall time per repeat | ~140 min (~2h 20m) |
| Disk per run (5 ckpts) | 5 x 603 MB = ~3.0 GB |
| Eval per run | 5 x ~3 min = ~15 min |
| Total for 2 repeats | ~5h training + 30 min eval |

> **GATE 2: do not launch until user replies "launch seeds".**

---

## 5. Step and Scheduler Analysis (Part E)

### 5.1 Epoch and step definition

Source: `train_engine.py` L97–128; `MeMOTR/data/vehiclemot.py` `__len__` and `set_epoch`. **VERIFIED.**

- One **epoch** = `len(dataset_train)` = `len(sample_begin_frames)` gradient steps
- batch_size=1, ACCUMULATION_STEPS=1: each item = 1 optimizer step
- `sample_begin_frames` recomputed by `set_epoch(epoch)` at the start of each epoch

### 5.2 Clip-length curriculum

`SAMPLE_STEPS: [6, 10, 14]` — curriculum stage advances at epochs 6, 10, 14. Training runs 5 epochs (0–4). Therefore `sample_stage=0` throughout and `sample_length = SAMPLE_LENGTHS[0] = 2` **for the entire 5-epoch run, both arms.** VERIFIED (`vehiclemot.py` set_epoch logic).

### 5.3 LR schedule

MultiStep, **epoch-indexed**. `lr_scheduler.step()` once per epoch. Milestone=[4] → LR drops at end of epoch 4 (the last epoch).

- Arm 1: drop after 4 x 2,528 = **10,112 steps**
- Arm 2: drop after 4 x 10,112 = **40,448 steps**

`checkpoint_3` = last pre-drop checkpoint for both arms. `checkpoint_4` = only checkpoint trained at 10x lower LR. VERIFIED from logs.

---

## 6. Fair Comparison Protocol (Part E)

### Protocol

Compare **Arm 1 checkpoint_3** (10,112 steps, LR=2e-5) vs **Arm 2 checkpoint_0** (10,112 steps, LR=2e-5). Same start, same step count, both pre-LR-drop. Evaluation: seq_02_30sec_a, det/track=0.5, miss_tol=30, use_c1=False, use_c2=False, evaluation-only GT.

### Result at equal steps (10,112)

| Run | Steps | HOTA | DetA | MOTA | FP | FN | Recall | Prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arm 1 ckpt_3 | 10,112 | **50.47** | 46.96 | 29.63 | 2,629 | 305 | 92.71 | 59.59 |
| Arm 2 ckpt_0 | 10,112 | **40.51** | 36.49 | 16.28 | 1,964 | 1,527 | 63.49 | 57.48 |

Arm 1 leads by HOTA +9.96. Arm 2 ckpt_0 has far more FN (1,527 vs 305) — the augmented model adapts more slowly to the real distribution.

### Limits

1. Both arms used the same seed (42). Not independent repeats.
2. One night validation sequence, 744 frames — ~10 pt differences are indicative, not statistically significant.
3. Arm 2 ckpt_0 may simply need more steps for the augmented distribution to become useful.
4. The LR drop falls at the same proportional point (80% of total) for both arms, but Arm 2 sees 4x more examples per epoch.

### Is a 20-epoch real-only run still needed?

Not immediately — existing data shows Arm 1 HOTA plateauing at 49–50 in 5 epochs. A 20-epoch Arm 1 run (~9h 20m) would cost the same as Arm 2 and determine whether longer real-only training degrades, plateaus, or improves. Defer until seed variability is measured.

---

## 7. Hyperparameter Tuning Plan (Part F)

### 7.1 Parameter table

| Parameter | Current | Candidates | Retraining? | Cost per setting | Selection criterion |
|---|---|---|---|---|---|
| A. Checkpoint | ckpt_4 (last) | ckpt_0–4 (existing) | No | ~3 min eval | Highest HOTA at thresh=0.5 |
| B. Non-backbone LR | 2e-5 | 1e-5, 5e-5 | Yes | ~2h 20m (Arm 1 only) | Highest HOTA at best ckpt |
| C. Inference threshold | 0.5 | 0.6, 0.7 | No (inference only) | ~3 min per run | Highest HOTA on val |
| D. C1 (PositionMLP) | disabled | enable | Yes (train from scratch) | TBD | BLOCKED — calibration not done |
| D. C2 (ThresholdAdapter) | disabled | enable | No (adapter trained) | ~3 min eval | NEEDS APPROVAL |
| Augmentation strength | v2 fixed | — | — | — | Not prioritised |

### 7.2 A — Checkpoint selection rule

> **Rule:** Evaluate all ckpt_0 through ckpt_last at det/track=0.5 on seq_02_30sec_a. Select the checkpoint with the highest HOTA. Report also the lowest-FP and lowest-FN checkpoints.

| Arm | Best HOTA ckpt | HOTA | Steps |
|---|---|---:|---:|
| Arm 1 | **ckpt_3** | 50.47 | 10,112 |
| Arm 2 | **ckpt_1** | 48.17 | 20,224 |

Neither arm's ckpt_4 is its best. The rule catches this automatically and applies to any future run.

### 7.3 B — Non-backbone LR

Current: 2e-5. `lr_points`=1e-6 (10x lower, VERIFIED from log). Backbone=0.0 (frozen). `lr_query_updater`=2e-5.

Candidates: **1e-5** (safer, lower risk of instability) and **5e-5** (more aggressive). Recommend testing 1e-5 first; with only one validation sequence, a 1–2 pt HOTA gap may be within noise until seed variability is measured.

Cost: ~2h 20m x 2 LR values x 1 arm = **~4h 40m minimum** (Arm 1 only).

### 7.4 C — Inference threshold

Confidence column is constant=1 in tracker outputs (VERIFIED). Every threshold requires fresh inference (~3 min/run).

Existing results (reusable for ckpt_4 only):
- thresh=0.7: arm1 HOTA=48.06, arm2 HOTA=47.46, phase5 HOTA=41.63
- thresh=0.8: arm1 HOTA=40.63, arm2 HOTA=48.14, phase5 HOTA=37.15

Proposed sweep for all best ckpts: **0.5, 0.6, 0.7**.

> Choosing a threshold on seq_02_30sec_a is tuning on the validation sequence. Any improvement is optimistic.

### 7.5 D — C1 (Perspective Memory Decay)

- Code: `modules/position_decay.py` — `PerspectiveZoneEncoder` (analytic) + `PositionMLP` (~1,000 params). **VERIFIED.**
- Zone boundaries: `BOUNDARY_FAR_MID=0.30`, `BOUNDARY_MID_NEAR=0.70` — **PLACEHOLDER constants** (comment: "replace after Phase 4 calibration"). **VERIFIED** (`position_decay.py` L27–28).
- Weighting direction: FAR zone (top of frame) → low lambda (rely on memory); NEAR zone (bottom) → high lambda (trust current observation). Gives **more weight to near-camera, larger boxes**. **VERIFIED** (docstring L59–61).
- **Project report contradiction:** report says "extra weight for distant, small boxes" — code is correct; report is wrong.
- **Status: BLOCKED** — Phase 4 calibration not run; boundaries are placeholders.

### 7.6 D — C2 (Adaptive Threshold Adapter)

- Architecture: 3→16→8→1 MLP + sigmoid scaled to [tau_min, tau_max]. **VERIFIED** (`modules/threshold_adapter.py` L63–76).
- **tau range in code:** [0.3, 0.7] (`TAU_MIN_DEFAULT=0.3`, `TAU_MAX_DEFAULT=0.7`). **One report said [0.2, 0.7] — code is correct.**
- **Training target:** SceneAnalyzer output = [brightness, contrast, density]. The adapter maps scene state to tau. It **cannot learn any altitude effect** — it adapts only to lighting and crowd density.
- **C2 trained:** avg_loss=0.000687, epoch=4, label=`C2_INTERIM_no_IOTD_2026-08-29`. VERIFIED.
- **Status: NEEDS APPROVAL** — no IOTD val evaluation with use_c2=True has been done; target is brightness/contrast/density only.

---

## 8. Run Budget from Measured Step Time

| Run type | Steps | Time | Notes |
|---|---:|---|---|
| Arm 1 (5 epochs) | 12,640 | ~2h 20m | VERIFIED |
| Arm 2 (5 epochs) | 50,560 | ~9h 25m | VERIFIED |
| Seed repeat x2 | 25,280 total | ~4h 40m | |
| LR sweep (1e-5, Arm 1) | 12,640 | ~2h 20m | per value |
| LOO fold (Arm 1) | 12,640 | ~2h 20m | per fold |
| 20-epoch Arm 1 | 50,560 | ~9h 20m | if needed |

Total estimated compute for minimum tuning plan (seed repeats + 1 LR value + threshold sweep): **~10h training + ~1h eval**.

---

## 9. Checkpoint Selection Rule and Validation Limits

**Checkpoint rule** (written before further runs):
Evaluate all saved checkpoints at det/track=0.5 on seq_02_30sec_a. Select the checkpoint with the highest HOTA. Also report lowest-FP and lowest-FN checkpoints as reference.

**Validation limits:**
- One validation sequence (seq_02_30sec_a): 744 frames, 50 tracks, night only (mean luma 35.6)
- No day, dusk, or test sequences
- All metric comparisons are from single-seed, single-sequence evaluations
- Differences of ≤3 HOTA points are likely within run-to-run noise until seed repeats are complete

**Leave-one-out cost estimate:**
3 folds. Per fold: Arm 1 ~2h 20m, Arm 2 ~9h 25m.
Full LOO (Arm 1 only, 3 folds): ~7h. Full LOO (both arms): ~35h.
**Recommendation:** LOO optional for final selected candidate only.

---

## 10. Status Table

| Item | Status | Notes |
|---|---|---|
| Pre-run hash | PASS | 3,283 files, changed=0, missing=0 |
| Post-run hash | PASS | 3,283 files, changed=0, missing=0 |
| Experiment inventory (Part A) | VERIFIED | All claims confirmed; no contradictions found |
| Annotation audit B1-B5 | COMPLETE | seq_02: 4 LIKELY_MISSING; seq_01: 5 LIKELY_OPPOSITE; seq_03: 0 |
| GATE 1 | WAITING | User must choose X / Y / Z |
| Re-score Part C | BLOCKED | Awaiting GATE 1 |
| Seed configs Part D | PREPARED | train_iotd_real_seed1.yaml (1234), train_iotd_real_seed2.yaml (2024) |
| GATE 2 | WAITING | User must say "launch seeds" |
| Step/scheduler analysis (Part E) | COMPLETE | Clip length=2 throughout; LR drop epoch 4; 0.664 s/step |
| Fair comparison protocol | COMPLETE | Arm1 ckpt3 vs Arm2 ckpt0 at 10,112 steps |
| Tuning plan (Part F) | COMPLETE | See Section 7 |
| C1 — PositionMLP | BLOCKED | Phase 4 calibration not run; zone boundaries are placeholders |
| C2 — ThresholdAdapter | NEEDS APPROVAL | Trained; target is brightness/contrast/density only; no altitude learning possible |
| Checkpoint selection rule | WRITTEN | Best HOTA per arm: Arm1=ckpt_3 (50.47), Arm2=ckpt_1 (48.17) |

---

## Required Caveats

1. **One validation sequence** — seq_02_30sec_a, 744 frames, 50 tracks, night only. All HOTA values are single-sequence, single-seed results.
2. **No independent-seed repeats** — run-to-run variability is unknown; differences of ≤3 HOTA points may be within noise.
3. **Augmented frames are training samples, not additional annotated data.** Real annotated training data: **3,274 frames** (seq_01: 1,599 frames + seq_03: 1,675 frames; val seq_02: 744 frames excluded from training). The v2 augmented copies add photometric variation but no new ground-truth annotations.
