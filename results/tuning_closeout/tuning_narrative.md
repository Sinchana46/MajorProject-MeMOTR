# Hyperparameter tuning — technical narrative

17 Oct 2026. Every number in this document uses the **corrected protocol** (upstream output filter) unless it is marked
**HISTORICAL**. Each number is **verified** against its TrackEval summary file by
`results/tuning_closeout/scripts/build_closeout.py` (970 values, 0 mismatches; `check/logs/closeout_01_build.txt`)
unless marked **assumed**. Interpretation is kept in separate "Interpretation" lines.

Tables: `results/tuning_closeout/tables/`. Figures: `results/tuning_closeout/figures/`.

---

## 3.1 Overview

The goal of this phase was to fine-tune MeMOTR on the IOTD vehicle data for the best achievable tracking quality on
the validation sequence `seq_02_30sec_a` (744 frames, 50 tracks, night), without changing the architecture: the
ResNet-50 backbone stays frozen (learning rate 0), and both proposed contributions, C1 (perspective-aware memory
decay) and C2 (scene-conditioned detection threshold), are off for every training run. Starting from the Phase 5
vehicle model (UA-DETRAC + VisDrone, checkpoint_4), two training arms, three seeds, two learning-rate variants and seven
inference settings were evaluated. The outcome is **Arm 1 checkpoint_3 at corrected HOTA 64.12** (seed 42), selected as
the baseline for the later C2/C1 evaluation (`table_corrected_all_runs`).

## 3.2 Protocol correction (most important finding)

**Measurement.** `tools/run_inference_mot.py` wrote every live track to the output, including tracks the model had
currently lost (lowest written confidence 0.0028). Upstream MeMOTR filters the output each frame
(`MeMOTR/submit_engine.py:94-95`): a track is written only if its current-frame score is greater than
`RESULT_SCORE_THRESH` (0.5, `configs/train_iotd_real.yaml:38`) and its box area is greater than 100 px². The corrected
runner `tools/run_inference_mot_filtered.py` adds only these two filters. Checks (verified,
`results_after_tuning/logs/00_equivalence_filters_off.txt`, `01_S_A.txt`): with both filters off it reproduces the
original S_A predictions byte for byte (SHA256 `46c4061d…`); with filters on it keeps 4,535 of the same 6,506 boxes,
unchanged, removing 1,971 by score and 0 by area.

**Effect.** HOTA rose by +7.23 to +13.96 across the 21 re-scored model checkpoints and by +7.41 to +17.26 across the
seven inference settings (`results_after_tuning/final_review/historical_vs_corrected.csv`,
`results_after_tuning/before_after_comparison.csv`):

| Model | HISTORICAL HOTA (unfiltered) | Corrected HOTA | FP historical → corrected |
|---|---|---|---|
| Phase 5 ckpt_4 | 50.51 | 57.74 | 2,117 → 940 |
| Arm 1 ckpt_3 | 50.47 | 64.12 | 2,629 → 841 |

(`table_protocol_comparison`; these are two different protocols, not a model improvement.)

**Implication.** Every number in `archive/pre_tuning_baseline_v1/` uses the unfiltered protocol and is HISTORICAL; it
is kept unchanged as the record of the pre-tuning state. All numbers here use the corrected protocol. The two sets are
never mixed in one table or figure, except `table_protocol_comparison` / `fig_protocol_comparison`, which exist only to
show the protocol difference and are labelled as such.

## 3.3 Training experiments

### 3.3.1 Arm 1: real IOTD fine-tuning (chosen baseline)

- **Configuration (verified, `configs/train_iotd_real.yaml`):** start from Phase 5 `checkpoint_4.pth` (weights only),
  data `IOTD_train_real` (seq_01_1min + seq_03_30sec_b, 2,530 frames, 92 tracks), 2,528 steps per epoch, 5 epochs
  (12,640 steps), batch 1, LR 2e-5 (points 1e-6, backbone 0) dropping ×0.1 after epoch 4 (from step 10,112), seed 42.
- **Corrected HOTA per checkpoint:** 58.86 / 62.48 / 62.79 / **64.12** / 63.55 (ckpt_0–4).
- **Best checkpoint (written rule: highest HOTA of five at threshold 0.5):** checkpoint_3 — HOTA 64.12, DetA 60.54,
  AssA 68.21, MOTA 68.15, IDF1 84.64, IDSW 3, FP 841, FN 488, recall 88.33 %, precision 81.45 %.
- **Against Phase 5 (57.74):** +6.38 HOTA; DetA +10.67 (60.54 vs 49.87); AssA +0.91 (68.21 vs 67.30); recall 88.33 % vs
  73.77 %; FP 841 vs 940; FN 488 vs 1,097; IDSW 3 vs 0 (`table_corrected_candidates`).
- **Why checkpoint_3:** highest HOTA of the five. Checkpoint_4 has fewer FN (395 vs 488) but more FP (962 vs 841) and
  lower HOTA (63.55).
- **Interpretation:** the gain over Phase 5 comes mainly from detection (DetA, recall); association is about the same.

### 3.3.2 Arm 2: real + augmented fine-tuning (discarded)

- **Configuration (verified, `configs/train_iotd_real_aug.yaml`):** as Arm 1 plus 7,590 photometrically augmented
  frames (`IOTD_train_aug`: 3 copies of each training sequence; exposure gain, gamma, hue/saturation, JPEG quality,
  noise), 10,112 steps per epoch, LR drop after 40,448 steps. Augmented frames are extra training samples with copied
  labels, not extra annotated data.
- **Corrected HOTA per checkpoint:** 52.02 / **61.61** / 54.94 / 54.01 / 53.10.
- **Comparison with Arm 1:** lower at every checkpoint index (52.02 vs 58.86 … 53.10 vs 63.55). At equal optimizer steps
  (10,112: Arm 1 ckpt_3 vs Arm 2 ckpt_0) 64.12 vs 52.02 (−12.10). Arm 2's best (ckpt_1, 20,224 steps) 61.61 is 2.51
  below Arm 1's best.
- **Detection pattern:** FP rises monotonically with steps (379, 601, 1,897, 2,268, 2,276). Recall also rises after
  ckpt_1 (81.88 % → 93.19 % at ckpt_2) but precision falls from 85.07 % to 67.26 %, so HOTA drops.
- **Interpretation (assumed):** on this sequence photometric augmentation slowed convergence and then pushed the model
  towards over-detection; it did not raise peak performance. That real annotated data, not augmentation quantity, is
  the limiting factor is a hypothesis, not a measured result. The 2.51 gap between bests is about 1.1 seed SD.

### 3.3.3 assoc_lr0: association module frozen (discarded)

- **Change from Arm 1:** query-updater learning rate 0 (`LR_QUERY_UPDATER: 0.0`, patch 004); everything else as Arm 1.
- **Motivation (HISTORICAL observation):** under the unfiltered protocol Arm 1 had lower AssA than Phase 5 (54.51 vs
  58.49), suggesting the association module was drifting.
- **Verified frozen:** query-updater weights identical to Phase 5 (max |Δ| = 0.0, `results/tuning_stage1/status.json`).
- **Corrected HOTA per checkpoint:** **62.76** / 49.37 / 59.18 / 60.38 / 58.60; best ckpt_0. It collapses at ckpt_1
  (IDSW 21, FN 1,931).
- **Decision rule:** a corrected noise bar exists only at ckpt_3 (66.14). The best checkpoint (62.76) and the ckpt_3
  value (60.38) are both below it; at ckpt_3 assoc_lr0 is also below the Arm 1 seed mean (61.64). **Not passed.**
- **Association, corrected (direction checked):** Arm 1 ckpt_3 AssA 68.21 is **above** Phase 5's 67.30 (+0.91), as is
  ckpt_4 (67.43, +0.13); Arm 1 ckpt_0–2 are 0.35–4.11 below. assoc_lr0's AssA is **below** Phase 5 at every checkpoint (−0.68 to −9.26) and
  below Arm 1 at checkpoints 1–4.
- **Interpretation:** the large association loss seen in the HISTORICAL results was mostly an artefact of the missing
  filter; after correction the chosen Arm 1 checkpoint matches Phase 5's association. Freezing the query updater did not
  improve association, so the query updater is not the source of any remaining gap. The shared decoder is a candidate,
  but it was **not tested**.

### 3.3.4 lr_half: halved learning rates (discarded)

- **Change from Arm 1:** every non-zero LR × 0.5 (2e-5 → 1e-5, points 1e-6 → 5e-7); backbone stays 0.
- **Weight drift vs Phase 5 (verified):** mean |Δ| over non-backbone parameters 0.855e-3 vs Arm 1's 1.338e-3 = 0.64×
  (`results/tuning_stage1/status.json`, `results/tuning_stage1/smoke/smoke_result.json`).
- **Corrected HOTA per checkpoint:** **64.26** / 51.65 / 60.98 / 56.15 / 58.98; best ckpt_0. It collapses at ckpt_1
  (IDSW 21, FN 1,589).
- **Decision rule:** best 64.26 < bar 66.14; at ckpt_3 56.15 is below the Arm 1 seed mean 61.64. **Not passed.**
- **Against Arm 1 at equal checkpoints:** above only at ckpt_0 (64.26 vs 58.86); below at ckpt_1–4 (e.g. 60.98 vs 62.79,
  56.15 vs 64.12, 58.98 vs 63.55), with more FN at ckpt_2–4 (828 / 1,243 / 1,052 vs 464 / 488 / 395).
- **Why discarded:** its best is a single run only 0.14 above Arm 1 ckpt_3, far inside the seed spread, and the run is
  unstable after the first epoch. **Interpretation:** less weight drift from Phase 5 did not give better tracking on 92
  training tracks.

### 3.3.5 Seed repeats (variability estimate)

- Seeds 42, 1234, 2024, same Arm 1 configuration. Corrected HOTA at ckpt_3: 64.12 / 61.07 / 59.74 → **mean 61.64 ±
  2.25 SD**; **noise bar = mean + 2 × SD = 66.14** (`table_seed_spread`). Corrected seed values exist only for
  checkpoint_3; the other checkpoints of seeds 1234 and 2024 were not re-scored.
- n = 3 gives only a rough SD estimate. Seed 42 (64.12) is the highest of the three, so part of Arm 1's margin over
  Phase 5 may reflect a favourable seed; all three seeds are still above Phase 5's 57.74 (lowest: 59.74).

## 3.4 Inference-time sweep (discarded, S_A retained)

- **Tested:** seven combinations of detection = tracking threshold (0.5, 0.6, 0.7) and miss tolerance (20, 30, 40) on
  Arm 1 ckpt_3 (`table_inference_sweep`): S_A 0.5/30 **64.12**, S_B 0.6/30 62.10, S_C 0.7/30 60.19, S_D 0.5/20 63.89,
  S_E 0.5/40 62.77, S_F 0.6/20 64.25, S_G 0.7/20 59.83.
- **Why the earlier S_D result was misleading:** HISTORICAL S_D scored 54.51 vs S_A 50.47 because a shorter miss
  tolerance deleted some of the stale, low-confidence boxes that the output filter now removes in every setting. Under the
  corrected protocol S_D is 0.23 below S_A, and over three seeds S_D − S_A = −0.23 / +0.02 / −0.01 (mean −0.07;
  `results_after_tuning/seed_check_results.csv`).
- **Decision rule:** no setting exceeds the noise bar 66.14. S_F (64.25) is one run, +0.12 over S_A, with IDSW 6 vs 3.
  **S_A (0.5 / 0.5 / 30) is retained.**
- **Why RESULT_SCORE_THRESH was not tuned:** 0.5 is taken from the project config, which equals every upstream MeMOTR
  config; tuning it on the single validation sequence would add another selected degree of freedom.

## 3.5 C2 evaluation (no gain; future work)

- **What C2 does (verified, `modules/threshold_adapter.py`, `MeMOTR/models/runtime_tracker.py:83-93`):** a 3-16-8-1 MLP
  (209 parameters) maps the scene vector [brightness, contrast, active-track density] to τ = 0.3 + 0.4 · sigmoid(·) ∈
  (0.3, 0.7). τ replaces the fixed **detection** threshold for starting new tracks; the tracking threshold stays 0.5.
  The adapter was trained on UA-DETRAC and VisDrone, not IOTD.
- **Evaluation:** Phase 5 ckpt_4 on all three IOTD sequences, C2 off vs on (`table_c2_results`):

| Sequence | Mean luma | HOTA off → on | IDSW off → on | τ mean (range) |
|---|---|---|---|---|
| seq_01_1min (night) | 17.15 | 28.87 → 27.82 (−1.05) | 0 → 139 | 0.311 (0.307–0.320) |
| seq_02_30sec_a (night) | 35.82 | 57.74 → 56.42 (−1.32) | 0 → 7 | 0.332 (0.327–0.340) |
| seq_03_30sec_b (dusk) | 79.39 | 26.25 → 27.04 (+0.79) | 2 → 2 | 0.405 (0.355–0.452) |

  (mean luma 0–255 from `data/IOTD/seq_profile.csv`.)
- **Measured:** τ differs across scenes and rises with brightness (+0.094 from seq_01 to seq_03). HOTA falls on both
  night sequences and rises by 0.79 on dusk. On seq_02 the change (−1.32) is a loss, not a gain, and is smaller than one
  seed SD (2.25).
- **Interpretation:** at night the adapter lowers τ to about 0.31–0.33; this adds detections (FN 4,790 → 4,363 on seq_01)
  but adds false alarms and ID switches faster (FP 1,445 → 4,611, IDSW 0 → 139). The adapter's training target is a fixed
  brightness rule — τ ≈ 0.33 for dark, 0.50 for mid, 0.63 for bright frames, L2 loss (verified,
  `tools/train_c2_only.py:106-113`) — so it never sees tracking quality and cannot learn whether a given τ reduces ID
  switches on this camera. This is a design limitation of the current supervision, not proof that C2 cannot work.
- **Conclusion:** C2 will be revisited after the supervision target is revised and the adapter is retrained on IOTD. It is
  **not** concluded that C2 is ineffective in general — only that the current adapter, trained on different data with a
  brightness-only target, shows no gain here.

## 3.6 Factors considered and their outcome

| Factor | Tried | Outcome | Reason discarded / retained |
|---|---|---|---|
| IOTD real-only fine-tuning | Yes | **Retained** (Arm 1 ckpt_3, HOTA 64.12) | Best corrected HOTA of the default configuration; +6.38 over Phase 5; all 3 seeds above Phase 5 |
| Photometric augmentation | Yes | Discarded | Lower HOTA than real-only at every checkpoint; −12.10 at equal steps; FP grows with training |
| Freezing association module | Yes | Discarded | Below noise bar; AssA below Phase 5 at every checkpoint; source of association differences not identified (decoder untested) |
| Halving learning rates | Yes | Discarded | Below noise bar; below Arm 1 at ckpt_1–4 with more FN at ckpt_2–4; unstable after epoch 1 |
| Inference threshold 0.6, 0.7 | Yes | Discarded | All below S_A and the noise bar; 0.7 loses many detections (FN 1,047–1,138) |
| Miss tolerance 20 / 30 / 40 | Yes | 20 ≈ 30 after filter; 40 lower in one run | 20 vs 30: −0.07 mean over 3 seeds; 40: 62.77 vs 64.12, single run, inside seed SD |
| Output filter (score > 0.5, area > 100 px) | Found and fixed | **Applied to all results** | Upstream default omitted from the project tool; an implementation fix, not a tuning gain |
| C2 adapter | Yes | No gain on IOTD; future work | Trained on other data with a brightness-only target; worse on both night sequences |
| C1 perspective calibration | Not evaluated | Deferred | Zone boundaries are placeholders (0.30/0.70); PositionMLP not trained; calibration not run |
| Learning rate 5e-5 | Not tried | — | Assumed unlikely to help given that more updates (Arm 2, later Arm 1 checkpoints) did not help; not tested |
| Architecture changes | Out of scope | — | Backbone frozen; no backbone/encoder edits per project rules |
| Leave-one-sequence-out | Not done | — | Only three sequences; would triple compute (assumed little additional signal) |

## 3.7 Chosen configuration

- **Checkpoint:** Arm 1 `results/train_iotd_real/checkpoint_3.pth`
- **Runner:** `tools/run_inference_mot_filtered.py`
- **Settings:** det_thresh 0.5, track_thresh 0.5, miss_tol 30, use_c1=False, use_c2=False
- **Protocol:** upstream output filter, score > RESULT_SCORE_THRESH (0.5), area > 100 px; TrackEval HOTA/CLEAR/Identity,
  DO_PREPROC=False, S0 GT (SHA256 `17ebe278…d3472`)
- **Corrected HOTA on seq_02_30sec_a:** 64.12 (single run, seed 42); seed mean at ckpt_3 61.64 ± 2.25

## 3.8 Limitations

- One validation sequence (744 frames, 50 tracks, night); all choices were made on it.
- Three seeds for the noise estimate; n = 3 is too small for a reliable SD, and the corrected seed band exists only at
  checkpoint_3.
- The checkpoint was selected on the same sequence used for tuning. There is no test set; all results are validation
  performance, not test performance.
- 92 real training tracks (3,274 annotated frames across all three sequences, 2,530 for training). More data is the
  clearest path to further improvement (assumed).
- C1 not evaluated; C2 showed no gain with its current supervision; C2 was evaluated with Phase 5 weights, and two of the
  three C2 sequences are IOTD training sequences.
- All runs use single-class (class-agnostic) tracking; per-class results are not available.

---

## Corrections to the close-out brief

The brief that requested this document contained pre-written statements. Where a statement did not match the verified
numbers, this document states the measured value instead:

1. "Arm 2 … recall gains little past ckpt_1" — recall rises from 81.88 % to 93.19 % at ckpt_2; it is precision that
   falls (85.07 % → 67.26 %).
2. "AssA … best 68.2 in Arm 1 vs 58.5 corrected for Phase 5" — 58.49 is Phase 5's HISTORICAL AssA; corrected Phase 5 AssA
   is 67.30, slightly below Arm 1 ckpt_3 (68.21). assoc_lr0's AssA is below Phase 5 by 0.68–9.26 (not "4–16").
3. "association gap traced to decoder" — the decoder was not tested; the table says so.
4. "lr_half … below Arm 1 seed mean from checkpoint 2 onward" — a corrected seed mean exists only at ckpt_3; lr_half is
   below Arm 1 (seed 42) at ckpt_1–4 and below the seed mean at ckpt_3.
5. "Miss tolerance 40 worse" — one run, 1.35 HOTA lower, inside the seed SD; stated as "lower in one run".
6. "HOTA suppressed by 7–14 points across all runs" — 7.23–13.96 across the 21 model checkpoints, 7.41–17.26 across
   the seven inference settings.
7. "C2 maps … to a tracking threshold" — τ replaces the detection threshold for new tracks; the tracking threshold is
   unchanged. "All three sequences are similar in lighting" — mean luma is 17, 36 and 79, and τ did vary across them.
8. "Real annotated data was the limiting factor" — kept only as an interpretation (assumed), not as a finding.
