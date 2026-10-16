# Selected settings and rationale

Final review, 2026-10-16. Every number below is **Verified** from the file named next to it unless marked otherwise.
"Corrected" = scored with the upstream MeMOTR output filter (`tools/run_inference_mot_filtered.py`).
"Historical" = scored with the old unfiltered output (`tools/run_inference_mot.py`); historical numbers are not
comparable with corrected ones.

## 1. Selected configuration

| Item | Setting | Kind | Status |
|---|---|---|---|
| Model | Arm 1 `results/train_iotd_real/checkpoint_3.pth` (seed 42) | trained model | selected (section 4) |
| Inference runner | `tools/run_inference_mot_filtered.py` | implementation | selected |
| Detection threshold | 0.5 | tunable | selected (S_A) |
| Tracking threshold | 0.5 | tunable | selected (S_A) |
| Miss tolerance | 30 | tunable | selected (S_A); 20 rejected |
| Output score filter | `RESULT_SCORE_THRESH` = 0.5 (keep score > 0.5) | fixed upstream setting, **not tuned** | applied |
| Box-area filter | area > 100 px² | fixed upstream setting, **not tuned** | applied |
| C1 | disabled | contribution | implemented; not calibrated, trained or evaluated |
| C2 | disabled | contribution | corrected evaluation done: no net benefit (section 5); needs camera-specific calibration |

Machine-readable copy: `selected_settings.csv`.

**The two output filters are not a tuning result.** They make the project's output step match upstream MeMOTR
(`MeMOTR/submit_engine.py:94-95`, value from `configs/train_iotd_real.yaml:38`). They raised every model's score
because the old script wrote boxes that upstream MeMOTR would never output. They are a correction of the
implementation, not a research contribution or an improvement of the model.

## 2. Why S_A (det 0.5, track 0.5, miss 30) was selected

**What we changed:** the detection/tracking threshold (0.5, 0.6, 0.7) and the miss tolerance (20, 30, 40), seven
settings S_A–S_G, on Arm 1 checkpoint_3.
**What stayed the same:** the model weights, the validation sequence, the GT, TrackEval, and the output filter.

| Setting | det/track | miss | HOTA | IDSW | FP | FN | Source |
|---|---|---|---|---|---|---|---|
| **S_A** | 0.5 | 30 | **64.12** | 3 | 841 | 488 | `results_after_tuning/corrected_sweep_results.csv` |
| S_B | 0.6 | 30 | 62.10 | 8 | 723 | 660 | same |
| S_C | 0.7 | 30 | 60.19 | 5 | 393 | 1138 | same |
| S_D | 0.5 | 20 | 63.89 | 2 | 903 | 430 | same |
| S_E | 0.5 | 40 | 62.77 | 6 | 736 | 644 | same |
| S_F | 0.6 | 20 | 64.25 | 6 | 770 | 497 | same |
| S_G | 0.7 | 20 | 59.83 | 9 | 423 | 1047 | same |

Three training seeds at checkpoint_3 (`results_after_tuning/seed_check_results.csv`):

| Seed | S_A | S_D | S_D − S_A |
|---|---|---|---|
| 42 | 64.12 | 63.89 | −0.23 |
| 1234 | 61.07 | 61.09 | +0.02 |
| 2024 | 59.74 | 59.73 | −0.01 |
| mean ± SD | 61.64 ± 2.25 | 61.57 ± 2.12 | −0.07 |

**What the results showed and why S_A is kept:**
- Changing miss tolerance from 30 to 20 makes no consistent difference (−0.23, +0.02, −0.01 HOTA across three seeds).
  With no evidence for the change, the ordinary default S_A stays.
- S_F has the highest single score (64.25), but it is one run, only +0.12 over S_A, has more ID switches (6 vs 3) and is
  far below the acceptance bar (seed mean + 2 SD = 66.14). A setting is not chosen from one favourable number.
- The earlier "S_D is best" result (54.51 vs 50.47, historical) came from the unfiltered output: a shorter miss
  tolerance happened to delete some junk boxes that the filter now removes for every setting. It is an artefact of the
  evaluation implementation, **not** evidence that miss tolerance 20 helps.

**Rejected alternatives:** S_D (no consistent gain), S_F (single run, more ID switches), higher thresholds S_B/S_C/S_G
(lower HOTA, many more missed vehicles), S_E (longer miss tolerance, lower HOTA).

## 3. Why the acceptance bar was re-measured
The original bar (seed mean 48.94 + 2 × 1.34) was measured on unfiltered output, which changed HOTA by +11.8 to +13.7 per
seed. It cannot be reused. It was re-measured by re-running inference only (no training) on the existing seed 1234 and
2024 checkpoints: corrected mean 61.64, SD 2.25, bar 66.14 (`seed_check_results.csv`). It exists only for
checkpoint_3; no corrected seed band exists for other checkpoints.

## 4. Why Arm 1 checkpoint_3 is the selected model
Corrected best checkpoint of each model (`final_review/corrected_model_comparison.csv`):

| Model | Best checkpoint | HOTA | DetA | AssA | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|
| lr_half | checkpoint_0 | 64.26 | 59.95 | 69.23 | 4 | 607 | 677 |
| **Arm 1** | **checkpoint_3** | **64.12** | 60.54 | 68.21 | 3 | 841 | 488 |
| assoc_lr0 | checkpoint_0 | 62.76 | 59.37 | 66.62 | 5 | 1134 | 355 |
| Arm 2 | checkpoint_1 | 61.61 | 58.10 | 65.51 | 3 | 601 | 758 |
| Phase 5 | checkpoint_4 | 57.74 | 49.87 | 67.30 | 0 | 940 | 1097 |

- Arm 1 checkpoint_3 is the best checkpoint of the standard configuration under the written rule (highest HOTA of
  five checkpoints at 0.5) and it is also the one with a corrected three-seed check.
- `lr_half` checkpoint_0 is 0.14 HOTA higher, but that is a single run, far inside the seed spread (SD 2.25), and the same
  run collapses one epoch later (checkpoint_1: HOTA 51.65, IDSW 21; checkpoint_3: 56.15). That is not a reliable
  improvement, so the default learning rate stays. **Inferred:** a repeat with other seeds would be needed to show anything.
- Arm 1 vs Phase 5: +6.38 HOTA (64.12 vs 57.74). All three Arm 1-configuration seeds at checkpoint_3 (64.12, 61.07, 59.74)
  are above Phase 5's 57.74, so fine-tuning on IOTD improves tracking on this sequence in every seed tried.
  This is **Inferred** from 3 seeds on one sequence; it is not a statistical test.

## 5. Why C2 stays disabled
Corrected evaluation, Phase 5 checkpoint_4, C2 off → on (`final_review/corrected_c2_comparison.csv`):

| Sequence | HOTA | IDSW | FP | FN | mean τ (range) |
|---|---|---|---|---|---|
| seq_01 night | 28.87 → 27.82 (−1.05) | 0 → 139 | 1,445 → 4,611 | 4,790 → 4,363 | 0.311 (0.307–0.320) |
| seq_02 night | 57.74 → 56.42 (−1.32) | 0 → 7 | 940 → 1,442 | 1,097 → 794 | 0.332 (0.327–0.340) |
| seq_03 dusk | 26.25 → 27.04 (+0.79) | 2 → 2 | 402 → 420 | 16,104 → 15,817 | 0.404 (0.355–0.452) |

C2 lowers the detection threshold to about 0.31–0.33 at night, finds more vehicles (fewer FN) but adds many false
alarms and ID switches; HOTA falls on both night sequences and rises by 0.79 on the dusk sequence, which is within
run-to-run noise. The adapter was trained on UA-DETRAC and VisDrone, not on this camera. C2 stays disabled until it is
calibrated for IOTD and shows a corrected, repeated improvement.

## 6. Why C1 stays disabled
C1 is implemented (`modules/position_decay.py`, wired into `MeMOTR/models/query_updater.py` behind `USE_C1`, unit tests
present) but is not calibrated (zone boundaries still the placeholders 0.30 / 0.70), not trained and not evaluated.
No performance claim is made.

## 7. Limitations
- One validation sequence (`seq_02_30sec_a`, 744 frames, night); choices made on it are optimistic.
- Three seeds only, and a corrected seed band only at checkpoint_3; one run for every other corrected row.
- No statistical significance is claimed.
- C2 was evaluated with Phase 5 weights (never trained on IOTD); seq_01 and seq_03 are IOTD training sequences.
