# Final Inference Verification — corrected output filter (no retraining)

Date: 2026-10-16. Arm 1 `checkpoint_3` (seed 42), validation sequence `seq_02_30sec_a` (744 frames), S0 GT
(SHA256 `17ebe278…d3472`, evaluation-only class-1 copy `results/eval_iotd_val/gt_eval`), TrackEval via
`tools/run_trackeval.py` (`DO_PREPROC=False`), `use_c1=False`, `use_c2=False`. Every number below was read from a
TrackEval summary file produced in this run (paths in the CSVs); raw logs in `results_after_tuning/logs/`.

## 1. What the upstream output filter does (verified in code)

| Item | Finding | Where |
|---|---|---|
| Filter | After each frame, a track is written only if `max(tracks.scores) > result_score_thresh`, then only if its box area in original pixels `> 100` | `MeMOTR/submit_engine.py:94-95`, `:122-131` |
| Score | `tracks.scores = sigmoid(logits)` of the **current frame's** decoder output for that track | `MeMOTR/models/runtime_tracker.py:61`, `MeMOTR/models/utils.py:171-172` |
| Config key | `RESULT_SCORE_THRESH` (`submit_engine.py:205`) | value **0.5** in `configs/train_iotd_real.yaml:38` (also 0.5 in every upstream MeMOTR config) |
| Interaction | A track enters its "missed" state when that score `< TRACK_SCORE_THRESH` (`runtime_tracker.py:63`) and keeps its ID for `MISS_TOLERANCE` frames; with the filter, missed tracks are not written | |
| Project tool | `tools/run_inference_mot.py:184-204` writes every track with `id >= 0` regardless of score; it reads `scores` (`:190`) but never filters | |

## 2. Correction and its verification

- Original script, predictions, results and the frozen archive are untouched (`tools/run_inference_mot.py` SHA256
  `7fa7b110…952dc4` before and after).
- New runner `tools/run_inference_mot_filtered.py`: a copy with only the two upstream filters added (diff:
  `results_after_tuning/logs/runner_diff.txt`). Threshold taken from the config (`RESULT_SCORE_THRESH` = 0.5), not
  chosen.
- **Equivalence test (filters disabled):** reproduced the original S_A prediction file byte for byte (SHA256
  `46c4061d…9c34b9`, 6,506 rows) — `logs/00_equivalence_filters_off.txt`.
- **Smoke test (corrected S_A):** same 6,506 candidate boxes (tracking unchanged — the filter acts only on output);
  1,971 removed by score, 0 by area; lowest written score 0.50004; all 4,535 written rows appear unchanged in the
  original file — `logs/01_S_A.txt`.
- The area filter removed **0** boxes in every run, so the whole effect comes from the score filter.

## 3. Corrected sweep (Arm 1 ckpt_3)

| Setting | det/track | miss | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Precision % | Recall % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S_A | 0.5 | 30 | 64.12 | 60.54 | 68.21 | 68.15 | 84.64 | 3 | 841 | 488 | 81.45 | 88.33 |
| S_B | 0.6 | 30 | 62.10 | 59.14 | 65.56 | 66.74 | 81.36 | 8 | 723 | 660 | 82.97 | 84.22 |
| S_C | 0.7 | 30 | 60.19 | 55.27 | 65.83 | 63.27 | 78.54 | 5 | 393 | 1138 | 88.57 | 72.79 |
| S_D | 0.5 | 20 | 63.89 | 60.42 | 67.89 | 68.08 | 84.51 | 2 | 903 | 430 | 80.60 | 89.72 |
| S_E | 0.5 | 40 | 62.77 | 59.22 | 66.87 | 66.86 | 82.76 | 6 | 736 | 644 | 82.78 | 84.60 |
| S_F | 0.6 | 20 | **64.25** | **60.98** | 68.01 | **69.56** | 84.10 | 6 | 770 | 497 | 82.72 | 88.12 |
| S_G | 0.7 | 20 | 59.83 | 56.37 | 63.81 | 64.63 | 77.36 | 9 | 423 | 1047 | 88.11 | 74.96 |

### Before (unfiltered) → after (filtered)

| Setting | HOTA before → after | IDSW | FP | FN |
|---|---|---|---|---|
| S_A | 50.47 → 64.12 (+13.65) | 9 → 3 | 2629 → 841 | 305 → 488 |
| S_B | 49.92 → 62.10 (+12.18) | 12 → 8 | 2225 → 723 | 454 → 660 |
| S_C | 49.35 → 60.19 (+10.84) | 8 → 5 | 1658 → 393 | 961 → 1138 |
| S_D | 54.51 → 63.89 (+9.38) | 9 → 2 | 2159 → 903 | 244 → 430 |
| S_E | 45.52 → 62.77 (+17.25) | 13 → 6 | 3175 → 736 | 444 → 644 |
| S_F | 55.45 → 64.25 (+8.80) | 10 → 6 | 1773 → 770 | 354 → 497 |
| S_G | 52.42 → 59.83 (+7.41) | 13 → 9 | 1250 → 423 | 883 → 1047 |

The filter removes boxes of tracks in their miss window; FP falls by 57–77 % in every setting while FN rises by
143–206 rows (some low-score boxes that the unfiltered output wrote were correct). The spread between settings
shrinks from 9.9 HOTA (45.52–55.45) to 4.4 (59.83–64.25).

## 4. Does S_D still improve on S_A?

**No.** Corrected S_D − S_A = **−0.23 HOTA** (63.89 vs 64.12), DetA −0.12, AssA −0.32, IDSW 2 vs 3, FP +62,
FN −58. The +4.04 HOTA gain seen before came from the missing filter: a shorter miss tolerance removed some of the
low-score boxes that the filter now removes for every setting.

## 5. Re-assessing the seed-based acceptance rule

The original band (seed mean 48.94, SD 1.34 at ckpt_3) was measured on **unfiltered** output and cannot be reused:
the filter changes HOTA by +11.8 to +13.7 per seed, and by different amounts per setting (+7.4 to +17.3). The band
was therefore **re-measured under the corrected filter** by re-running inference only (no training) on the existing
seed1 (1234) and seed2 (2024) `checkpoint_3` files:

| Seed | S_A HOTA | S_D HOTA | S_D − S_A | IDSW S_A / S_D | FP S_A / S_D | FN S_A / S_D |
|---|---|---|---|---|---|---|
| 42 (Arm 1) | 64.12 | 63.89 | −0.23 | 3 / 2 | 841 / 903 | 488 / 430 |
| 1234 | 61.07 | 61.09 | +0.02 | 3 / 4 | 1297 / 1315 | 346 / 318 |
| 2024 | 59.74 | 59.73 | −0.01 | 1 / 1 | 582 / 593 | 886 / 877 |
| mean ± SD | 61.64 ± 2.25 | 61.57 ± 2.12 | −0.07 | | | |

- Corrected band at ckpt_3: mean 61.64, SD 2.25 (n = 3) → bar (mean + 2 SD) = **66.14**. It is wider than the original
  band (SD 1.34) because the seeds differ in their FP/FN balance (seed 1234: FP 1,297; seed 2024: FN 886).
- Applying the rule (HOTA > 66.14, IDSW ≤ S_A, ≥ 1 HOTA over S_A): **no setting passes**. The best, S_F (64.25),
  is +0.12 over S_A with IDSW 6 vs 3.
- The paired S_D − S_A difference is within ±0.25 HOTA for all three seeds, i.e. miss tolerance 20 vs 30 makes no
  measurable difference once the output is filtered.
- The seed 2024 / S_D run crashed once (native exit `0xC0000409`, no Python traceback, at the start of inference); the
  single permitted retry ran normally. The failed attempt is kept in
  `seed_check/seed2024_S_D_failed_attempt1/`.

## 6. Recommendation

**Inference configuration for Arm 1: S_A — det 0.5, track 0.5, miss tolerance 30, with the upstream output filter
(`RESULT_SCORE_THRESH` 0.5, area > 100 px) via `tools/run_inference_mot_filtered.py`.** No other setting passes the
rule, S_D is no longer better, and S_A is the configuration's own default. C2 stays **disabled** (no new evidence;
it was not re-evaluated here). C1 has **not** been evaluated and no claim is made about it.

## 7. Limitations

1. **All earlier conclusions were made on unfiltered output** — Phase 5 vs Arm 1/Arm 2, the Stage 1 variants
   (assoc_lr0, lr_half), the original sweep and the C2 evaluation, and the frozen archive
   `archive/pre_tuning_baseline_v1/`. They should be re-scored with the filtered runner before being reported;
   the archive itself was not changed.
2. One validation sequence (744 frames, night); settings chosen on the validation sequence are optimistic.
3. Three seeds only; SD from n = 3 is a rough estimate. One run per inference setting for seeds 1234/2024 (S_A, S_D only).
4. RESULT_SCORE_THRESH was taken from the config (0.5) and not tuned; with det/track 0.6–0.7 it is below the track
   threshold, so tracks scoring between 0.5 and the track threshold are written while counted as missed.
5. The tracker output's confidence column is still a constant 1 (unchanged format), so no score-based curves.
6. No statistical significance is claimed.

## 8. Output paths

| File | Content |
|---|---|
| `results_after_tuning/corrected_sweep_results.csv` | 7 corrected runs, registry columns + det/track/miss, filter settings and per-run filter counts |
| `results_after_tuning/before_after_comparison.csv` | Every metric before / after / delta for the 7 settings |
| `results_after_tuning/seed_check_results.csv` | Corrected S_A and S_D for seeds 42, 1234, 2024 |
| `results_after_tuning/final_results_dashboard.png` | Panels A–G: before/after HOTA, IDSW, MOTA, FP, FN, seed check, results table |
| `results_after_tuning/final_results_report.md` | This report |
| `results_after_tuning/sweep/S_A … S_G/` | Corrected predictions `seq_02_30sec_a.txt`, `inference.log`, TrackEval output |
| `results_after_tuning/seed_check/seed{1234,2024}_{S_A,S_D}/` | Seed-check predictions and TrackEval output |
| `results_after_tuning/verify_filters_off/S_A/` | Equivalence run (filters disabled) |
| `results_after_tuning/logs/` | All raw logs, including `runner_diff.txt` |
| `tools/run_inference_mot_filtered.py`, `tools/after_tuning_*.py` | Corrected runner, sweep/seed runners, batch driver, report builder |
