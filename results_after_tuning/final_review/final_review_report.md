# Final review report — MeMOTR fine-tuning on IOTD

2026-10-16. No training was run for this review; 25 inference runs were made with existing checkpoints.
Labels: **Verified** (read from the named file in this review), **Inferred** (reasoned from verified numbers),
**Historical** (old unfiltered output protocol — not comparable with corrected numbers).

## 1. Summary
- The project's original output script wrote boxes that upstream MeMOTR would never output (tracks currently missed,
  confidence as low as 0.003). After matching upstream (keep score > 0.5 and area > 100 px), **every** model scores
  much higher and some earlier conclusions change. All final comparisons below use this corrected procedure.
- **Selected:** Arm 1 checkpoint_3 with detection 0.5, tracking 0.5, miss tolerance 30, filtered runner; C1 and C2 off.
- **Corrected HOTA on `seq_02_30sec_a`:** Arm 1 checkpoint_3 **64.12** vs Phase 5 **57.74** (+6.38). All three Arm 1-config
  seeds (64.12 / 61.07 / 59.74) are above Phase 5. **Verified** values; robustness **Inferred** (3 seeds, one sequence).
- The earlier "S_D (miss tolerance 20) is better" finding does not hold under the corrected procedure (−0.07 HOTA mean
  over three seeds).
- C2 lowers HOTA on both night sequences under the corrected procedure; it stays off. C1 is not evaluated.

## 2. Method
| Item | Value |
|---|---|
| Validation sequence | `seq_02_30sec_a`, 744 frames, 50 tracks, night, 1920×1080 |
| GT | original `data/IOTD_val/seq_02_30sec_a/gt/gt.txt`, SHA256 `17ebe278…d3472`; evaluation-only class-1 copy `results/eval_iotd_val/gt_eval` |
| Inference | `tools/run_inference_mot_filtered.py`; det 0.5, track 0.5, miss 30; output filter score > `RESULT_SCORE_THRESH` (0.5, `configs/train_iotd_real.yaml:38`) and area > 100 px (`MeMOTR/submit_engine.py:94-95`) |
| Configs | `configs/train_iotd_real.yaml` (Phase 5, Arm 1, Stage 1 variants); `configs/train_iotd_real_aug.yaml` (Arm 2) — identical except training-data keys |
| Scoring | `tools/run_trackeval.py` — TrackEval HOTA, CLEAR, Identity, `DO_PREPROC=False` |
| Best-checkpoint rule | highest HOTA of the five checkpoints at threshold 0.5, under the corrected procedure |
| Runner check | filters off → byte-identical to the original output; filters on → strict subset of the original boxes (`results_after_tuning/logs/00_…`, `01_S_A.txt`) |

## 3. Stage 1 completion (Verified — `final_review/logs/E002_verify.txt`)
Seed 1234, seed 2024, `assoc_lr0`, `lr_half`: each 5/5 checkpoints, 5/5 S0 evaluation rows in
`results/tuning_stage1/stage1_results.csv`, 0 mismatches against the TrackEval summaries, status "done". Arm 1 seed 42:
5/5 rows in the frozen registry. Historical (unfiltered) seed spread per checkpoint: 46.78 ± 1.41, 48.94 ± 0.83,
49.40 ± 0.60, 48.94 ± 1.34, 48.99 ± 0.99 (ranges 1.18–2.67). Historical best checkpoints: Arm 1 ckpt_3 (50.47), seed1
ckpt_2 (50.05), seed2 ckpt_4 (49.54), assoc_lr0 ckpt_0 (50.45), lr_half ckpt_0 (51.19), Arm 2 ckpt_1 (48.17).

## 4. Corrected model comparison (Verified — `final_review/corrected_model_comparison.csv`)

| Model | Ckpt | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Prec % | Rec % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 5 | 4 | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | 0 | 940 | 1097 | 76.65 | 73.77 |
| Arm 1 | 0 | 58.86 | 55.20 | 63.20 | 59.73 | 77.15 | 12 | 908 | 764 | 79.01 | 81.73 |
| Arm 1 | 1 | 62.48 | 58.70 | 66.96 | 67.24 | 82.47 | 7 | 718 | 645 | 83.13 | 84.58 |
| Arm 1 | 2 | 62.79 | 60.09 | 65.94 | 68.67 | 82.57 | 7 | 839 | 464 | 81.59 | 88.91 |
| **Arm 1** | **3** | **64.12** | 60.54 | 68.21 | 68.15 | 84.64 | 3 | 841 | 488 | 81.45 | 88.33 |
| Arm 1 | 4 | 63.55 | 60.14 | 67.43 | 67.48 | 84.25 | 3 | 962 | 395 | 79.74 | 90.56 |
| Arm 2 | 0 | 52.02 | 45.71 | 59.36 | 50.17 | 68.37 | 6 | 379 | 1699 | 86.76 | 59.37 |
| **Arm 2** | **1** | **61.61** | 58.10 | 65.51 | 67.43 | 82.83 | 3 | 601 | 758 | 85.07 | 81.88 |
| Arm 2 | 2 | 54.94 | 51.38 | 58.94 | 47.80 | 77.99 | 1 | 1897 | 285 | 67.26 | 93.19 |
| Arm 2 | 3 | 54.01 | 49.78 | 58.75 | 41.06 | 76.06 | 1 | 2268 | 196 | 63.73 | 95.31 |
| Arm 2 | 4 | 53.10 | 48.53 | 58.25 | 38.76 | 75.22 | 1 | 2276 | 284 | 63.14 | 93.21 |
| **assoc_lr0** | **0** | **62.76** | 59.37 | 66.62 | 64.28 | 82.47 | 5 | 1134 | 355 | 77.14 | 91.51 |
| assoc_lr0 | 1 | 49.37 | 42.27 | 58.04 | 46.58 | 62.78 | 21 | 282 | 1931 | 88.87 | 53.83 |
| assoc_lr0 | 2 | 59.18 | 56.16 | 62.75 | 59.35 | 79.02 | 8 | 1282 | 410 | 74.63 | 90.20 |
| assoc_lr0 | 3 | 60.38 | 55.27 | 66.27 | 55.05 | 79.43 | 4 | 1510 | 366 | 71.65 | 91.25 |
| assoc_lr0 | 4 | 58.60 | 55.11 | 62.63 | 55.84 | 78.06 | 6 | 1450 | 391 | 72.33 | 90.65 |
| **lr_half** | **0** | **64.26** | 59.95 | 69.23 | 69.20 | 83.67 | 4 | 607 | 677 | 85.24 | 83.81 |
| lr_half | 1 | 51.65 | 46.97 | 57.14 | 52.22 | 67.82 | 21 | 388 | 1589 | 86.98 | 62.00 |
| lr_half | 2 | 60.98 | 56.40 | 66.28 | 65.66 | 80.86 | 7 | 601 | 828 | 84.80 | 80.20 |
| lr_half | 3 | 56.15 | 51.37 | 61.70 | 57.08 | 73.63 | 13 | 539 | 1243 | 84.50 | 70.28 |
| lr_half | 4 | 58.98 | 53.59 | 65.22 | 60.45 | 77.13 | 8 | 594 | 1052 | 84.05 | 74.84 |

Bold = best checkpoint of each model. Arm 1 checkpoint_3 is reused from the corrected sweep (S_A); all other rows were
newly evaluated in this review. All rows seed 42.

### Findings (corrected procedure)
1. **IOTD fine-tuning helps on this sequence.** Arm 1 ckpt_3 64.12 vs Phase 5 57.74 (+6.38); DetA 60.54 vs 49.87; recall
   88.33 % vs 73.77 %; AssA 68.21 vs 67.30. All three Arm 1-config seeds at ckpt_3 exceed Phase 5. (**Verified** numbers;
   **Inferred** robustness.)
2. **The earlier "association got worse after fine-tuning" finding was an evaluation artefact.** Historical AssA:
   Arm 1 54.51 vs Phase 5 58.49; corrected: 68.21 vs 67.30. Phase 5 still has the fewest ID switches (0 vs 3).
3. **Augmentation (Arm 2) did not help.** Best Arm 2 61.61 (ckpt_1) vs Arm 1 64.12; from ckpt_2 on, Arm 2 produces
   ~1,900–2,280 false positives. One run each; the 2.51 gap is about 1.1 seed SD (**Inferred**: not established).
4. **The Stage 1 variants are not better.** `lr_half` ckpt_0 is 0.14 above Arm 1 ckpt_3 (single run, well inside SD 2.25)
   and both variants collapse at ckpt_1 (HOTA 49.37 / 51.65, IDSW 21 / 21) — the same pattern seen historically. The
   corrected seed band exists only at ckpt_3, where `assoc_lr0` (60.38) and `lr_half` (56.15) are below the Arm 1 seed
   mean (61.64). Decision: keep the default learning rates.
5. **Every model gained from the filter**, by different amounts (best-checkpoint HOTA historical → corrected:
   Phase 5 50.51 → 57.74, Arm 1 50.47 → 64.12, Arm 2 48.17 → 61.61, assoc_lr0 50.45 → 62.76, lr_half 51.19 → 64.26;
   `historical_vs_corrected.csv`). Historical rankings must therefore not be quoted as final.

## 5. Inference settings
See `selected_settings_and_rationale.md` section 2: S_A (0.5 / 0.5 / 30) kept; S_D gives −0.07 HOTA over three seeds; S_F
is a single run below the bar 66.14.

## 6. C2 — corrected evaluation (Verified — `final_review/corrected_c2_comparison.csv`)

| Sequence | Lighting | C2 | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Prec % | Rec % | mean τ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seq_01_1min | night | off | 28.87 | 16.69 | 50.02 | 3.30 | 34.66 | 0 | 1445 | 4790 | 53.43 | 25.71 | 0.500 |
| seq_01_1min | night | on | 27.82 | 14.60 | 53.17 | −41.33 | 29.57 | 139 | 4611 | 4363 | 31.14 | 32.34 | 0.311 |
| seq_02_30sec_a | night | off | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | 0 | 940 | 1097 | 76.65 | 73.77 | 0.500 |
| seq_02_30sec_a | night | on | 56.42 | 49.17 | 65.34 | 46.37 | 73.83 | 7 | 1442 | 794 | 70.14 | 81.01 | 0.332 |
| seq_03_30sec_b | dusk | off | 26.25 | 18.39 | 37.54 | 23.39 | 37.54 | 2 | 402 | 16104 | 93.12 | 25.26 | 0.500 |
| seq_03_30sec_b | dusk | on | 27.04 | 19.24 | 38.08 | 24.64 | 39.19 | 2 | 420 | 15817 | 93.17 | 26.59 | 0.404 |

Phase 5 checkpoint_4, adapter `results/phase6_c2_interim/c2_adapter_epoch_4.pth`. The adapter's threshold is higher for the
brighter dusk sequence (0.404) than for the night sequences (0.311, 0.332), as designed, but at night the lower threshold
adds false alarms and ID switches faster than it adds correct detections. Historical (unfiltered) C2 rows are kept
separately in `historical_c2_unfiltered.csv` (seq_01 27.44 → 7.71, seq_02 50.51 → 45.76, seq_03 29.82 → 30.72) and are
not comparable. **Decision:** C2 off.

## 7. C1
Implemented (code, `USE_C1` flag, unit tests); zone boundaries are placeholders (0.30 / 0.70), PositionMLP untrained, no
evaluation. C1 has been implemented but is not yet fully calibrated and evaluated; no performance-improvement claim is made.

## 8. Integrity checks (Verified)
- IOTD hashes before and after inference: 3,283 files, changed 0, missing 0, added 0
  (`final_review/logs/E004_hash_before.txt`, `hash_after.txt`).
- Frozen archive: 55 files and 66 source files unchanged; zip read-only (`final_review/logs/archive_verify.txt`).
- Original `tools/run_inference_mot.py` unchanged (SHA256 `7fa7b110…952dc4`); no earlier result folder was written to.
- Failures recorded: Arm 2 ckpt_4 first attempt crashed natively (exit `0xC0000409`; kept as
  `runs/Arm2_ckpt_4_failed_attempt1`), retry succeeded; E002 verify script failed once on a column name, fixed.

## 9. Limitations and open items
1. One validation sequence (night, 744 frames); every selection was made on it, so results are optimistic.
2. One run per corrected row except Arm 1 ckpt_3 (three seeds). No significance claims.
3. The frozen archive and all historical reports use the unfiltered procedure; they are kept unchanged for the record and
   must be read as historical.
4. C2 needs camera-specific calibration (adapter trained on UA-DETRAC/VisDrone); C1 needs calibration, training and
   evaluation.
5. Four medium-confidence GT gaps in seq_02 are unverified; no corrected-GT (S1) scoring exists.
6. A day-time validation sequence would be needed to say anything about daylight performance.
