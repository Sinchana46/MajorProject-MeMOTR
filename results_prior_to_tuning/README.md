# Pre-Tuning Baseline v1

**Created:** 2026-10-14T03:17:07Z

## Protocol (fixed for this package)

- **Validation sequence:** seq_02_30sec_a (744 frames, 50 tracks, night scene, mean luma 35.6)
- **Detection and tracking threshold:** 0.5
- **Miss tolerance:** 30 frames
- **C1/C2 modules:** disabled (use_c1=False, use_c2=False)
- **Evaluation GT:** S0 — original unmodified seq_02_30sec_a/gt/gt.txt
  - SHA256: 17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472
- **TrackEval class remapping:** all classes set to 1 (evaluation-only copy, never modifying original GT)
- **DO_PREPROC:** False — no distractor/ignore regions; every unmatched prediction is a FP

**Single validation sequence. Single run per setting. Seed 42 for both IOTD arms.
One run per setting. No statistical significance can be claimed.**

The tracker confidence column is a constant 1 in all tracker outputs.
No score-based precision-recall curves are possible or drawn.

## Experiments

| Run | Config | Start weights | Seed | Steps/epoch | Epochs | LR |
|---|---|---|---|---|---|---|
| Phase 5 | (UA-DETRAC+VisDrone fine-tune) | DanceTrack pretrain | 42 | — | — | 1e-4 (MultiStep) |
| Arm 1 | configs/train_iotd_real.yaml | Phase5 ckpt4 (wt only) | 42 | 2,528 | 5 | 2e-5→2e-6 (drop ep4) |
| Arm 2 | configs/train_iotd_real_aug.yaml | Phase5 ckpt4 (wt only) | 42 | 10,112 | 5 | 2e-5→2e-6 (drop ep4) |

Clip length = 2 throughout all 5 epochs for both IOTD arms (SAMPLE_STEPS=[6,10,14] milestone never triggered).
Backbone frozen (LR_BACKBONE=0.0) in both arms. Step time: 0.664 s/step (verified from logs).

## How to regenerate

From the project root (D:/MajorProject-MeMOTR):

```bash
python results/pre_tuning_baseline_v1/scripts/generate_all.py
```

This re-runs tools/pretune_stepA.py, tools/pretune_stepB.py, tools/pretune_stepC.py in order.
All numbers are read from the saved TrackEval summaries; none are hard-coded.

## How to compare new results

```bash
python results/pre_tuning_baseline_v1/scripts/compare_results.py \
    --new_dir <folder> --out_dir <output> [--run Arm1 --ckpt ckpt_3 --thresh 0.5]
```

The script warns if scoring, threshold, GT hash, or sequence differ from the registry.
It does NOT modify the frozen registry.

## Answers to protocol questions

### Q1: Did Arm 1 improve HOTA over Phase 5?
- **Best ckpt (ckpt_3, 10,112 steps):** HOTA 50.47 vs Phase 5 HOTA 50.51 → **-0.04 (within noise)**
- **Last ckpt (ckpt_4, 12,640 steps):** HOTA 49.58 → **-0.93 below Phase 5**
- MEASURED FACT: Arm 1 ckpt_3 essentially matches Phase 5 HOTA with a strong gain in recall (92.71 vs 79.75) and reduction in FN (305 vs 847), at the cost of higher FP (2,629 vs 2,117). AssA does not match Phase 5 (54.51 vs 58.49).

### Q2: Did Arm 2 improve HOTA over Phase 5?
- **Best ckpt (ckpt_1, 20,224 steps):** HOTA 48.17 → **-2.34 below Phase 5**
- **Last ckpt (ckpt_4, 50,560 steps):** HOTA 42.84 → **-7.67 below Phase 5**
- MEASURED FACT: No Arm 2 checkpoint surpasses Phase 5 HOTA. FP grows substantially in later epochs (4,266 at ckpt_4 vs 2,117 for Phase 5).

### Q3: Checkpoint with highest HOTA / lowest FP / lowest FN / lowest IDSW (all runs):
- **Highest HOTA:** Phase 5 ckpt_4 (HOTA 50.51)
- **Highest HOTA (Arm 1):** ckpt_3 (50.47)  |  **Arm 2:** ckpt_1 (48.17)
- **Lowest FP (all):** Phase 5 ckpt_4 (2,117)  |  Arm2: ckpt_0 (1,964) at cost of massive FN
- **Lowest FN:** Arm 2 ckpt_3 (120) at MOTA=-3.97 (unusable)
- **Lowest IDSW:** Phase 5 ckpt_4 (1)  |  Arm2 ckpt_2/3 (3)

### Q4: Does highest-HOTA ckpt also have best identity metrics?
- Phase 5 ckpt_4 has best HOTA (50.51) AND best AssA (58.49) AND lowest IDSW (1). Yes for Phase 5.
- Arm 1: ckpt_3 has best HOTA and best IDF1 (70.23) but IDSW=9 (ckpt_4 is lower at 7).
- MEASURED FACT: highest-HOTA ckpt ≠ lowest-IDSW ckpt for Arm 1.

### Q5: Equal steps comparison (10,112 steps)?
- Arm 1 ckpt_3 vs Arm 2 ckpt_0 (both 10,112 steps):
  - HOTA: 50.47 vs 40.51 (+9.96 for Arm 1)
  - FN: 305 vs 1,527 (Arm 2 misses far more detections at equal steps)
  - MEASURED FACT: Arm 1 substantially outperforms Arm 2 at equal step count.

### Q6: FP and FN trends with steps?
- **Arm 1 FN:** 602→413→363→305→277 (monotonically decreasing — model finds more vehicles)
- **Arm 1 FP:** 2847→2499→2643→2629→2855 (non-monotonic, U-shaped; lowest at ckpt_1)
- **Arm 2 FN:** 1527→490→156→120→185 (large drop by ckpt_1, then near-zero — model finds almost all GT vehicles but at high FP cost)
- **Arm 2 FP:** 1964→2432→3671→4225→4266 (monotonically increasing — model over-detects heavily in later epochs)

### Q7: Which threshold results exist?
- **Threshold 0.5:** All 11 checkpoints (Phase5 + Arm1 ckpt0-4 + Arm2 ckpt0-4) — PRIMARY
- **Threshold 0.7:** Phase5 ckpt4, Arm1 ckpt4, Arm2 ckpt4 only — EXPLORATORY (ckpt4 only)
- **Threshold 0.8:** Phase5 ckpt4, Arm1 ckpt4, Arm2 ckpt4 only — EXPLORATORY (ckpt4 only)
- **Threshold 0.6:** NOT RUN for any checkpoint
- **Best checkpoints (Arm1 ckpt3, Arm2 ckpt1) at any threshold other than 0.5:** NOT RUN

### Q8: What remains uncertain?
- Run-to-run variability (σ) is unknown: both arms used seed 42, no independent repeats exist yet.
  Differences of ≤3 HOTA points are uninterpretable without σ.
- All metrics are from a single night validation sequence (744 frames). No day or dusk evaluation.
- The 4 GAP_FILL annotation candidates in seq_02 (medium confidence) remain unverified in CVAT.
- C1 module (PositionMLP) has placeholder zone boundaries (0.30/0.70) — Phase 4 calibration not run.
- C2 module (ThresholdAdapter) is trained on brightness/contrast/density only — altitude effect not modelled.
- Selecting best checkpoints or thresholds on seq_02_30sec_a is tuning on the validation sequence.

## Limits

1. **Single validation sequence** (seq_02_30sec_a, 744 frames, 50 tracks, night only)
2. **Single seed** (42) for both IOTD arms — no independent repeats
3. **One run per setting** — no ensemble or cross-validation
4. **No statistical significance** can be claimed from any comparison
5. Evaluation GT not fully audited (4 medium-confidence annotation gaps remain)
6. Confidence column constant=1 — no score-based curves possible

## Status

| Item | Status |
|---|---|
| Data CSVs (A) | CREATED |
| Figures fig01-fig08 (B) | CREATED |
| Dashboard (B) | CREATED |
| Tables table01-table06 (C) | CREATED |
| compare_results.py (D) | CREATED and tested |
| generate_all.py (D) | CREATED |
| README.md (E) | CREATED |
| figure_captions.md (E) | CREATED |
| MANIFEST.json (E) | CREATED |
| ZIP archive (E) | CREATED |
| Original GT unchanged | CONFIRMED (SHA256 identical) |
| Post-hash | PASS (3,283 files, changed=0, missing=0) |
