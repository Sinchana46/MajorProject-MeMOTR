# Pre-tuning baseline v1 — frozen archive

Frozen, read-only snapshot of every evaluation made before hyperparameter tuning of the MeMOTR IOTD fine-tune,
plus the three-seed spread (Tuning Stage 1), the inference-time threshold sweep and the C2 evaluation.
Created 2026-10-16. Nothing in `results/` was moved or changed; this folder only holds copies and derived files.
Every file is listed with its SHA256 in `MANIFEST.json`, together with the SHA256 of every project file it was
generated from. All files are set read-only.

## Contents

| Path | What |
|---|---|
| `figures/fig01…fig09` (`.png` 300 dpi + `.pdf`) | Paper figures, Okabe-Ito colours (Phase 5 grey, Arm 1 blue, Arm 2 vermilion), no in-image titles |
| `figures/dashboard.png` | Figures 1–9 as panels A–I plus a status box (150 dpi, 24 × 20 in) |
| `tables/table01…table07` (`.csv`, `.tex`, `.md`, `.png`) | Tables; `.tex` uses booktabs, best value per column in bold |
| `data/results_registry_v1.csv` | Every evaluated run: run, checkpoint, steps, seed, config, sequence, scoring, thresholds, miss tolerance, C2, GT SHA256, all metrics, source summary path and SHA256 |
| `data/detection_outcomes.csv` | TP/FP/FN from own per-frame Hungarian matching (IoU ≥ 0.5) for Phase 5, Arm 1 ckpt_3, Arm 2 ckpt_1, next to TrackEval's CLEAR counts |
| `figure_captions.md` | One draft caption per figure, measured facts only |
| `scripts/generate_all.py` | Regenerates every figure, table and data file above from the saved evaluation files |
| `scripts/compare_results.py` | Compares new TrackEval summaries or a registry-format CSV with the frozen registry; refuses to write inside `archive/` |
| `scripts/make_manifest.py`, `scripts/verify_archive.py` | Write / verify `MANIFEST.json` |

## Protocol (fixed for every number here)

- **Evaluation sequence:** `seq_02_30sec_a` (744 frames, 50 tracks, night, 1920×1080). Task 2 (C2) additionally uses
  `seq_01_1min` (1,599 frames, night) and `seq_03_30sec_b` (931 frames, dusk, portrait 1080×2336), both IOTD
  training sequences, with Phase 5 weights only.
- **Inference:** `tools/run_inference_mot.py`, config `configs/train_iotd_real.yaml`, det/track threshold 0.5 and miss
  tolerance 30 unless a row says otherwise, `use_c1=False`, `use_c2=False` unless a row says C2 on.
- **Scoring S0:** original `gt/gt.txt` (seq_02 SHA256 `17ebe278…d3472`), evaluation-only copy with every class set to 1,
  TrackEval via `tools/run_trackeval.py` (HOTA, CLEAR, Identity; `DO_PREPROC=False`, so every unmatched prediction is
  a false positive). No S1 (corrected GT) scoring exists.
- **Best-checkpoint rule:** highest HOTA among the five checkpoints at det/track 0.5.
- **Noise yardstick:** three seeds (42, 1234, 2024) of the Arm 1 configuration. A difference counts only if it exceeds
  the seed mean by more than 2 seed SD (n = 3, a rough estimate).
- The tracker output's confidence column is a constant 1, so score-based curves are impossible; every threshold change
  is a new inference run.

## Sources

| Source | Used for |
|---|---|
| `results/pre_tuning_baseline_v1/data/results_registry.csv` and the TrackEval summaries it points to | Phase 5, Arm 1, Arm 2 (5 checkpoints each) and the exploratory 0.7 / 0.8 thresholds |
| `results/tuning_stage1/stage1_results.csv` (+ summaries) | seed1, seed2, assoc_lr0, lr_half |
| `results/inference_sweep/sweep_results.csv` (+ summaries) | 7 inference settings on Arm 1 ckpt_3 |
| `results/c2_eval/c2_results.csv` (+ summaries, `tau_per_frame.csv`) | C2 off/on on three sequences |
| `results/eval_iotd_val/{phase5,arm1_ckpt3,arm2_ckpt1}/seq_02_30sec_a.txt` | Detection outcomes |
| `data/IOTD/seq_profile.csv`, `gt/gt.txt`, `seqinfo.ini` of each sequence | Dataset summary |

`generate_all.py` re-reads every metric from its TrackEval summary file and aborts if any value differs from its
source CSV by more than 0.01. One correction relative to the source registry: the exploratory 0.7 / 0.8 threshold rows
for Phase 5 listed 12,640 steps; Phase 5 has 0 IOTD steps, and the archive registry says 0.

## How to regenerate

From the project root, with the `memotr_project` environment (Python 3.9.25, matplotlib, scipy):

```bash
python archive/pre_tuning_baseline_v1/scripts/generate_all.py --out D:/tmp/regen
python archive/pre_tuning_baseline_v1/scripts/verify_archive.py --sources
```

The archive itself is read-only, so regenerate into another folder (`--out`) and compare.
To compare new results with the frozen registry:

```bash
python archive/pre_tuning_baseline_v1/scripts/compare_results.py --new results/inference_sweep/sweep_results.csv \
       --ref-run Arm1 --ref-ckpt ckpt_3 --ref-thresh 0.5 --out D:/tmp/diff.csv
```

## Limitations

1. **One validation sequence** (seq_02, 744 frames, night). No day evaluation; dusk only through the C2 check on a
   training sequence.
2. **Single run per variant.** Only the Arm 1 configuration has three seeds; SD from n = 3 is itself uncertain.
3. **Tuning on the validation sequence.** Best checkpoints, the recommended inference setting and any later choice are
   selected on the same sequence used for scoring, so they are optimistic.
4. **No statistical significance** is claimed anywhere.
5. Real annotated data: 3,274 frames (1,599 + 744 + 931); 2,530 real training frames. Augmented frames are training
   samples, not additional annotated data.
6. Four medium-confidence annotation gaps in seq_02 remain unverified.
7. `tools/run_inference_mot.py` writes a box for every track with id ≥ 0, including tracks inside their miss-tolerance
   window, and applies no result-score filter (upstream MeMOTR's `submit_engine.py` filters with
   `RESULT_SCORE_THRESH`). This affects every number here equally and explains why miss tolerance changes FP so much
   in the sweep.
8. The C2 adapter was trained on UA-DETRAC and VisDrone, not on IOTD; its brightness-to-τ mapping is not calibrated to
   this camera.
