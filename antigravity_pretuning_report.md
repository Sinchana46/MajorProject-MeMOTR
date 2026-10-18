# Antigravity Instructions: Pre-Tuning Results Package (figures, tables, frozen registry)

Written 11 Oct 2026. Goal: turn the **existing** results into (1) paper-ready figures, (2) result tables in CSV, LaTeX and Markdown, (3) one combined dashboard image, and (4) a **frozen results registry** that later tuning results can be compared against. Everything is saved in one folder.

**Nothing is retrained and no new inference or GPU work is done.** Use only existing predictions, TrackEval summaries, logs and configs. The `seq_02_30sec_a` validation annotations have been reviewed by the user and confirmed as proper.

---

## 0. Rules

1. **Do not train. Do not run inference. Do not use the GPU.** Do not delete, move, rename or overwrite anything. Do not modify: datasets, any `gt.txt`, `labels.txt`, `seqinfo.ini`, CVAT data, checkpoints, configs, code, existing predictions, existing results or reports.
2. Write **only** under `results/pre_tuning_baseline_v1/` (new). If that folder already exists, stop and ask. Layout:
   ```
   results/pre_tuning_baseline_v1/
     README.md                 what is here, how it was made, protocol, caveats
     MANIFEST.json             SHA256 of every output file and every source file used
     figures/                  fig01 ... fig08 as PNG (300 dpi) and PDF (vector), plus dashboard
     tables/                   table01 ... table06 as .csv, .tex, .md
     data/                     results_registry.csv, per_checkpoint_metrics.csv, detection_outcomes.csv
     scripts/                  generate_all.py, compare_results.py
     figure_captions.md        draft captions, measured facts only
   ```
   Also create `results/pre_tuning_baseline_v1.zip` of the finished folder (new file).
3. **No hard-coded results.** Every number is read from the saved files and carries its source path and SHA256 in `results_registry.csv`. The numbers listed in section 2 are **claims to verify**: print any mismatch and use the file value. Missing data is shown as missing, never estimated.
4. Run each step as its own script with `python -u`, output to `check/logs/pretune_<step>.txt`, hard timeout (300 s), no polling loops, no echo or latency tests, no reading `transcript.jsonl`. Retry a failed command at most once, then report the error and continue with the next independent step. If a needed Python package is missing, **report it instead of installing**.
5. Paste raw output. Label claims **verified** (you read the file or ran the check) or **assumed**. Do not write "PASS" without evidence.
6. Re-hash all real IOTD files against `check/hashes_before.json` at the start and end (thread pool): `changed`, `missing`, `added` must be 0.

---

## 1. Protocol (fixed for the whole package)

- Validation sequence `seq_02_30sec_a` only (744 frames, 50 tracks, night, mean luma 35.6), detection and tracking thresholds 0.5, miss tolerance 30, `use_c1=False`, `use_c2=False`, TrackEval with the evaluation-only GT (all classes set to 1).
- **Primary scoring = S0** (the original GT, which reproduces the historical numbers). If the corrected-GT scoring **S1** exists (`results/eval_iotd_val/rescored_v2/`), include it as extra columns in the registry and in a supplementary table, never mixed with S0. Record the SHA256 of the GT file each scoring used.
- Single seed (42) for both arms; one run per setting; one validation sequence. State this on every figure (footer text, see section 4) and in the README. Do not claim statistical significance.
- The tracker confidence column is a constant 1, so no score-based precision-recall curves can be drawn. Do not draw any.

---

## 2. Experiments to include (claims to verify against the files)

Naming used everywhere: **Phase 5 baseline** (UA-DETRAC + VisDrone, no IOTD data), **Arm 1 (real only)**, **Arm 2 (real + photometric aug.)**.

| Run | Checkpoint | Steps | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 5 | checkpoint_4 | 0 | 50.51 | 43.88 | 58.49 | 29.10 | 68.84 | 1 | 2,117 | 847 |
| Arm 1 | ckpt_0 | 2,528 | 45.72 | 42.49 | 49.58 | 17.03 | 63.08 | 21 | 2,847 | 602 |
| Arm 1 | ckpt_1 | 5,056 | 49.30 | 46.42 | 52.76 | 30.03 | 68.38 | 14 | 2,499 | 413 |
| Arm 1 | ckpt_2 | 7,584 | 48.87 | 45.93 | 52.27 | 27.83 | 68.43 | 12 | 2,643 | 363 |
| Arm 1 | ckpt_3 | 10,112 | 50.47 | 46.96 | 54.51 | 29.63 | 70.23 | 9 | 2,629 | 305 |
| Arm 1 | ckpt_4 | 12,640 | 49.58 | 45.70 | 54.04 | 24.94 | 69.66 | 7 | 2,855 | 277 |
| Arm 2 | ckpt_0 | 10,112 | 40.51 | 36.49 | 45.15 | 16.28 | 56.63 | 10 | 1,964 | 1,527 |
| Arm 2 | ckpt_1 | 20,224 | 48.17 | 45.54 | 51.14 | 29.84 | 68.64 | 12 | 2,432 | 490 |
| Arm 2 | ckpt_2 | 30,336 | 44.77 | 41.22 | 48.77 | 8.42 | 67.13 | 3 | 3,671 | 156 |
| Arm 2 | ckpt_3 | 40,448 | 43.41 | 39.18 | 48.21 | -3.97 | 64.56 | 3 | 4,225 | 120 |
| Arm 2 | ckpt_4 | 50,560 | 42.84 | 38.32 | 48.01 | -6.58 | 63.72 | 6 | 4,266 | 185 |

Also verify and use these run facts (from logs and configs, not from file names): Arm 1 = `configs/train_iotd_real.yaml`, 2,528 steps per epoch, 5 epochs; Arm 2 = `configs/train_iotd_real_aug.yaml`, 10,112 steps per epoch, 5 epochs; both start from Phase 5 `checkpoint_4` (weights only), seed 42, batch 1, LR 2e-5 for epochs 0-3 and 2e-6 for epoch 4 (the drop happens after 10,112 steps for Arm 1 and after 40,448 for Arm 2), backbone frozen, clip length 2 throughout, about 0.664 s per step, Arm 1 about 2h20m and Arm 2 about 9h25m. A checkpoint saved at the end of epoch N is a completed epoch; label epochs as N+1 completed epochs.

**Existing threshold results** (S0, final checkpoints only, exploratory): thresholds 0.7 and 0.8 exist for Phase 5, Arm 1 ckpt_4 and Arm 2 ckpt_4 (for example HOTA 41.63 / 48.06 / 47.46 at 0.7 and 37.15 / 40.63 / 48.14 at 0.8). **0.6 has not been run**, and no threshold sweep exists for the best checkpoints (Arm 1 ckpt_3, Arm 2 ckpt_1). Report this status exactly; do not run anything to fill it.

---

## 3. Step A: build the data (read-only)

1. Locate the TrackEval summary file for every row of section 2 (the folders under `results/eval_iotd_val/`: `trackeval_*`, the `arm{1,2}_ckpt{0-3}` runs, `diag/`, `rescored_v2/`). Print a table: run, checkpoint, summary path, tracker-output path, row count, SHA256 of the summary. Print every mismatch against the claims above (tolerance 0.01).
2. Build `data/per_checkpoint_metrics.csv` with columns: run, checkpoint, steps, epochs_completed, seed, config, scoring (S0 or S1), threshold, GT_sha256, HOTA, DetA, AssA, MOTA, IDF1, IDSW, TP, FP, FN, recall, precision, summary_path, summary_sha256. TP, FP and FN come from the TrackEval CLEAR counts; precision and recall are derived.
3. Build `data/results_registry.csv` as the **frozen baseline registry**: one row per run, checkpoint, scoring and threshold, with the same columns plus `registry_version = v1` and `created_utc`. This is the file later tuning runs are compared against.
4. **Detection outcomes (own matching).** For the three candidates (Phase 5 checkpoint_4, Arm 1 best-HOTA checkpoint, Arm 2 best-HOTA checkpoint) read the saved tracker outputs and the evaluation GT and match per frame, one to one (Hungarian on IoU, IoU >= 0.5). Count TP, FP, FN, precision, recall. Write `data/detection_outcomes.csv` and print how these counts differ from TrackEval's CLEAR counts (the two matchings are not identical; state the difference, which should be small).

Select the best checkpoint **per arm by HOTA at det/track 0.5** (the written rule), and separately report the lowest-FP, lowest-FN and lowest-IDSW checkpoint over all saved checkpoints. They need not coincide.

---

## 4. Step B: figures (paper-ready)

Style for all figures: matplotlib, Okabe-Ito colour-blind-safe palette, **the same colour and marker for each run in every figure** (Phase 5 grey, Arm 1 blue, Arm 2 vermilion), font at least 8 pt at final size, figure width 3.4 in (single column) or 7.0 in (double column) as noted, 300 dpi PNG **and** vector PDF, no chart junk, value labels where they fit, axes with units. Every figure gets this footer in small type: *"Validation: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42. Thresholds 0.5. Not statistically established."* Do not draw titles inside the image that duplicate the caption.

| Figure | Content |
|---|---|
| fig01_hota_vs_steps | HOTA against optimizer steps, one line per arm, Phase 5 as a horizontal dashed reference, value labels on every checkpoint, best checkpoint of each arm highlighted, the learning-rate drop marked for each arm (after 10,112 and 40,448 steps), and the equal-step pair (Arm 1 ckpt_3 and Arm 2 ckpt_0, both 10,112 steps) connected and annotated. Secondary top axis in epochs is optional. 7.0 in wide |
| fig02_deta_assa_vs_steps | Two stacked subplots, DetA and AssA against steps for both arms, Phase 5 reference lines. 3.4 in wide |
| fig03_fp_fn_vs_steps | FP and FN counts against steps for both arms (two subplots or a shared axis), Phase 5 as reference lines, plus precision and recall as a second pair of subplots. Caption states only what is measured |
| fig04_detection_outcomes | For the three candidates, a **detection-outcome summary** as a 2 x 2 grid per model: TP, FN, FP, and the fourth cell greyed and labelled "TN not defined (single-class detection)". Counts and precision and recall printed beside each. Do not invent classes or a TN value. 7.0 in wide |
| fig05_candidate_comparison | Grouped bars for HOTA, DetA, AssA, MOTA and IDF1 for the three candidates, actual values labelled, MOTA allowed to go below zero (separate panel or clearly labelled axis). No normalised score that could be mistaken for the real metric |
| fig06_identity_metrics | IDSW (lowest highlighted), AssA and IDF1 for the three candidates, with a note that the checkpoint with the best detection need not have the best association |
| fig07_epochwise_summary | HOTA of every saved checkpoint as grouped bars (Arm 1 and Arm 2 side by side by epoch index), best of each arm highlighted, plus a compact table under the bars with DetA, AssA, MOTA, IDF1 |
| fig08_qualitative | **Optional, only if the frames and tracker outputs load.** Frames 298 and 447 of `seq_02_30sec_a`, three columns (Phase 5, Arm 1 best, Arm 2 best), GT in green and tracker boxes in red, no region polygon, no overlaid text except column headers. 7.0 in wide |
| pre_tuning_dashboard | One combined image of figures 01 to 07 in a labelled grid (A to G), including a status box: best baseline, best Arm 1, best Arm 2, threshold sweep status as stated in section 2, and the missing items needed before tuning. About 24 x 16 in, 200 dpi |

Write `figure_captions.md` with a draft caption for each figure that states **only measured facts** and the limits (one sequence, one seed). Keep interpretation out of captions.

---

## 5. Step C: tables

Write each table as `.csv`, `.tex` (booktabs, best value per column in bold, a caption and a label) and `.md`:

| Table | Content |
|---|---|
| table01_all_checkpoints | The 11 rows of section 2 with steps, epochs completed, HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, recall, precision |
| table02_selected_candidates | Phase 5, Arm 1 best, Arm 2 best, with each metric and the **difference relative to Phase 5**; best per metric marked |
| table03_detection_outcomes | TP, FP, FN, precision, recall from the own matching, beside TrackEval's CLEAR counts and the difference |
| table04_threshold_exploratory | The existing 0.5 / 0.7 / 0.8 results for the three ckpt_4 runs, marked **exploratory**, with 0.6 shown as "not run" |
| table05_run_inventory | Run, config path, start checkpoint, seed, data folders, steps per epoch, epochs, LR schedule, backbone frozen, time per step, wall time |
| table06_dataset_summary | Per sequence: frames, GT rows, distinct tracks, boxes per class, mean luma and its label (night, night, dusk), role (train, validation, train). State: real annotated frames 3,274 in total (1,599 + 744 + 931), of which 2,530 real training frames; augmented frames are training samples, not additional annotated data |

---

## 6. Step D: comparison tool for later results

Write `scripts/compare_results.py` (CPU only, no installs). Given a folder of new TrackEval summary files (or a CSV in the registry's format) it must:

1. read the new metrics, check that scoring, threshold, GT hash and sequence match the registry (print a warning for any mismatch so unlike protocols are never compared silently)
2. print a table with the new value, the registry value and the difference for HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, for the Phase 5 baseline, Arm 1 best and Arm 2 best, and for the same step count where available
3. append nothing to the frozen registry; write the comparison to a new CSV in a folder given on the command line

Test it on the existing 0.7 and 0.8 results and show the output.

`scripts/generate_all.py` must recreate every figure, table and CSV from the saved evaluation data in one command. Run it once from a clean state and show it reproduces the files (compare hashes of the CSVs).

---

## 7. Step E: README, manifest, and answers

`README.md` must state the protocol, the exact sources used, how to regenerate, the list of limits, and "single validation sequence, single seed, one run per setting; no statistical significance". `MANIFEST.json` lists the SHA256 of every output and every source file.

Answer in the README, with measured results separate from hypotheses:

1. Did real-only IOTD fine-tuning improve HOTA over Phase 5 (best checkpoint and last checkpoint)?
2. Did real + augmented fine-tuning improve HOTA over Phase 5?
3. Which checkpoint has the highest HOTA, the lowest FP, the lowest FN and the lowest IDSW?
4. Does the highest-HOTA checkpoint also have the best identity metrics (AssA, IDF1, IDSW)?
5. At equal steps (10,112), how do Arm 1 ckpt_3 and Arm 2 ckpt_0 compare?
6. What trend do FP and FN show against steps for each arm?
7. Which threshold results exist and which are still missing?
8. What remains uncertain and why?

Finish with a status table: each figure and table **CREATED** or **MISSING** with the reason, and the verification results (value mismatches, hash re-check of the originals).

Then stop. Do not start tuning, seed repeats, C1 or C2, or any training.
