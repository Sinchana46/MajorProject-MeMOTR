# Guide Presentation Package — Before vs After Tuning

Created: 17 Oct 2026
Prepared for: Guide review

## What is in this folder

- figures/: six comparison figures (PNG + PDF) and one combined dashboard (PNG)
- tables/: three summary tables (CSV, LaTeX, Markdown, PNG)
- scripts/build_guide.py: rebuilds every figure and table from the source files; scripts/make_manifest.py writes MANIFEST.json
- MANIFEST.json: SHA256 of every file in this folder and every source file used

## Two protocols — important

All results in this folder under **CORRECTED** protocol use the upstream MeMOTR
output filter (score > 0.5, area > 100 px), which was missing from the project's
inference tool. All previous results (in archive/pre_tuning_baseline_v1/) used the
HISTORICAL protocol without this filter. The same models produce different numbers
under the two protocols, because the filter removes stale low-confidence tracks.

The HISTORICAL archive is preserved exactly as it was. Do not use HISTORICAL and
CORRECTED numbers in the same comparison table. (The only exceptions are fig02 and
table02, which exist to show the protocol difference and label every value.)

## Key result

Arm 1 checkpoint_3 (real IOTD fine-tuning, 10,112 steps) under the corrected
protocol:
  HOTA  64.12   (+6.4 vs Phase 5 baseline 57.74)
  DetA  60.54   IDF1  84.64   IDSW  3   FP  841   FN  488

Against the baseline: FP −10.5 % (940 → 841), FN −55.5 % (1,097 → 488).
Three training seeds of the same configuration give 64.12 / 61.07 / 59.74 at
checkpoint_3 (mean 61.64 ± 2.25); all three are above the baseline.
Single validation sequence (seq_02_30sec_a, night); not statistically established.

## What was tried and discarded

See tables/table03_what_was_tried and figures/fig05_what_was_tried.
In short: augmentation (Arm 2, best 61.61), freezing the association module
(assoc_lr0, best 62.76), halving the learning rates (lr_half, best 64.26 in a single
unstable run) and the C2 adapter (HOTA −1.05 / −1.32 / +0.79 on the three sequences)
did not pass the decision rule (noise bar 66.14).

## Sources

Every number is read from the TrackEval summary files and corrected-results CSVs
in results_after_tuning/ and results/tuning_closeout/ (and, for HISTORICAL values,
archive/pre_tuning_baseline_v1/data/results_registry_v1.csv, read only). See
MANIFEST.json for exact paths and SHA256 values. The build check is in
check/logs/guide_pres_01_build.txt (440 values checked, 0 mismatches).

## Do not edit

This folder is a presentation snapshot. For the full technical record see:
- results/tuning_closeout/tuning_narrative.md  (full narrative)
- results/tuning_closeout/tuning_completion_checklist.md  (status)
- archive/pre_tuning_baseline_v1/  (pre-tuning frozen archive)
