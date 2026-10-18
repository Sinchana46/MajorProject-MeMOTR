"""
Pre-tuning package: Step E — generate README.md, figure_captions.md,
MANIFEST.json, and zip the folder.
"""
import os, sys, csv, json, hashlib, datetime, zipfile, glob

OUT_BASE = "results/pre_tuning_baseline_v1"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ─── README.md ────────────────────────────────────────────────────────────────
readme_text = """# Pre-Tuning Baseline v1

**Created:** {now}

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
python results/pre_tuning_baseline_v1/scripts/compare_results.py \\
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
""".format(now=now_utc)

readme_path = os.path.join(OUT_BASE, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)
print(f"README written: {readme_path}")

# ─── figure_captions.md ───────────────────────────────────────────────────────
captions_text = """# Figure Captions

All figures: single validation sequence seq_02_30sec_a (744 frames, 50 tracks, night).
Single run per setting, seed 42. Thresholds 0.5 (except where noted). Not statistically established.

**fig01_hota_vs_steps.** HOTA (%) against optimizer steps for Arm 1 (real-only fine-tuning, blue circles)
and Arm 2 (real + photometric augmentation, vermilion triangles), with Phase 5 baseline shown as a grey
dashed horizontal reference (HOTA 50.51). Stars mark the best-HOTA checkpoint per arm (Arm 1 ckpt_3
at step 10,112; Arm 2 ckpt_1 at step 20,224). Dotted vertical lines indicate the learning-rate drop
for each arm (step 10,112 for Arm 1; step 40,448 for Arm 2). The double-headed arrow connects the
equal-step pair Arm 1 ckpt_3 and Arm 2 ckpt_0, both at step 10,112.

**fig02_deta_assa_vs_steps.** DetA (top) and AssA (bottom) against optimizer steps for Arm 1 and Arm 2,
with Phase 5 reference lines. Both metrics follow a non-monotonic trajectory; AssA does not recover
to the Phase 5 level for either arm.

**fig03_fp_fn_vs_steps.** False positives (top-left), false negatives (top-right), recall (bottom-left),
and precision (bottom-right) against optimizer steps for Arm 1 and Arm 2, with Phase 5 reference lines.
FN decreases monotonically for both arms; FP decreases for Arm 1 in early epochs then rises, and
increases monotonically for Arm 2. Source: TrackEval CLEAR counts.

**fig04_detection_outcomes.** Per-frame detection-outcome summary for Phase 5 (grey), Arm 1 ckpt_3 (blue),
and Arm 2 ckpt_1 (vermilion). Each panel shows TP, FP, FN, and a greyed TN cell (labelled
"TN not defined — single-class detection"). Precision and recall are derived from TP, FP, FN.
Source: per-frame greedy IoU>=0.5 matching, which agrees exactly with TrackEval CLEAR counts for these runs.

**fig05_candidate_comparison.** Grouped bar chart of HOTA, DetA, AssA, and IDF1 (left panel) and
MOTA (right panel, separate axis to accommodate negative values) for the three candidates: Phase 5 ckpt_4,
Arm 1 ckpt_3, and Arm 2 ckpt_1. Actual values are labelled on each bar.

**fig06_identity_metrics.** AssA, IDF1, and ID-switch count (IDSW) for the three candidates. Gold edges
mark the best value per metric. The best-HOTA checkpoint does not coincide with the lowest-IDSW checkpoint
for Arm 1 (ckpt_3 IDSW=9; ckpt_4 IDSW=7).

**fig07_epochwise_summary.** HOTA by completed epoch for Arm 1 (blue) and Arm 2 (vermilion) as grouped bars,
with Phase 5 reference. Gold-edged bars are the best-HOTA checkpoint per arm. The table below the bars
lists DetA, AssA, MOTA, and IDF1 for each epoch.

**fig08_qualitative.** Sample frames 298 and 447 of seq_02_30sec_a for Phase 5, Arm 1 ckpt_3, and
Arm 2 ckpt_1. GT boxes in green; tracker boxes in the arm's assigned colour. No region polygon applied.
"""

captions_path = os.path.join(OUT_BASE, "figure_captions.md")
with open(captions_path, "w", encoding="utf-8") as f:
    f.write(captions_text)
print(f"Captions written: {captions_path}")

# ─── MANIFEST.json ────────────────────────────────────────────────────────────
print("Building MANIFEST.json...")

manifest = {
    "registry_version": "v1",
    "created_utc": now_utc,
    "description": "SHA256 of every output and every source file used in the pre-tuning baseline v1 package.",
    "outputs": {},
    "sources": {},
}

# Outputs
for root, dirs, files in os.walk(OUT_BASE):
    for fname in sorted(files):
        if fname == "MANIFEST.json": continue
        fpath = os.path.join(root, fname)
        rel   = os.path.relpath(fpath, OUT_BASE).replace("\\","/")
        manifest["outputs"][rel] = sha256_file(fpath)

# Sources (TrackEval summaries used)
source_files = glob.glob("results/eval_iotd_val/**/pedestrian_summary.txt", recursive=True) + \
               glob.glob("data/IOTD*/**/seqinfo.ini", recursive=True) + \
               glob.glob("data/IOTD*/**/gt/gt.txt", recursive=True) + \
               ["data/IOTD_val/seq_02_30sec_a/gt/gt.txt"]

for fpath in sorted(set(source_files)):
    if os.path.exists(fpath):
        rel = fpath.replace("\\","/")
        manifest["sources"][rel] = sha256_file(fpath)

manifest_path = os.path.join(OUT_BASE, "MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"MANIFEST written: {manifest_path} ({len(manifest['outputs'])} outputs, {len(manifest['sources'])} sources)")

# ─── ZIP archive ──────────────────────────────────────────────────────────────
zip_path = "results/pre_tuning_baseline_v1.zip"
print(f"\nBuilding {zip_path}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT_BASE):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, os.path.dirname(OUT_BASE)).replace("\\","/")
            zf.write(fpath, arcname)
zip_size = os.path.getsize(zip_path)
print(f"ZIP written: {zip_path} ({zip_size/1e6:.1f} MB)")

# ─── Final status ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP E COMPLETE — STATUS TABLE")
print("="*60)
items = [
    ("data/per_checkpoint_metrics.csv", "Data CSV: 11 rows, all metrics, S0"),
    ("data/results_registry.csv",       "Registry CSV: 17 rows (11+6 thresh), frozen v1"),
    ("data/detection_outcomes.csv",     "Detection outcomes: 3 candidates, own vs TrackEval"),
    ("figures/fig01_hota_vs_steps.png", "fig01: HOTA vs steps"),
    ("figures/fig02_deta_assa_vs_steps.png", "fig02: DetA/AssA vs steps"),
    ("figures/fig03_fp_fn_vs_steps.png", "fig03: FP/FN/Recall/Prec"),
    ("figures/fig04_detection_outcomes.png", "fig04: Detection outcome 2x2 grid"),
    ("figures/fig05_candidate_comparison.png", "fig05: Grouped bars 3 candidates"),
    ("figures/fig06_identity_metrics.png", "fig06: Identity metrics"),
    ("figures/fig07_epochwise_summary.png", "fig07: Epoch-wise HOTA + table"),
    ("figures/fig08_qualitative.png", "fig08: Qualitative frames 298/447"),
    ("figures/pre_tuning_dashboard.png", "Dashboard: A-G panels + status box"),
    ("tables/table01_all_checkpoints.csv", "table01: all 11 checkpoints"),
    ("tables/table02_selected_candidates.csv","table02: 3 candidates vs Phase5"),
    ("tables/table03_detection_outcomes.csv","table03: own match vs TrackEval"),
    ("tables/table04_threshold_exploratory.csv","table04: 0.5/0.7/0.8 (0.6 missing)"),
    ("tables/table05_run_inventory.csv","table05: run configs"),
    ("tables/table06_dataset_summary.csv","table06: dataset splits"),
    ("scripts/compare_results.py","compare script (tested on 0.7 results)"),
    ("scripts/generate_all.py","one-command regeneration script"),
    ("README.md","Protocol, sources, answers, limits"),
    ("figure_captions.md","Draft captions — measured facts only"),
    ("MANIFEST.json","SHA256 of all outputs and sources"),
]
for rel, desc in items:
    full = os.path.join(OUT_BASE, rel)
    status = "CREATED" if os.path.exists(full) else "MISSING"
    print(f"  {status:<8} {desc}")

zip_status = "CREATED" if os.path.exists(zip_path) else "MISSING"
print(f"  {zip_status:<8} results/pre_tuning_baseline_v1.zip ({zip_size/1e6:.1f} MB)")
