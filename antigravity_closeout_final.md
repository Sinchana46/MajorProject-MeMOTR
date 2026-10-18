# Antigravity Instructions: Close-out of the IOTD Augmentation Phase (final)

Written 7 Oct 2026. Read with `context__1_.md` and `context_frame_extraction_update.md`. This file **replaces** `antigravity_closeout.md` and the unfinished parts of `antigravity_phase3_augmentation.md`. Do not repeat steps already done.

---

## 0. How you must work

1. Run the parts in order. There are two fixed **GATEs** (E and H) and one conditional stop in Part A (if v1 fails the numeric rule). Stop at each and wait for the user. Everything else runs straight through, but stop and report immediately at any **STOP** condition.
2. **Paste raw printed output.** Do not summarise numbers. The user checks them.
3. Label every claim **verified** (you ran it or read the code in this session, with output or file and line) or **assumed**. Never use "expected behaviour" as a reason to skip a check.
4. If a path, number or file differs from this document, stop and report. Do not guess.
5. **Do not start training. Do not delete, rename, move or overwrite anything.** If a target path exists, stop and ask.
6. Windows and PowerShell. Set `NO_ALBUMENTATIONS_UPDATE=1` before running anything that imports albumentations (it avoids a false exit code 1 from its network version check).

---

## 1. State and decisions (all confirmed by the user, do not re-open)

| Item | Decision |
|---|---|
| Split (provisional) | Train = `seq_01_1min` + `seq_03_30sec_b`. Validation = `seq_02_30sec_a` |
| Training type | **Single-class (class-agnostic).** The loader stays as it is (every box label 0). Do **not** edit `vehiclemot.py` or add class-aware code. The class column in `gt.txt` is kept for later per-class analysis only |
| Lighting labels | `seq_01_1min` night, `seq_02_30sec_a` night, `seq_03_30sec_b` **dusk** (mean luma 17.2, 35.8, 79.4 of 255). Report the luma numbers next to the labels |
| Visibility 0.0 rows (seq_03) | Keep unchanged. Do not filter or edit |
| Manual tasks | Done: CVAT frame counts, last-frame overlays, labelling policy, backup |
| Labelling policy | Only vehicles on the monitored carriageway are annotated; opposite-carriageway and out-of-region vehicles are deliberately unannotated |
| Augmentation v1 | Exists in `data/IOTD/augmented_train/` (6 copies, 7,590 frames). Counts, dimensions, annotation hashes and real-sequence hashes passed. **Strength not yet accepted** (Part A) |
| v1 overlays | The user has inspected the v1 overlays and judged the frames natural (7 Oct). Their visual judgment stands for v1 only; any regenerated v2 needs its own inspection |
| Training data in configs | **IOTD-only for now.** Do not add the existing dataset entries. Confirmed by the user |
| Augmented copies are for | Fine-tuning (Baseline B) and C1 PositionMLP/TIM training |
| Augmented copies are NOT for | C2 adapter training, C1 calibration, validation, test, or any reported dataset size. `"use_for_c2": false` in the manifest |

---

## 2. Files you may create

New files and folders only: `check/`, `augmented_train_v2/`, `split_manifest_v2.json`, scripts in `tools/`, `data/IOTD_train_real/`, `data/IOTD_train_aug/`, `data/IOTD_val/`, and two new files in `configs/` (Part C). Everything else is read-only: all existing configs and MeMOTR code, `split_manifest.json`, the originals, and v1.

---

## Part A. Test whether v1 is acceptable (read-only, automatic decision)

**Why:** the Step 8 pixel differences are suspiciously large for the scenes. `seq_01_1min` has mean luma 17/255, but `seq_01_1min_aug_02` differs from the source by 34 on average, and `aug_01` on the same sequence by 11. Pixel difference alone cannot separate mild from destroyed, and the note "expected photometric behaviour" was never tested.

**A1. Semantics test.** Hypothesis: in albumentations 2.0.8 `RandomBrightnessContrast` with `brightness_by_max=True` **adds** `beta x 255` levels to every pixel. Apply `RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0, 0), p=1)` and the same with `(-0.1, -0.1)` to flat uint8 images of value 0, 17, 79 and 255. Print the outputs and state confirmed or not confirmed.

**A2.** Print the sampled parameters from `augmented_train/<copy>/aug_params.json` for all six copies.

**A3. Pixel statistics.** For each copy and its source, on 20 evenly spaced frames, print one table (luma = `cv2.COLOR_BGR2GRAY`):

| Copy | Source mean luma | Aug mean luma | Ratio | % pixels == 0 (source / aug) | % pixels >= 250 (source / aug) |

**A4. Full integrity of v1.** Check **every** file in all six copies: header dimensions equal the source (PIL `Image.open(...).size`) and a full `cv2.imread` decode returns a non-empty array. Print counts and list failures.

**A5. Write scope.** List every file under the project root (excluding `augmented_train/`, `check/`, `tools/`, `.git`, caches) modified after `check/hashes_before.json`. Print the list.

**Acceptance rule (fixed):** a copy passes if its aug/source mean-luma ratio is between **0.6 and 2.0** and its share of pixels equal to 0 rises by **no more than 10 percentage points** over the source. A4 must show zero failures.

**Branch:**

- All six copies pass, A4 clean: **v1 is accepted.** Skip Part B. The accepted folder is `augmented_train`.
- Any copy fails the rule: **STOP.** Print the failing rows and the A1 result, say whether you recommend keeping v1 or regenerating and why, and wait. The user replies **"keep v1"** (skip Part B) or **"regenerate"** (do Part B). The user has already judged the v1 overlays natural, so do **not** regenerate on your own.
- A4 shows any unreadable file or wrong dimension: **STOP** and report.

Print which branch you took and why.

---

## Part B. Regenerate with exposure-gain augmentation (only if the user replies "regenerate")

v1 stays untouched as `augmented_train/` and is marked rejected. The user deletes it manually later.

**B1. Preconditions.** Measure the real size of v1 on disk; require free disk of at least 1.5 times that, otherwise **STOP**. Create `tools/augment_photometric_v2.py` as a **new** file.

**B2. Implementation.** Plain OpenCV and NumPy (no albumentations) so every line is explainable. Sample each parameter **once per copy** with `np.random.default_rng(seed)` and apply identically to every frame. Per frame, in this order:

1. Exposure gain: `x = clip(gain * x, 0, 255)` (multiplicative, never an additive offset)
2. Gamma by 256-entry lookup table: `255 * (x/255) ** gamma`
3. Hue and saturation shift in HSV (OpenCV hue 0 to 179), value channel untouched
4. JPEG round trip at the sampled quality (`cv2.imencode` / `cv2.imdecode`)
5. Gaussian noise **different on every frame**: `np.random.default_rng(seed * 1_000_003 + frame_index)`, sigma on the 0 to 255 scale, then clip and cast to uint8
6. Save at JPEG quality 98, names `000001.jpg` onward

| Copy | Seed | Gain | Gamma | Hue / Sat shift | JPEG quality | Noise sigma |
|---|---|---|---|---|---|---|
| aug_01 | 1001 | 0.85 to 0.95 | 1.02 to 1.08 | +/-3 / +/-10 | 75 to 90 | 2 to 4 |
| aug_02 | 1002 | 1.10 to 1.30 | 0.92 to 0.98 | +/-5 / +/-15 | 65 to 85 | 3 to 6 |
| aug_03 | 1003 | 0.95 to 1.10 | 0.95 to 1.05 | +/-5 / +/-20 | 60 to 80 | 4 to 8 |

Write sampled values to `augmented_train_v2/<copy>/aug_params.json`. No motion blur. Do not tune these ranges yourself.

**B3. Safety.** Refuse sources listed under `val` or `test` in `split_manifest.json`; refuse to write into any existing path; no resizing (seq_03 is 1080x2336 portrait); copy `gt/gt.txt`, `gt/labels.txt`, `seqinfo.ini` byte for byte with `shutil.copyfile` and compare SHA256 with the source; support `--dry-run N`. Output: `augmented_train_v2/seq_01_1min_aug_01` ... `seq_03_30sec_b_aug_03`.

**B4. Dry run (automatic check).** Run `--dry-run 20` for all six copies into `check/dryrun_v2/`. Print the A3 table for those frames, the sampled parameters, the noise independence test (correlation of noise arrays for frame indices 1 and 2 on a flat grey image, must be under 0.05), and a reproducibility test (regenerate frame 1 of one copy twice; the two SHA256 values must be identical). If any copy fails the acceptance rule, or either test fails: **STOP** and report; do not run the full job.

**B5. Full run.** Run all six copies. Then print raw output for:

1. File count per copy (1599 or 931)
2. Header dimensions and full decode of every file
3. SHA256 equality of the three annotation files, copy vs source (18 comparisons)
4. The A3 table on 20 evenly spaced frames per copy, with pass or fail per copy
5. Re-hash of all real sequence files against `check/hashes_before.json`: changed, missing, added must all be 0
6. 18 overlays (first, middle, last frame of each copy) in `check/aug_overlays_v2/`, boxes from gt column 8
7. Totals: real train 2,530, augmented 7,590, sum 10,120

Create `split_manifest_v2.json` (never overwrite the old one): same fields as `split_manifest.json`, the six v2 copies each with `"use_for_c2": false`, and `"rejected": ["augmented_train (v1)"]`.

---

## GATE E (user)

Show the Part A output (and Part B output if it ran) and state the accepted folder (`augmented_train` or `augmented_train_v2`).

- **v1 accepted:** the user has already inspected the v1 overlays and judged them natural, so do not wait. Go straight to Part C.
- **v2 generated:** point to `check/aug_overlays_v2/` and **wait for the user to inspect those overlays and reply "approved".** Do not start Part C before that.

---

## Part C. Wire up train and validation folders

The dataloader loads **every** sequence in each folder listed in `DATASET_DIRS`, so pointing it at `data/IOTD/` would train on the validation sequence. Do not move or copy the originals.

**C1. Read-only report.** Print (file and line numbers): how validation or evaluation sequences are chosen in this repo (training-time validation, if any, and the evaluation or submit script); the existing `DATASET_DIRS` list in `configs/train_vehicle_mot_interim.yaml`; every config key that names an output or experiment directory.

**C2. Junction parents.** Using the accepted augmentation folder, create (PowerShell `New-Item -ItemType Directory`, then `New-Item -ItemType Junction -Path <link> -Target <target>`):

```
data/IOTD_train_real/  seq_01_1min, seq_03_30sec_b       (junctions to the originals)
data/IOTD_train_aug/   the six accepted augmented copies (junctions)
data/IOTD_val/         seq_02_30sec_a                     (junction)
```

**Warning:** never run `Remove-Item -Recurse` on a junction; it can delete the target's contents. Remove a junction only with `cmd /c rmdir <link>`. Print a listing of each parent, then re-hash the originals to confirm nothing changed.

**C3. Two new config files**, copied from `configs/train_vehicle_mot_interim.yaml` (never edit the original):

- `configs/train_iotd_real.yaml`: `DATASET_DIRS: ["IOTD_train_real"]`
- `configs/train_iotd_real_aug.yaml`: `DATASET_DIRS: ["IOTD_train_real", "IOTD_train_aug"]`

These are **IOTD-only, confirmed by the user for now** (a clean with/without-augmentation ablation). Do not add the existing dataset entries; still print them in C1 for the record. Change only `DATASET_DIRS` and the output or experiment names, so nothing overwrites an existing run. If the config has a validation or evaluation key, point it at `IOTD_val`; if it has none, say so and leave it. Print a line-by-line diff of each new config against the original. The two new configs must differ from each other only in `DATASET_DIRS` and the output name.

**C4. Dataset dry check (no training).** With the repo's own dataset builder, for each new config print: sequences loaded (names), number of clip start frames, and confirmation that `seq_02_30sec_a` is **absent**. Fetch 20 training samples per config (include portrait `seq_03_30sec_b`), print tensor shapes and box counts, and confirm every label is 0. Save 6 visualised samples (image after transforms, boxes drawn) to `check/dataset_samples/`. Report any error verbatim. If the validation sequence appears in either training list: **STOP**.

---

## Part D. Records and handoff

**D1. Visibility, information only.** Per sequence, the count and percentage of rows with visibility 0.0 and the distinct values in column 9.

**D2. Effective resize table.** Using the repo's own resize function with the training scales and `max_size=1536` (call it on dummy sizes, read-only), print the resulting width x height at each scale for 1920x1080 and 1080x2336, and the scale from which the long-side cap applies for each.

**D3. Dataset facts.** Per sequence: rows, distinct track IDs, boxes per class; the real train totals (tracks, frames); the gap to the spec's 500 or more tracks. Brightness per sequence: mean luma, plus mean and standard deviation of luma across sampled frames, next to the confirmed labels (night, night, dusk).

**D4. Write `check/phase3_final_summary.md`** with verified facts only: paths, the accepted augmentation folder and parameters, the pixel-statistics table, all verification results, the junction layout, the two config names, D1 to D3, and a list of items still open: validation procedure (from C1), the existing dataset entries versus IOTD-only, 30 fps vs 24 fps with the same sampling interval, no day-lighting IOTD sequence yet, and track count below the spec. Do not edit any existing context file.

End with: "Augmented frames are training samples, not additional annotated data. Real annotated data: 3,274 frames."

---

## GATE H (user)

Stop after D4. Print the path of the summary and wait.

---

## Standing rules

Split by sequence, never by frame. Never touch test data before the final evaluation. Do not modify the MeMOTR backbone or encoder. Changes that alter default fixed-lambda or fixed-tau behaviour need a feature flag. Report more than MOTA. Keep only code the user can explain line by line.
