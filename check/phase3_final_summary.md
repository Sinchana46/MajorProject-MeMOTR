# Phase 3 Final Summary & Close-Out

**Generated:** 2026-10-07  
**Status:** Verification Complete — Handoff at Gate H  

---

## 1. Directory & File Paths

- **Workspace Root:** `D:\MajorProject-MeMOTR`
- **Original IOTD Data:** `D:\MajorProject-MeMOTR\data\IOTD`
- **Accepted Augmentation Directory:** `data/IOTD/augmented_train_v2`
- **Rejected Augmentation Directory (preserved untouched on disk):** `data/IOTD/augmented_train`
- **Split Manifests:**
  - Original: `data/IOTD/split_manifest.json` (untouched)
  - V2 Manifest: `data/IOTD/split_manifest_v2.json`
- **New Training Configs:**
  - `configs/train_iotd_real.yaml` (Real train sequences only: `["IOTD_train_real"]`)
  - `configs/train_iotd_real_aug.yaml` (Real + Augmented: `["IOTD_train_real", "IOTD_train_aug"]`)
- **Junction Parents:**
  - `data/IOTD_train_real/` -> `seq_01_1min`, `seq_03_30sec_b` (junctions to originals)
  - `data/IOTD_train_aug/` -> `seq_01_1min_aug_01..03`, `seq_03_30sec_b_aug_01..03` (junctions to v2)
  - `data/IOTD_val/` -> `seq_02_30sec_a` (junction to original)

---

## 2. Accepted Augmentation Parameters (v2)

Pure OpenCV/NumPy multiplicative exposure gain pipeline:
- `x_gain = clip(gain * x, 0, 255)`
- `gamma = 255 * (x / 255) ** gamma` (LUT)
- Hue shift & Saturation shift in HSV
- JPEG round-trip encoding/decoding
- Frame-independent Gaussian noise: `seed * 1_000_003 + frame_idx`

| Copy | Source | Seed | Gain | Gamma | Hue Shift | Sat Shift | JPEG Quality | Noise Sigma |
|---|---|---|---|---|---|---|---|---|
| `seq_01_1min_aug_01` | `seq_01_1min` | 1001 | 0.9113 | 1.0209 | +0 | -7 | 86 | 2.15 |
| `seq_01_1min_aug_02` | `seq_01_1min` | 1002 | 1.1762 | 0.9414 | +4 | +8 | 77 | 4.01 |
| `seq_01_1min_aug_03` | `seq_01_1min` | 1003 | 0.9782 | 0.9757 | -2 | +0 | 77 | 7.01 |
| `seq_03_30sec_b_aug_01` | `seq_03_30sec_b` | 1001 | 0.9113 | 1.0209 | +0 | -7 | 86 | 2.15 |
| `seq_03_30sec_b_aug_02` | `seq_03_30sec_b` | 1002 | 1.1762 | 0.9414 | +4 | +8 | 77 | 4.01 |
| `seq_03_30sec_b_aug_03` | `seq_03_30sec_b` | 1003 | 0.9782 | 0.9757 | -2 | +0 | 77 | 7.01 |

---

## 3. Pixel Statistics Table (Part B5 Check 4)

Measured across 20 evenly-spaced frames per copy:

| Copy | Source Mean Luma | Aug Mean Luma | Ratio | % Pix == 0 (src / aug) | % Pix >= 250 (src / aug) | Pass/Fail |
|---|---|---|---|---|---|---|
| `seq_01_1min_aug_01` | 17.10 | 13.71 | 0.802 | 0.01% / 4.22% | 0.13% / 0.00% | PASS |
| `seq_01_1min_aug_02` | 17.10 | 19.83 | 1.159 | 0.01% / 0.80% | 0.13% / 0.08% | PASS |
| `seq_01_1min_aug_03` | 17.10 | 16.88 | 0.987 | 0.01% / 2.36% | 0.13% / 0.03% | PASS |
| `seq_03_30sec_b_aug_01` | 78.93 | 69.80 | 0.884 | 0.00% / 0.03% | 0.36% / 0.00% | PASS |
| `seq_03_30sec_b_aug_02` | 78.93 | 94.72 | 1.200 | 0.00% / 0.00% | 0.36% / 0.36% | PASS |
| `seq_03_30sec_b_aug_03` | 78.93 | 77.07 | 0.976 | 0.00% / 0.03% | 0.36% / 0.08% | PASS |

- **Noise Independence Test (B4):** `corr = 0.000300 < 0.05` (PASS)
- **Reproducibility Test (B4):** Frame 1 SHA256 identical on duplicate runs (PASS)

---

## 4. Verification Results Summary

1. **Check 1 (File Counts):** All 6 copies match expected (1599, 1599, 1599, 931, 931, 931) — PASS
2. **Check 2 (Header Dims & Full cv2 Decode):** 7,590 files checked, 0 failures — PASS
3. **Check 3 (Annotation SHA256):** 18/18 comparisons match source files byte for byte — PASS
4. **Check 4 (A3 Luma Table):** All 6 copies pass numeric criteria (ratio in [0.6, 2.0], delta %pix==0 <= 10pp) — PASS
5. **Check 5 (Re-hash Originals):** vs `hashes_before.json`: `changed=0, missing=0, added=0` — PASS
6. **Check 6 (18 Overlays):** Saved to `check/aug_overlays_v2/`, 18 OK, 0 Fail — PASS
7. **Check 7 (Totals):** Real train 2,530, Augmented train 7,590, Total train samples 10,120, Real val 744 — PASS
8. **Junction Verification (C2):** All 9 junctions created, verified resolving, real sequences re-hashed (`changed=0, missing=0, added=0`) — PASS
9. **Dataset Dry Check (C4):** Validation sequence `seq_02_30sec_a` verified ABSENT from both configs; 20 samples per config fetched; tensor shapes valid; 100% of labels confirmed 0; 6 sample visualisations saved to `check/dataset_samples/` — PASS

---

## 5. Junction Layout

```
data/IOTD_train_real/
  seq_01_1min               -> D:/MajorProject-MeMOTR/data/IOTD/seq_01_1min (1,599 frames)
  seq_03_30sec_b            -> D:/MajorProject-MeMOTR/data/IOTD/seq_03_30sec_b (931 frames)

data/IOTD_train_aug/
  seq_01_1min_aug_01        -> D:/MajorProject-MeMOTR/data/IOTD/augmented_train_v2/seq_01_1min_aug_01 (1,599 frames)
  seq_01_1min_aug_02        -> D:/MajorProject-MeMOTR/data/IOTD/augmented_train_v2/seq_01_1min_aug_02 (1,599 frames)
  seq_01_1min_aug_03        -> D:/MajorProject-MeMOTR/data/IOTD/augmented_train_v2/seq_01_1min_aug_03 (1,599 frames)
  seq_03_30sec_b_aug_01     -> D:/MajorProject-MeMOTR/data/IOTD/augmented_train_v2/seq_03_30sec_b_aug_01 (931 frames)
  seq_03_30sec_b_aug_02     -> D:/MajorProject-MeMOTR/data/IOTD/augmented_train_v2/seq_03_30sec_b_aug_02 (931 frames)
  seq_03_30sec_b_aug_03     -> D:/MajorProject-MeMOTR/data/IOTD/augmented_train_v2/seq_03_30sec_b_aug_03 (931 frames)

data/IOTD_val/
  seq_02_30sec_a            -> D:/MajorProject-MeMOTR/data/IOTD/seq_02_30sec_a (744 frames)
```

---

## 6. Config Files

- `configs/train_iotd_real.yaml`: Sets `DATASET_DIRS: ["IOTD_train_real"]`, output `results/train_iotd_real`, label `IOTD_REAL_2026-10-07`.
- `configs/train_iotd_real_aug.yaml`: Sets `DATASET_DIRS: ["IOTD_train_real", "IOTD_train_aug"]`, output `results/train_iotd_real_aug`, label `IOTD_REAL_AUG_2026-10-07`.
- Both configs differ from each other only in `DATASET_DIRS` and the output/label names.

---

## 7. Part D Records

### D1: Visibility Information
- `seq_01_1min`: 6,448 rows total, 0 rows with vis=0.0 (0.00%), distinct visibility values: `[1.0]`
- `seq_02_30sec_a`: 4,182 rows total, 0 rows with vis=0.0 (0.00%), distinct visibility values: `[1.0]`
- `seq_03_30sec_b`: 21,547 rows total, 1,315 rows with vis=0.0 (6.10%), distinct visibility values: `[0.0, 1.0]`

### D2: Effective Resize Table
Training scales with `max_size=1536`:

| Scale | 1920x1080 (Landscape) | Capped? | 1080x2336 (Portrait) | Capped? |
|---|---|---|---|---|
| 608 | 1080x608 | no | 608x1315 | no |
| 640 | 1137x640 | no | 640x1384 | no |
| 672 | 1194x672 | no | 672x1454 | no |
| 704 | 1251x704 | no | 704x1523 | no |
| 736 | 1308x736 | no | 710x1536 | **YES** |
| 768 | 1365x768 | no | 710x1536 | **YES** |
| 800 | 1422x800 | no | 710x1536 | **YES** |
| 832 | 1479x832 | no | 710x1536 | **YES** |
| 864 | 1536x864 | no | 710x1536 | **YES** |
| 896 | 1536x864 | **YES** | 710x1536 | **YES** |
| 928 | 1536x864 | **YES** | 710x1536 | **YES** |
| 960 | 1536x864 | **YES** | 710x1536 | **YES** |
| 992 | 1536x864 | **YES** | 710x1536 | **YES** |

- 1920x1080 long-side cap applies from scale **896** onward (ratio 16:9, threshold 864).
- 1080x2336 long-side cap applies from scale **736** onward (aspect ratio ~2.163, threshold 710).

### D3: Dataset Facts
- `seq_01_1min` (night, mean luma 17.10 +/- 0.93 of 255): 1,599 frames, 6,448 rows, 37 distinct tracks (car: 1,680, two_whl: 3,661, auto: 1,107).
- `seq_02_30sec_a` (night, mean luma 35.60 +/- 3.07 of 255): 744 frames, 4,182 rows, 50 distinct tracks (car: 2,285, two_whl: 1,038, auto: 337, truck: 522).
- `seq_03_30sec_b` (dusk, mean luma 78.93 +/- 9.02 of 255): 931 frames, 21,547 rows, 55 distinct tracks (car: 9,382, two_whl: 7,176, auto: 1,842, bus: 1,556, truck: 1,591).
- **Real train totals:** 92 distinct tracks (37 + 55), 2,530 frames.
- **Gap to spec (>=500 tracks):** 408 tracks remaining (current real train tracks = 92, target = 500).

---

## 8. Items Still Open

1. **Validation procedure (from C1):** There is no training-time validation loop in MeMOTR. Validation is performed post-epoch or post-training via `main.py --mode submit` followed by TrackEval. For IOTD validation, evaluation scripts should evaluate against `data/IOTD_val/seq_02_30sec_a`.
2. **Existing dataset entries vs IOTD-only:** The current configurations are isolated to IOTD-only (`train_iotd_real` and `train_iotd_real_aug`) to provide a clean ablation baseline before mixing with UA-DETRAC and VisDrone.
3. **Frame rates & sampling intervals:** `seq_01_1min` was recorded at 24 fps, while `seq_02_30sec_a` and `seq_03_30sec_b` were recorded at 30 fps. The dataloader uses fixed frame-step sampling intervals (`SAMPLE_INTERVALS: [10]`), meaning temporal step is ~0.42s on seq_01 vs ~0.33s on seq_02/seq_03.
4. **Lighting diversity:** Current IOTD dataset consists of 2 night sequences and 1 dusk sequence. No day-lighting IOTD sequence has been annotated yet.
5. **Track count below specification:** Current real train tracks total 92, which is 408 short of the 500-track specification target.

---

Augmented frames are training samples, not additional annotated data. Real annotated data: 3,274 frames.
