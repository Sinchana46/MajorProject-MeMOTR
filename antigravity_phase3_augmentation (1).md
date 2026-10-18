# Antigravity Instructions: IOTD Augmentation and Training-Readiness

Written 6 Oct 2026. Read together with `context__1_.md` (project context) and `context_frame_extraction_update.md` (extraction results). This file covers the augmentation phase and the readiness report that follows it.

---

## 0. How you must work

1. Do the steps **in order**. Stop at every line marked **GATE** and wait for the user to reply. Do not continue on your own.
2. **Paste raw printed output** (tables, counts, hashes, command output). Do not summarise numbers; the user checks them. An earlier class check was reported as passing when it was wrong.
3. Never say a check passed unless you ran it in this session and show the output.
4. If a path, column, file or number differs from this document, **stop and report the difference**. Do not guess or silently adapt.
5. Never delete anything. Never overwrite anything that already exists. If a target path exists, stop and ask.

---

## 1. Protected files (read-only, never modify)

- All `gt.txt`, `labels.txt`, `seqinfo.ini` in the real sequence folders
- All original videos (`raw_videos/`) and all extracted JPEGs in the real sequence folders
- All MeMOTR code, configs and checkpoints, and everything in `modules/`, `tools/` that already exists

You may only **create new files** in: `augmented_train/`, `check/`, `split_manifest.json`, and new scripts under `tools/`.

---

## 2. Decisions already made (do not re-open)

| Topic | Decision |
|---|---|
| Provisional split | Train = `seq_01_1min` + `seq_03_30sec_b`. Validation = `seq_02_30sec_a`. Label it provisional; the final split is 12/4/4 by sequence once more data exists |
| What gets augmented | **Train sequences only**, 3 copies each. Never augment the validation sequence |
| Target size | 2,530 real train frames x (1 original + 3 copies) = 10,120 training frames. These are training samples, not new annotations. Real annotated total stays 3,274 |
| Augmentation type | **Photometric only.** No flip, crop, scale, rotate, shift, perspective |
| Labels | `gt.txt`, `labels.txt`, `seqinfo.ini` copied **unchanged** to each copy, SHA256 verified |
| Class column | In `gt.txt`, column 8 is class (1 car, 2 two_wheeler, 3 auto, 4 bus, 5 truck). Column 7 is the "not ignored" flag. Column 9 is visibility |
| Frame naming | `000001.jpg` onward, 1-based, JPEG quality 98, same as the extraction |
| C2 training | Augmented copies are **never** used to train the C2 ThresholdAdapter. Mark them `"use_for_c2": false` in the manifest |
| C1 calibration | Uses original frames only (the one-time regression of box height against vertical position; photometric copies repeat identical boxes, so they add no information and would over-weight the same tracks) |
| Augmented copies ARE for | Baseline B fine-tuning of the full model, and C1 PositionMLP/TIM training (box positions and heights are unchanged by photometric edits) |
| Augmented copies are NOT for | C2 adapter training, C1 calibration, validation, test, or any reported dataset size |

---

## 3. Tasks the user does, not you

You cannot do these. Do not claim them done:

- Compare frames read against the frame count on each CVAT task page (1599 / 744 / 931)
- Inspect the last-frame overlays, especially seq_03 (portrait)
- Decide the labelling policy for vehicles with no boxes (opposite carriageway, frame edges)
- Back up `raw_videos/`, CVAT exports and the CVAT project

---

## 4. Steps

### Step 1. Discover paths (read-only)

Print the real location of every sequence folder (`img1/`, `gt/gt.txt`, `gt/labels.txt`, `seqinfo.ini`), `raw_videos/`, `check/`. The update file says `annotated_pending_split/`, but an earlier summary said `data/IOTD/seq_*/img1/`. Report which is true. Also print free disk space on the drive that will hold `augmented_train/`.

Estimate the disk needed: (average JPEG size per sequence) x (frames) x 3 copies, summed over the two train sequences. If free space is under 1.5x the estimate, stop.

### Step 2. Profile the sequences (read-only)

The user will place `seq_profile.py` in `tools/`. If it is missing, ask. Run it on the folder holding the sequences and paste the full printed output and `seq_profile.csv`. Also report:

- Distinct track IDs and row counts per sequence
- Distinct values in column 9 (visibility) per sequence
- Mean brightness per sequence and whether each is day, dusk or night

**GATE A.** Wait for the user to confirm the split against the profile. If the profile contradicts the provisional split, say so plainly and stop.

### Precondition for Step 3 (user confirmation required)

Steps 1 and 2 are read-only and may run while the user works on the tasks in section 3. **Do not start Step 3 until the user has said, in their own words, that all four tasks in section 3 are done.** If the user changes the labelling policy and re-exports from CVAT, `gt.txt` will change, the hashes and the profile must be redone, and no augmented copy may exist yet. Ask the user to confirm explicitly; do not assume.

### Step 3. Integrity manifest before any change

Compute SHA256 of every protected file in the three real sequences (all `gt.txt`, `labels.txt`, `seqinfo.ini`, and every JPEG in `img1/`). Save as `check/hashes_before.json` and print a per-sequence file count.

### Step 4. Write `split_manifest.json`

Create it (do not overwrite if present). Schema:

```json
{
  "status": "provisional",
  "note": "3 sequences only. Final split will be 12/4/4 by sequence.",
  "train": ["seq_01_1min", "seq_03_30sec_b"],
  "val": ["seq_02_30sec_a"],
  "test": [],
  "augmented": {
    "seq_01_1min_aug_01": {"source": "seq_01_1min", "seed": 1001, "use_for_c2": false}
  }
}
```

List every augmented copy that will exist (six in total) with its source, seed and `"use_for_c2": false`.

### Step 5. Write the augmentation script, then STOP

Create `tools/augment_photometric.py`. Do not run it.

**Library check first.** Print `albumentations.__version__` and the signatures of `RandomBrightnessContrast`, `RandomGamma`, `HueSaturationValue`, `ImageCompression`. Argument names changed between versions (for example `GaussNoise` uses `var_limit` in older releases and `std_range` in newer ones). Use the names that exist in the installed version.

**Per-copy parameters.** Sample each value **once per copy** from the range below using that copy's seed, then apply the same values to every frame of the copy (`A.ReplayCompose`: build the replay data on the first frame, replay on the rest). Write the sampled values to `augmented_train/<copy>/aug_params.json`.

| Copy | Seed | Brightness limit | Contrast limit | Gamma (%) | Hue / Sat shift | JPEG quality | Noise sigma (0-255) |
|---|---|---|---|---|---|---|---|
| aug_01 | 1001 | -0.15 to -0.05 | -0.10 to 0.10 | 110 to 130 | +/-3 / +/-10 | 75 to 90 | 2 to 4 |
| aug_02 | 1002 | +0.05 to +0.15 | -0.10 to 0.10 | 80 to 95 | +/-5 / +/-15 | 65 to 85 | 3 to 6 |
| aug_03 | 1003 | -0.08 to +0.08 | -0.15 to 0.15 | 90 to 110 | +/-5 / +/-20 | 60 to 80 | 4 to 8 |

Value (brightness) shift in `HueSaturationValue` is 0, because brightness is handled separately. Motion blur is **off** by default (config flag, `p=0`): a fixed blur kernel applied to a fixed camera blurs the static background uniformly, which is not realistic.

**Noise must differ on every frame.** Sensor noise is independent from frame to frame in real video. If one noise pattern is replayed on every frame it becomes a fixed texture on the image that a tracker with memory could latch onto. So:

- Apply the replayed deterministic transforms (brightness, contrast, gamma, hue/sat, JPEG) first.
- Then add Gaussian noise **outside** `ReplayCompose`, using `np.random.default_rng(copy_seed * 1_000_003 + frame_index)` so it is different per frame yet fully reproducible.
- Clip to 0-255 and cast to uint8 after adding noise.

**Safety in the script:**

- Refuse to run if the source sequence is listed under `val` or `test` in `split_manifest.json`
- Refuse to write into any path that already exists
- Write only under `augmented_train/<seq>_aug_NN/` with the structure `img1/`, `gt/gt.txt`, `gt/labels.txt`, `seqinfo.ini`
- Handle each sequence's own width and height (seq_03 is 1080x2336 portrait). No resizing
- Copy the three label files byte for byte (`shutil.copyfile`), then compare SHA256 with the source
- Support `--dry-run N` (process only the first N frames of one sequence into a temporary folder under `check/`)

**GATE B.** Show the full script and the library signatures. Wait for approval.

### Step 6. Dry run, then STOP

Run `--dry-run 20` on `seq_03_30sec_b` for all three copies (output under `check/dryrun/`). Print:

- Output file count and image dimensions versus the source
- Mean absolute pixel difference versus the source frame, per copy
- Noise independence test: generate the noise array for frame index 1 and 2 of one copy on a flat grey image, and print their correlation (should be close to 0, under 0.05)
- Three overlay images per copy (first, middle, last of the 20 frames) using `gt.txt` **column 8** for the class label

**GATE C.** Wait for approval.

### Step 7. Full run

Run the real generation for the two train sequences, 3 copies each, into `data/IOTD/augmented_train/`. Print progress per sequence and the final file counts. Report the number of `imwrite` failures (expected 0).

### Step 8. Verification, then STOP

Print raw output for each item:

1. File count per augmented folder equals the source (1599, 931)
2. Image dimensions match the source for every file (sample every 50th frame is acceptable; say which)
3. SHA256 of `gt.txt`, `labels.txt`, `seqinfo.ini` in each copy equals the source
4. Mean absolute pixel difference per copy versus source. Flag any copy where it is 0 or above 40
5. `check/hashes_after.json` computed the same way as before, and a line-by-line comparison with `hashes_before.json`: the number of changed, missing or new files in the real sequences must be 0
6. Confirm nothing was written outside `augmented_train/`, `check/`, `split_manifest.json`, `tools/`
7. Overlays: first, middle and last frame of every copy (18 images) in `check/aug_overlays/`, using column 8 for class
8. Total frame count: real train frames, augmented frames, and the sum (expected 2,530 / 7,590 / 10,120)

**GATE D.** Wait for the user to inspect the overlays.

### Step 9. Training-readiness report (read-only on MeMOTR)

Do **not** modify any MeMOTR file. Read the dataset and dataloader code and report, with file names and line numbers:

1. Which folder layout the dataloader expects, and whether our layout matches. If not, list the exact differences
2. How `gt.txt` is parsed: which column it takes as class, as visibility, as the "not ignored" flag
3. Whether it uses the class column at all, or treats every box as one class (do not assume; quote the code)
4. How it handles frames of different sizes in one run (landscape 1920x1080 and portrait 1080x2336): resize, pad, or error
5. How it samples clips (clip length, sampling interval) and whether sequences shorter than that would fail
6. How train and val sequence lists are specified, and where our `split_manifest.json` would plug in
7. Any frame-rate assumptions (sequences run at 24, 24 and 30 fps)

Then a one-page summary table:

| Item | seq_01 | seq_03 | seq_02 (val) |
|---|---|---|---|
| Real frames | | | |
| Augmented copies | | | none |
| Distinct tracks | | | |
| Boxes per class | | | |
| Lighting | | | |

and a list of risks you found, each labelled **verified** (you ran it or read the code) or **assumed**.

End the report with the statement: "Augmented frames are training samples, not additional annotated data. Real annotated data: 3,274 frames."

---

## 5. Standing rules (unchanged)

Split by sequence, never by frame. Never touch test data before the final evaluation. Do not modify the MeMOTR backbone or encoder. Changes that alter default fixed-lambda or fixed-tau behaviour need a feature flag. Report more than MOTA. Keep only code the user can explain line by line.
