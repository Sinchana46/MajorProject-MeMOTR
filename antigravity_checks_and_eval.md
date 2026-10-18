# Antigravity Instructions: Pre-Evaluation Checks, then Evaluation of Three Checkpoints

Written 9 Oct 2026. Context: Phase 7 trained two arms (real only; real + augmented v2), 5 epochs each. The project report contains claims that conflict with verified facts, and some are not yet checked. This file verifies them, then evaluates on the validation sequence. **No training in this task.**

---

## 0. Rules

1. **Do not train. Do not delete, move, rename or overwrite anything.** You previously deleted `check/dataset_samples` and `check/box_alignment_diagnostic`; do not delete diagnostics again. If disk space is a concern, ask.
2. **Do not modify** any existing file: MeMOTR code, existing configs, originals under `data/IOTD/`, v1 or v2 augmentation folders, existing `results/*` folders, existing `tools/*.py`. Allowed new files only: `tools/diag_*.py`, `tools/eval_*.py`, `check/logs/`, `check/box_alignment_diagnostic_v2/` (at most 10 images), and new folders under `results/eval_iotd_val/`.
3. **Run each step as its own script** with `python -u`, output redirected to `check/logs/<step>.txt`, hard timeout (300 s; 1800 s for inference), no polling loops, no echo or latency tests, do not read `transcript.jsonl`. Retry a failed command at most once.
4. Paste **raw output**. Label every claim **verified** (you ran it or read the code, with file:line) or **assumed**.
5. If a step fails, report the exact error and line, then continue with the next independent step. **STOP conditions** are marked and override everything.
6. Before the first step and after the last, re-hash all real IOTD files against `check/hashes_before.json` (use a thread pool). `changed`, `missing` and `added` must all be 0.

Verified facts you can rely on: `seq_02_30sec_a` has 744 frames, 4,182 rows, 50 tracks, and **is already annotated** (no annotation task exists). The original `gt.txt` files have classes 1 to 5 in column 8. v2 augmentation was specified as: no flip, gains sampled once per copy (0.85-0.95, 1.10-1.30, 0.95-1.10), gamma, hue/sat, JPEG, per-frame noise, `gt.txt` copied byte for byte.

---

## Part 1. Checks (all read-only)

### Check 1. Was any augmented copy horizontally flipped? (decisive for Arm 2)

The project report claims v2 used "H-flip on 3 of 6 variants" with gains 0.6/0.8/1.2. That contradicts the spec and the verified luma ratios. For each of the six copies in `data/IOTD/augmented_train_v2/` and its source, on frames 000100, 000400 and 000800 (000100, 000400 and 000900 for the 931-frame sequence), load grayscale and print:

- Pearson correlation of source vs augmented
- Pearson correlation of **horizontally flipped source** vs augmented

Flag a copy as FLIPPED if the flipped correlation exceeds the plain one on any of the frames. Also print each copy's `aug_params.json`. Read `tools/augment_photometric_v2.py` and quote any line containing `flip`, `fliplr`, `[:, ::-1]` or `cv2.flip` (or state there are none, with the search you ran).

**STOP if any copy is FLIPPED.** Report the copies and frames; do not continue.

### Check 2. C2 adapter checkpoint files

The report says `results/phase6_c2_interim/c2_adapter_epoch_*.pth` are "0 bytes". Print each file's exact size in **bytes** (`Length`). Then `torch.load` `c2_adapter_epoch_4.pth` (CPU) and print its keys, tensor shapes and total parameter count (expected 209 for the 3-16-8-1 MLP). Say whether a file of that size is plausible for 209 parameters. Do not regenerate or modify anything.

### Check 3. Box alignment in the real training path (numeric paint test)

The report calls the earlier misaligned preview a "false alarm" caused by "multi-call stochastic transforms" but shows no numeric evidence. Test the actual training path.

1. Read the code that drew `check/dataset_samples/` (in `tools/run_partC_and_partD.py`). Quote the lines. State whether it called the random transform more than once per sample (once for the image and again for the boxes), what box format it assumed, and what format the loader returns (file:line).
2. Print the exact `split="train"` transform list of `VehicleMOT` for `configs/train_iotd_real_aug.yaml`, marking which are geometric and which also update boxes.
3. **Paint test** (`tools/diag_paint_test.py`): take a raw frame, paint a solid distinctive colour (pure magenta) inside every GT box, run it through the **same** dataset transform and normalization, de-normalize the output, build a mask of the painted colour, and compare it with the loader's returned boxes after decoding them to pixels in the **transformed** image. For every returned box at least 4 px on each side compute the share of its area covered by the mask; also compute the share of mask pixels outside all boxes. Run 200 draws over both orientations, real and aug copies, and both frames of the clip. **Pass rule:** each evaluated box covers at least 0.85, and at most 10 percent of mask pixels fall outside all boxes. Report the pass rate and the worst case.
4. Save at most 10 diagnostic images (correctly drawn boxes on transformed frames) to `check/box_alignment_diagnostic_v2/`.

**STOP if the paint test fails.** Report the exact transform and the coordinates before and after.

### Check 4. Training provenance for both arms

Using the training logs and any config copy saved under `results/train_iotd_real/` and `results/train_iotd_real_aug/` (note: the configs were edited after the first attempt, so use what was effective at training time), print:

1. The line showing which pretrained or resume weights were loaded, and whether the loader restored only weights or also epoch and optimizer (Phase 5 `checkpoint_4.pth` is the intended start)
2. The line `[VehicleMOT split=train] Loaded N sequences` and the sequence names, and iterations per epoch for every epoch (expected 2,528 and 10,112; those equal `(1599-1)+(931-1)` and four times that, which also implies `seq_02` was absent), and the clip sample length used per epoch
3. LR per epoch and the epoch of the MultiStep drop
4. Checkpoint comparison with `torch.load` on CPU: for Arm 1 `checkpoint_4.pth`, Arm 2 `checkpoint_4.pth` and Phase 5 `checkpoint_4.pth`, report epoch/metadata keys; the maximum absolute difference of **backbone** parameters between Phase 5 and each arm (expected exactly 0, frozen); the mean absolute difference of decoder, encoder-adjacent and query-updater parameters (expected non-zero); and whether Arm 1 and Arm 2 differ from each other

**STOP if** the logged start weights were not the Phase 5 checkpoint, or if `seq_02_30sec_a` appears in either training list.

---

## Part 2. Evaluation on `IOTD_val` (only if Part 1 passed)

Evaluate **three** checkpoints on `data/IOTD_val/seq_02_30sec_a` with identical settings:

| Run | Checkpoint | Purpose |
|---|---|---|
| `phase5` | `results/phase5_interim_baseline/checkpoint_4.pth` | Baseline without IOTD fine-tuning |
| `arm1` | `results/train_iotd_real/checkpoint_4.pth` | Fine-tuned, real only |
| `arm2` | `results/train_iotd_real_aug/checkpoint_4.pth` | Fine-tuned, real + augmented |

### E1. Find the evaluation path (read-only)

Prefer the existing tools used for Phase 5 and Phase 6 (`tools/run_inference_mot.py`, `tools/run_trackeval.py`) so the numbers are comparable (det threshold 0.5, track threshold 0.5, miss tolerance 30 frames). Print the exact commands, config keys, and how they handle image naming (IOTD frames are `000001.jpg` onward), `seqinfo.ini`, the sequence map, and the inference resize for a 1920x1080 frame (the portrait case is not needed here). Use `use_c1=False` and `use_c2=False`. Confirm the junction `data/IOTD_val/` works as the data source. If anything must be created, write only under `results/eval_iotd_val/`. Check free GPU memory (`nvidia-smi`) and that no other training process is running.

### E2. Ground truth for evaluation (important)

The original `gt.txt` has classes 1 to 5 in column 8. I believe TrackEval's MOTChallenge loader evaluates class 1 as the target, treats some other class IDs as distractors (so predictions matched to them are ignored), and ignores the rest (so predictions on them count as false positives). The earlier UA-DETRAC and VisDrone sets were remapped to class 1 for this reason. Read the TrackEval code and the wrapper's GT builder and state, with file:line, how classes 2 to 5 would be treated. Then:

- Build an **evaluation-only copy** of the GT at `results/eval_iotd_val/gt_eval/seq_02_30sec_a/gt/gt.txt` with column 8 set to 1 for every row, copying `seqinfo.ini` unchanged. Never modify the original `gt.txt`.
- Print the row count (expected 4,182), distinct track IDs (50), and the SHA256 of the original (unchanged).
- State whether `run_trackeval.py` rewrites `seqinfo.ini` (its auto-fix of `seqLength`) and confirm it only touches copies under `results/eval_iotd_val/`.

### E3. Smoke test, then full runs

Run `phase5` on the first 50 frames only and print the tracker output file (row count, columns, frame range). If the output is empty or malformed, **STOP** and report. Then run all three checkpoints on the full 744 frames with the same seed and settings, outputs to `results/eval_iotd_val/phase5/`, `arm1/`, `arm2/` (refuse to overwrite). Render the tracker output for the three runs on 3 evenly spaced frames each as `check/eval_samples/` images (boxes and IDs drawn in pixel coordinates), so the user can look.

### E4. TrackEval

Run TrackEval on all three against the evaluation-only GT. Print one table:

| Run | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall | Precision | Tracker rows | GT rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Also print the raw summary file paths.

---

## Part 3. Final report (compact, then stop)

```
1. FLIP CHECK: PASS / FAIL   (correlations per copy)
2. C2 CHECKPOINTS: sizes in bytes, parameter count, plausible: YES/NO
3. BOX ALIGNMENT (paint test): PASS / FAIL, pass rate, worst case,
   root cause of the earlier misaligned preview (one sentence, with evidence)
4. TRAINING PROVENANCE: start weights, resume or weights-only,
   sequences loaded per arm, iterations per epoch, backbone diff = 0: YES/NO
5. EVAL GT: how classes 2-5 would be treated, evaluation GT built: YES/NO
6. RESULTS TABLE (3 runs) and render paths
7. Original IOTD files re-hash: changed / missing / added
8. Unverified or assumed:
```

Required caveats to print under the results table, in your own words: (a) Arm 2 had four times the optimizer steps of Arm 1 (50,560 vs 12,640), so Arm 1 vs Arm 2 is **not** a controlled test of augmentation; (b) validation is one night sequence of 744 frames and 50 tracks, each run done once, so small differences are not meaningful; (c) no day sequence exists in IOTD.

Then stop. Do not start training, do not delete anything, do not modify any original, augmentation, config or existing result.
