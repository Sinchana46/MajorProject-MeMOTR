# Antigravity Instructions: Evaluation Diagnostics (no training)

Written 9 Oct 2026. Context: your last report passed all four checks (no flipped copies, C2 files 3,894 bytes, paint test 3,425/3,425, provenance clean) and gave this validation result on `seq_02_30sec_a` (det 0.5, track 0.5, miss tol 30, `use_c1=False`, `use_c2=False`):

| Run | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall | Precision |
|---|---|---|---|---|---|---|---|---|---|---|
| phase5 | 50.51 | 43.88 | 58.49 | 29.10 | 68.84 | 1 | 2,117 | 847 | 79.75 | 61.17 |
| arm1 | 49.58 | 45.70 | 54.04 | 24.94 | 69.66 | 7 | 2,855 | 277 | 93.38 | 57.77 |
| arm2 | 42.84 | 38.32 | 48.01 | -6.58 | 63.72 | 6 | 4,266 | 185 | 95.58 | 48.37 |

Fine-tuning raised recall but also false positives, and the augmented arm is clearly worst. We need to know **why** before anything else is built on these checkpoints. Two hypotheses: (H1) many false positives are real vehicles outside the annotated region (the labelling policy leaves opposite-carriageway and out-of-region vehicles unannotated), so the metric is dominated by a labelling convention; (H2) the fine-tuned models genuinely over-detect after longer training (Arm 2 had 4x the optimizer steps).

---

## 0. Rules

1. **Do not train.** Do not delete, move, rename or overwrite anything. Do not modify existing files: MeMOTR code, existing configs, originals under `data/IOTD/`, augmentation folders, existing `results/*` folders, existing `tools/*.py`, `check/logs/part3_final_report.txt`.
2. Allowed new outputs only: `tools/diag_*.py`, `tools/eval_diag_*.py`, `check/eval_diag/` (including images), `check/logs/`, and new subfolders under `results/eval_iotd_val/diag/`.
3. Run **each step as its own script** with `python -u`, output redirected to `check/logs/<step>.txt`, a hard timeout (300 s; 1800 s per inference run), no polling loops, no echo or latency tests, no reading of `transcript.jsonl`. Retry a failed command at most once. If a step fails, report the exact error and continue with the next independent step.
4. Paste **raw output**. Label every claim **verified** (you ran it or read the code, with file:line) or **assumed**.
5. Use the same evaluation setup as before (same tools, same evaluation-only GT with all classes set to 1, same seed and thresholds) unless a step says otherwise. Re-hash the real IOTD files at the start and end (thread pool); `changed`, `missing`, `added` must be 0.

---

## Part A. Equal-step control from existing checkpoints (no training needed)

Arm 1 ran 2,528 steps per epoch and Arm 2 ran 10,112 (batch 1, accumulation 1). So **Arm 1 `checkpoint_3` (end of epoch 3, 4 x 2,528 = 10,112 steps)** and **Arm 2 `checkpoint_0` (end of epoch 0, 10,112 steps)** have seen the same number of optimizer steps, both still at LR 2e-5 before the drop.

1. Verify the checkpoint files exist (`results/train_iotd_real/checkpoint_0..4.pth`, `results/train_iotd_real_aug/checkpoint_0..4.pth`) and print each file's `epoch` metadata. Confirm that `checkpoint_N` was saved after epoch N (0-indexed), so the step counts above hold. If not, recompute which checkpoints are equal-step and say so.
2. **Priority 1:** evaluate Arm 1 `checkpoint_3` and Arm 2 `checkpoint_0`.
3. **Priority 2:** evaluate Arm 1 `checkpoint_0, 1, 2` and Arm 2 `checkpoint_1, 2, 3`.
4. Print one table, ordered by cumulative optimizer steps, with the columns HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, Recall, Precision, for: phase5 (0 steps), both arms at every evaluated checkpoint. Add the step count per row.
5. State which arm is better at equal steps, by how much, and whether either arm's FP rises monotonically with steps.

---

## Part B. Where are the false positives? (tests H1)

For each of phase5, arm1 and arm2 (final `checkpoint_4` outputs already in `results/eval_iotd_val/`):

1. **Region of interest from the annotations.** From the original `data/IOTD_val/seq_02_30sec_a/gt/gt.txt` (read only), take the bottom-centre point `(x + w/2, y + h)` of every GT box over all frames and build its convex hull; dilate it by 5 percent of the frame width. Print the hull as a polygon and its share of the frame area. This is a diagnostic region only; do not call it ground truth.
2. **Classify predictions.** Per frame, match tracker boxes to GT boxes by IoU >= 0.5 (Hungarian). For every unmatched prediction (false positive) record whether its bottom-centre point is inside or outside the hull. Print FP inside / FP outside / TP for each run.
3. **ROI-filtered evaluation.** Make a copy of each tracker output with all rows whose bottom-centre lies outside the hull removed (new files under `results/eval_iotd_val/diag/roi/`), and run TrackEval on those with the same evaluation-only GT. Print the same metrics table for the three runs. State clearly that this is a diagnostic and apply the **same** filter to all three.
4. **Render.** For each run render 6 evenly spaced frames with GT boxes (green), tracker boxes (red) and the hull (yellow) to `check/eval_diag/render_<run>_fXXXXXX.jpg`. This render step was required last time and was skipped; do it now.

State whether H1 is supported: what fraction of each run's false positives lies outside the hull, and whether the ranking phase5 > arm1 > arm2 changes after ROI filtering.

---

## Part C. Threshold sweep (tests H2)

1. Inspect the tracker output files: print the distribution of the confidence column (column 7) for each run. If it holds real scores (not a constant), filter the existing outputs post hoc at 0.5, 0.6, 0.7, 0.8 and 0.9 and run TrackEval for each (label this **approximate**, since the tracker's internal logic used 0.5). If the column is constant, instead rerun inference with `--det-thresh` 0.7 and 0.8 and `--track-thresh` 0.7 and 0.8 for the three final checkpoints (new folders, same seed).
2. Print a table of HOTA, DetA, AssA, MOTA, FP, FN, Precision against threshold for each run, both on the full GT and on the ROI-filtered outputs from Part B.
3. State whether a higher threshold closes the gap between the arms and phase5.

---

## Part D. Checks on your own report (read-only)

1. **C2 adapter files.** You reported `state_dict` has 0 tensor parameters and yet answered "plausible: YES". Print the exact keys and value types of `state_dict` in `results/phase6_c2_interim/c2_adapter_epoch_4.pth`. Find where the C2 adapter weights are actually stored and how the interim C2 evaluation loaded them: search the training and inference code for `c2`, `adapter`, `tau`, and the full Phase 6 checkpoints for matching keys. Report, with file:line, whether the Phase 6 interim C2 result (HOTA 60.42) used **trained** adapter weights, random-initialised weights, or none.
2. **Parameter differences.** Your numbers cannot all be right: for the same set of parameters, mean absolute difference is a metric, so `d(arm1, arm2)` must be at least `|d(phase5, arm2) - d(phase5, arm1)|`, about 0.0009 here, but you reported 0.0000271. Recompute all three pairwise mean absolute differences over the **identical** set of non-backbone parameters, give them per top-level module as well, and print SHA256 of `results/train_iotd_real/checkpoint_4.pth` and `results/train_iotd_real_aug/checkpoint_4.pth` to confirm the files differ.
3. **Corrections.** Search `check/logs/part3_final_report.txt` and your last message for these errors and write a corrected copy as a **new** file `check/logs/part3_final_report_corrected.txt`:
   - IOTD class names are 1 car, 2 two_wheeler, 3 auto, 4 bus, 5 truck. "3 = car, 5 = motorbike" are TrackEval's own names, not ours; `seq_02` has classes 1, 2, 3 and 5 (car, two_wheeler, auto, truck).
   - `seq_02_30sec_a` is **night** (mean luma 35.6), not daytime. The real gap is that IOTD has no day or dusk validation sequence.
   - Remove "Render paths" from the results table unless renders exist.
   - Mark the explanation of the earlier misaligned preview as supported only by the paint test (the training path is correct); the cause of the preview error itself is not proven.

---

## Part E. Final report (compact, then stop)

```
1. EQUAL-STEP CONTROL: arm1 ckpt_3 vs arm2 ckpt_0 (10,112 steps) table.
   Better arm at equal steps, margin.  FP vs steps for both arms.
2. FALSE-POSITIVE LOCATION: per run, FP inside / outside the ROI hull, TP.
   ROI-filtered metrics table. Is H1 supported: YES/NO/PARTLY.
3. THRESHOLD SWEEP: table, and whether gaps close. Is H2 supported.
4. C2 FILES: where the adapter weights are; trained / random / none used in Phase 6.
5. PARAMETER DIFFERENCES: corrected values; files differ: YES/NO.
6. CORRECTIONS written to: <path>.
7. Real IOTD file re-hash: changed / missing / added.
8. Unverified or assumed:
```

Required caveats under the tables, in your own words: validation is one night sequence of 744 frames and 50 tracks; every number is from a single run; the ROI is derived from the annotations and is a diagnostic only; IOTD has no day or dusk validation data.

Then stop. Do not start training, do not delete anything, do not modify originals, configs, augmentation folders or existing results.
