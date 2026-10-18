# Antigravity Instructions: Gate 1 Resolution (validation annotations only, no retraining)

Written 11 Oct 2026. Read together with `data/IOTD/check/hyperparameter_tuning_readiness.md` (your previous report) and `check/annotation_audit/summary.md`.

**Decision already taken by the user (branch "validation only"):**
- Correct only the **validation** sequence `seq_02_30sec_a`, and only for vehicles that are genuinely missing under the existing policy.
- Do **not** touch the training annotations (`seq_01_1min`, `seq_03_30sec_b`). Your audit found no policy violation there: `seq_01_1min` has 5 candidates, all `LIKELY_OPPOSITE` (consistent with the exclusion policy), and `seq_03_30sec_b` has none. So **no retraining** and Arm 1, Arm 2 and Phase 5 stay valid.
- Do not change the split. Do not launch seed repeats or any training. Do not invent a distance or image-position cutoff.

Policy (unchanged): only vehicles belonging to the monitored traffic carriageway are annotated; vehicles on the opposite carriageway and vehicles outside the intended surveillance region are not.

---

## 0. Rules

1. **Do not delete, move, rename or overwrite anything.** The original `data/IOTD/seq_02_30sec_a/gt/gt.txt` must stay byte-for-byte unchanged and in place; the corrected annotation is written as a **separate file** (Part C). Do not modify: any `labels.txt` or `seqinfo.ini`, images, training annotations, augmented folders, existing configs and code, checkpoints, existing predictions, or any existing result or report.
2. Allowed new outputs only: `tools/gate1_*.py`, `check/annotation_audit/gate1/`, `check/annotation_audit/backup/`, `check/logs/`, `results/eval_iotd_val/rescored_v2/`, and the report `check/annotation_audit/gate1_resolution_report.md`.
3. Run **each step as its own script** with `python -u`, output redirected to `check/logs/gate1_<step>.txt`, hard timeout (300 s), no polling loops, no echo or latency tests, no reading `transcript.jsonl`. Retry a failed command at most once; if a step fails, report the exact error and continue with the next independent step.
4. Paste **raw output**. Label every claim **verified** (you ran it or looked at the pixels or code, with the path) or **assumed**. Never write "PASS" without showing the evidence.
5. Re-hash all real IOTD files against `check/hashes_before.json` at the start and at the end (thread pool). `changed`, `missing`, `added` must all be 0, which also proves the original `seq_02` GT was not touched.
6. No training, no inference with new checkpoints. Everything below uses the **existing** predictions.

---

## Part A. Inputs

1. Read `check/annotation_audit/summary.md` and `proposals.csv`. Verify the ten `seq_02_30sec_a` candidate IDs against the file (four `REVIEW_ADD`: 007, 005, 006, 009; six `REVIEW_UNCERTAIN`: 004, 002, 008, 000, 001, 003) and print the CSV rows. If the IDs differ, use the file and say so.
2. List the evidence images in `check/annotation_audit/seq_02_30sec_a/` and the CVAT import file. State which tracker output each candidate came from.
3. Print the structure of the existing `seq_02_30sec_a/gt/gt.txt`: line endings (LF or CRLF), trailing newline, number format (decimal places), sort order, ID range (minimum and maximum), maximum frame, distinct visibility values, and the class IDs present. The new rows must follow exactly these conventions.

---

## Part B. Review the ten candidates

Work from the **original frames**, not from the model renders alone. For each candidate:

1. View the track at its first, last and at least three evenly spaced middle frames, as crops enlarged about 3x, plus the full frame. Save these as `check/annotation_audit/gate1/cand_<id>_review.jpg`.
2. Decide from the pixels whether it is a real vehicle, and where it is: on the **monitored carriageway** (the road the camera is aimed at, with the annotated flow), or on a different road, beyond a wall, barrier or median, or travelling the opposite way. Use the physical scene (barrier, wall, road edge), the direction of travel (most GT tracks move down the image, toward the camera) and the position relative to the GT lanes. `UP` direction in a `DOWN` sequence suggests the opposite carriageway or another road, but check it, do not assume it.
3. Also check **gap-fill and duplicates**: is the candidate the continuation of an existing GT track whose boxes are missing for some frames (same position, same size, same direction, an existing ID that vanishes where the candidate begins)? Is it a duplicate of an annotated vehicle? List any GT track that has frame gaps in the neighbourhood.
4. Assign exactly one disposition:
   - `ADD`: a real vehicle on the monitored carriageway missing from the GT
   - `GAP_FILL`: a missing stretch of an existing GT track (extend that track's ID; do not create a new one)
   - `EXCLUDE_POLICY`: a real vehicle that the policy leaves unannotated (opposite carriageway, other road, outside the region)
   - `FALSE_POSITIVE`: not a vehicle, or not a stable object
   - `DUPLICATE`
   - `UNCERTAIN`: cannot be settled from the images
5. Give a confidence (high, medium, low) and a one-sentence reason that refers to what is visible. **Model agreement is not evidence that a vehicle is real or should be annotated.**

Write `check/annotation_audit/gate1/disposition_table.csv` (candidate, frame range, disposition, confidence, reason, evidence image) and print it.

**Only `ADD` and `GAP_FILL` with high confidence may be applied.** Medium confidence goes in the review pack as "proposed, not applied". Anything about whether a vehicle falls inside the monitored region that is truly ambiguous is `UNCERTAIN` and is left unchanged.

For `seq_01_1min`, do not change anything. Just confirm in the report, with two evidence images, that the five `LIKELY_OPPOSITE` candidates look like opposite-carriageway vehicles.

---

## Part C. Build the corrected validation GT (separate file)

For every candidate with disposition `ADD` or `GAP_FILL` and high confidence:

1. **Backups first.** Copy the original `seq_02_30sec_a/gt/gt.txt` byte for byte to `check/annotation_audit/backup/seq_02_30sec_a_gt_ORIGINAL.txt`, and record both SHA256 values (they must be equal).
2. **Boxes.** The only box source you have is the model predictions, so they cannot be copied in as truth. Use them as **starting points** only: choose the most complete single-run track, then for every 10th frame of the vehicle's span (and its first and last frames) look at the crop and adjust the box so it tightly encloses the visible vehicle, matching how existing GT boxes are drawn (look at 5 neighbouring annotated vehicles for the convention). Interpolate linearly between those keyframes, then re-check the overlays and fix any frame where the box has drifted. Print the keyframe table for each added track.
3. **Rows.** Follow the conventions found in Part A exactly: `frame,id,x,y,w,h,1,class,1.0`. Class is your best visual estimate from {1 car, 2 two_wheeler, 3 auto, 4 bus, 5 truck}; if the class is uncertain but the vehicle is certain, use your best estimate and mark `class_uncertain` in the change log (the evaluation is single-class, so this does not affect scores). New track IDs start above the existing maximum and are unique. A `GAP_FILL` reuses the existing ID.
4. **Write** the corrected file to `check/annotation_audit/gate1/seq_02_30sec_a_gt_CORRECTED.txt` (never in place). Also write `change_log.json` listing every added row with the reason and the candidate; no row may be modified or removed unless a `GAP_FILL` requires it, and any such row is listed.
5. If no candidate qualifies, say so, write no corrected file, and skip to Part E with the original GT only.

### Verify the corrected GT (print the evidence)

- File parses with the same loader the evaluation tools use; row count equals the original plus the added rows; all IDs unique per frame (no duplicate (frame, id)); frame numbers within 1 to 744; all `w, h > 0`; boxes inside the frame; no new box overlaps an existing annotated vehicle with IoU above 0.3 in the same frame
- Overlays: 3 frames per added track (first, middle, last) with new boxes in cyan and existing GT in green, saved under `check/annotation_audit/gate1/overlays/`
- SHA256 of the original (unchanged), the backup, and the corrected file
- Confirm training annotations and every other protected file are unchanged (the Part 0 re-hash)

---

## Part D. Completeness check and review pack

The candidates came from model predictions, so any vehicle that **no model detected** is still missing and cannot be found this way. Estimate how much remains:

1. Take 20 evenly spaced frames of `seq_02_30sec_a` and the corrected GT (or the original if none was corrected). For each, list by looking at the image every vehicle you can see on the monitored carriageway that is not covered by a GT box. Save one contact sheet per frame with GT in green and any missed vehicles marked in magenta.
2. Report the count of visible-but-unannotated vehicles per frame and in total, and state it as an **estimate by visual inspection, not ground truth**. State plainly whether the validation GT should now be considered complete enough for model comparison, or whether a full pass in CVAT by the user is still needed.
3. Assemble the review pack in `check/annotation_audit/gate1/review_pack/`: the disposition table, all `ADD` and `GAP_FILL` overlays, the `UNCERTAIN` and medium-confidence cases, and the 20 completeness sheets, so the user can review everything in one place.

---

## Part E. Re-score the existing predictions (no retraining, no new inference)

1. Locate the existing tracker outputs for: `phase5`; `arm1` ckpt_0, 1, 2, 3, 4; `arm2` ckpt_0, 1, 2, 3, 4 (all at det/track 0.5 and miss tolerance 30), and the existing 0.7 and 0.8 results for the three ckpt_4 runs. Print the paths and row counts. State which are missing (none are expected).
2. Use the same TrackEval settings and the same single-class conversion as before (column 8 set to 1 in an **evaluation-only copy**, never in the original). Confirm and print how the evaluator treats detections in unannotated areas (the wrapper uses `DO_PREPROC=False`, so there are no distractor or ignore regions; every unmatched prediction is a false positive). **Do not add any region filter or change any evaluation rule.**
3. Score every run three ways, writing to `results/eval_iotd_val/rescored_v2/<scoring>/<run>/`, with all earlier results left untouched:
   - **S0, original GT** (reproduces the historical numbers; they must match the earlier table, else say why)
   - **S1, corrected GT** (the added tracks count as ground truth)
   - **S2, sensitivity: added tracks as ignored.** Using the original GT, remove from each tracker output the rows whose boxes match an added vehicle (IoU >= 0.5), so that finding or missing those vehicles neither helps nor hurts. This is a clearly labelled **sensitivity analysis**, not the evaluation protocol. Its purpose is to check that results do not depend on the added boxes having been derived from model predictions.
4. Print one table with HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, Recall, Precision for all 11 runs at 0.5 under S0, S1 and S2 (side by side or three tables), and the 0.7 and 0.8 ckpt_4 rows for the three runs under S1.
5. State whether the ranking of the runs, and the best checkpoint per arm under the written rule (highest HOTA at det/track 0.5), is the **same under S0, S1 and S2**. If it differs, say where and by how much.

Required caveat for this part: the corrected GT was built using the same models that are being scored, and the checkpoint is then selected on the same single sequence. Both make the numbers optimistic. Selecting a checkpoint or a threshold on `seq_02_30sec_a` is tuning on the validation sequence.

---

## Part F. Report and status (then stop)

Write `check/annotation_audit/gate1_resolution_report.md` with:

1. The final disposition of all ten candidates (the table from Part B)
2. Evidence for every annotation correction (overlay and keyframe paths)
3. Backup paths and before and after SHA256 values; confirmation that the original `seq_02` GT, the training data and all checkpoints were not modified (re-hash result)
4. The Part D completeness estimate and your recommendation on whether the validation GT is complete enough
5. The three-way re-scoring tables and the ranking comparison
6. The best checkpoint per arm under the written rule, per scoring
7. Unresolved cases that need the user's review, each with its evidence image
8. State plainly: real annotated data is **3,274 frames in total** (seq_01 1,599 + seq_02 744 + seq_03 931), of which **2,530 are real training frames** and 744 are validation. (The previous report's caveat 3 gives wrong numbers: it calls seq_03 1,675 frames.) Augmented frames are training samples, not additional annotated data
9. A status: **PASS**, **PARTIAL** or **BLOCKED**, with the reason, and the recommended next step. Do not start it

Stop and ask the user, instead of continuing, if: a vehicle's inclusion under the policy is genuinely ambiguous; a track cannot be annotated reliably; a correction would require changing the training data or the model setup; the evaluator would need a new exclusion rule; or the original GT hash changes at any point.

Do not launch seed repeats, hyperparameter tuning or any training job. Stop after presenting the report.
