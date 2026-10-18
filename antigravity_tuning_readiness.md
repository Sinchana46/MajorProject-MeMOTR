# Antigravity Instructions: Readiness for Hyperparameter Tuning

Written 9 Oct 2026. Goal: settle the evaluation and annotation issues, measure run-to-run variability, define a fair comparison protocol, and produce a tuning plan. **The search itself is not part of this task.**

Dataset preparation, v2 augmentation, both training arms, the box-alignment investigation (paint test passed, 3,425/3,425) and the checkpoint/threshold evaluation are **done**. Do not repeat them, do not restart augmentation or any existing training run.

---

## 0. Rules

1. **Do not delete, move, rename or overwrite anything.** Do not modify: MeMOTR code, existing configs, `data/IOTD/` originals (`gt.txt`, `labels.txt`, `seqinfo.ini`, JPEGs), v1 or v2 augmentation folders, existing `results/*`, existing `tools/*.py`, existing logs and reports.
2. Allowed new outputs only: `tools/tune_*.py`, `check/annotation_audit/`, `check/logs/`, `results/eval_iotd_val/rescored/`, `results/eval_iotd_train_audit/`, new configs `configs/train_iotd_real_seed*.yaml`, and the final report `data/IOTD/check/hyperparameter_tuning_readiness.md`.
3. **Do not edit any annotation file until the user approves** (GATE 1). **Do not launch any training run until the user approves** (GATE 2).
4. Run **each step as its own script** with `python -u`, output redirected to `check/logs/<step>.txt`, hard timeout (300 s; 1800 s per inference run), no polling loops, no echo or latency tests, no reading `transcript.jsonl`. Retry a failed command at most once; if a step fails, report the exact error and continue with the next independent step.
5. Paste **raw output**. Label every claim **verified** (you ran it or read the code, with file:line) or **assumed**. Do not write "PASS" unless you show the evidence. Check free GPU memory before any inference.
6. Re-hash all real IOTD files against `check/hashes_before.json` (thread pool) at the start and end: `changed`, `missing`, `added` must be 0.

---

## Part A. Experiment inventory (read-only, reuse existing records)

Verify the following **claims** against logs, configs and result files. Do not rely on filenames or comments alone. Report each as verified or contradicted, with the source.

| Name | Meaning (claimed) |
|---|---|
| Phase 5 | `results/phase5_interim_baseline/checkpoint_4.pth`, UA-DETRAC + VisDrone fine-tune of the DanceTrack-pretrained MeMOTR, no IOTD data |
| Arm 1 | `configs/train_iotd_real.yaml`, IOTD real only (`IOTD_train_real`: seq_01 + seq_03), start = Phase 5 `checkpoint_4`, weights-only, 5 epochs, 2,528 steps per epoch, LR 2e-5 for epochs 0-3 and 2e-6 in epoch 4, backbone frozen, batch 1 |
| Arm 2 | `configs/train_iotd_real_aug.yaml`, real + six v2 copies, 10,112 steps per epoch, otherwise identical to Arm 1 |

Claimed results on `seq_02_30sec_a` (det 0.5, track 0.5, miss tolerance 30, `use_c1=False`, `use_c2=False`, evaluation-only GT with all classes set to 1):

| Run | Steps | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5 | 0 | 50.51 | 43.88 | 58.49 | 29.10 | 68.84 | 1 | 2,117 | 847 |
| arm1 ckpt_3 | 10,112 | 50.47 | 46.96 | 54.51 | 29.63 | 70.23 | 9 | 2,629 | 305 |
| arm1 ckpt_4 | 12,640 | 49.58 | 45.70 | 54.04 | 24.94 | 69.66 | 7 | 2,855 | 277 |
| arm2 ckpt_0 | 10,112 | 40.51 | 36.49 | 45.15 | 16.28 | 56.63 | 10 | 1,964 | 1,527 |
| arm2 ckpt_1 | 20,224 | 48.17 | 45.54 | 51.14 | 29.84 | 68.64 | 12 | 2,432 | 490 |
| arm2 ckpt_4 | 50,560 | 42.84 | 38.32 | 48.01 | -6.58 | 63.72 | 6 | 4,266 | 185 |

Also report for **every existing run and checkpoint**: the random seed (is a seed set at all? where, `main.py` or config), the training duration and **measured seconds per optimizer step** from the logs (claimed about 0.67 s per step: Arm 1 2h22m, Arm 2 9h27m), the scheduler type and milestones, the clip sample length actually used per epoch, which inference thresholds produced each reported metric (the 0.7 and 0.8 results exist for the final checkpoints only), and the locations of all saved tracker output files. State whether **any independent-seed repeat already exists** (search `results/` and the logs). Also list every result folder that exists and what produced it.

---

## Part B. Annotation audit (read-only, produces proposals only)

**Known problem:** renders of `seq_02_30sec_a` frames 298 and 447 show real vehicles on the monitored carriageway with no GT box (far traffic up the road, and two motorbikes in frame 447 that the fine-tuned runs detect). The documented policy is: only vehicles on the monitored carriageway are annotated; opposite-carriageway and out-of-region vehicles are not. Two frames cannot establish a cutoff, so measure it.

### B1. Candidate generator

- `seq_02_30sec_a`: use the three existing final tracker outputs (phase5, arm1, arm2).
- `seq_01_1min` and `seq_03_30sec_b`: no tracker output exists. Run **Phase 5 only** (it never saw IOTD, so it is the unbiased candidate generator; do not use the fine-tuned arms here, they were trained on these labels) with the same inference tool and thresholds, output to `results/eval_iotd_train_audit/phase5/`. Report the inference time. Use the same resize path as training evaluation; seq_03 is portrait 1080x2336.

### B2. Candidate missing-annotation tracks

For each sequence, find predicted tracks of at least 15 frames in which at least 80% of the frames have max IoU with any GT box below 0.2. For `seq_02` also record how many of the three runs agree (a track is "agreed" by a run if its box overlaps with IoU >= 0.5 on at least half of the frames).

### B3. Triage by motion and position (evidence, not a decision)

For every candidate track compute: frame range, mean box size, bottom-centre trajectory, **vertical travel sign** (moving down the image means approaching the camera, as in `tools/seq_profile.py`), and its relation to the GT: lateral overlap with the GT hull, and distance to the nearest GT track. Compare its direction with the majority direction of the GT tracks in that sequence. Assign a **suggested** class and never a final one:

- `LIKELY_OPPOSITE`: moves opposite to the majority of GT tracks and lies outside the GT lane span
- `LIKELY_MISSING`: same direction as the GT majority, inside or adjacent to the GT lane span
- `UNCERTAIN`: anything else, or fewer than 15 frames, or heavily occluded

### B4. Is there a consistent cutoff? (statistics over the whole sequence)

For each sequence and for all frames, compute the distribution of the **bottom edge `y + h` of GT boxes** (minimum, 1st, 5th percentile) and of the bottom edge of the `LIKELY_MISSING` candidates. Report per 100-frame window: the smallest GT bottom-edge `y`, the largest candidate bottom-edge `y` above it, and whether the two ranges overlap. State plainly whether the data supports a **fixed cutoff** (GT never extends above some `y` across the whole sequence while many same-direction tracks exist above it), a **moving or soft boundary**, or **incomplete annotation** (GT and candidates interleave). Do not create an exclusion region because it would improve scores.

### B5. Evidence and proposal outputs

Under `check/annotation_audit/`:

- `proposals.csv` with columns: sequence, candidate id, frame start, frame end, frames, mean box (x, y, w, h), runs agreeing, suggested class, evidence image path, proposed action (`REVIEW_ADD` / `REVIEW_EXCLUDE_BY_POLICY` / `REVIEW_UNCERTAIN`; never `APPLIED`)
- evidence images: for each of the top 40 candidates per sequence (ranked by persistence and size), a contact sheet with three frames (start, middle, end), the candidate box in cyan, the GT boxes in green, and a full-frame thumbnail
- 3 per-sequence overview images of GT bottom-edge versus frame, with candidate tracks overlaid
- `summary.md` with counts per sequence and per suggested class, and the B4 conclusion
- **CVAT-importable pre-annotations** (MOT 1.1 zip) of the `REVIEW_ADD` candidates in `check/annotation_audit/cvat_import/`, track IDs starting above the existing maximum for that sequence, for the user to review and correct in CVAT. **Do not apply them to any `gt.txt`.**

### GATE 1 (user)

Print the B4 conclusion, the per-sequence candidate counts and the paths to the evidence. Then **stop and wait**. The user will reply with one of:

- **X, policy only:** a written exclusion rule for evaluation (GT unchanged, applied identically to all runs). Continue with Part C, then Part D.
- **Y, correct annotations:** the user re-annotates in CVAT. Stop after this part; a follow-up prompt will cover applying corrections. Say which trained models would become invalid (any model trained on changed training labels: all existing Arm 1 and Arm 2 checkpoints if training GT changes; Phase 5 is unaffected) and that the seeds in Part D must not be run on old labels.
- **Z, both.**

---

## Part C. Re-score existing predictions (only after GATE 1, branch X)

1. Locate the saved predictions for Phase 5, Arm 1 (ckpt_3 and ckpt_4) and Arm 2 (ckpt_1 and ckpt_4). State which can be reused (same inference thresholds) and which need inference again.
2. Apply the user's approved rule **identically to every run**, by filtering tracker output rows (the rule must be written down in the report). The GT stays unchanged. Use the same evaluation-only class-1 GT copy and the same TrackEval settings as before.
3. Save to `results/eval_iotd_val/rescored/<run>/` and **keep all earlier results untouched**. Print one table: HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN, Recall, Precision, for all runs, original score beside corrected score.
4. Say whether the ranking changes.

---

## Part D. Training variability (prepare; launching needs GATE 2)

Only if GATE 1 gave branch X (training labels unchanged).

1. Inspect `main.py`, the config and the engines for how randomness is controlled: Python `random`, NumPy, `torch` CPU and CUDA, `torch.backends.cudnn` flags, DataLoader `worker_init_fn` and generator, and any random transforms (the training pipeline includes random resize and crop). State what is seeded today and what a new seed value would change.
2. Prepare two new configs `configs/train_iotd_real_seed1.yaml` and `..._seed2.yaml`, copied from `train_iotd_real.yaml`, changing **only** the seed (and only the output directory and label so nothing is overwritten). Same Phase 5 `checkpoint_4` start, weights-only, same data, schedule, batch size, 5 epochs. Print a diff of each against `train_iotd_real.yaml`. If the original Arm 1 had no explicit seed, say so and state what seed value its randomness effectively had.
3. From the measured seconds per step in the logs, give the expected wall time per run (12,640 steps), the disk needed (about 575 MB per checkpoint, five per run), and the evaluation cost (five checkpoints per run).
4. Prepare an evaluation script that scores all five checkpoints of each repeat under the Part C protocol, and prints mean, standard deviation and range across Arm 1 and its two repeats at **each** checkpoint (same step count), stating clearly that three runs give only a rough estimate.

### GATE 2 (user)

Show the configs, the diffs and the time estimate, then **stop**. Launch nothing until the user replies "launch seeds". When launched, run the two repeats one after the other, each as a separately logged background task, with no polling loop.

---

## Part E. Fair real-only versus augmented comparison

Do not run anything new. Inspect and report, with file:line:

1. How an epoch and an optimizer step are defined in `train_engine.py` and `vehiclemot.py` (batch size 1, accumulation 1, number of clip starts per epoch).
2. Whether the learning-rate schedule is epoch-indexed or step-indexed, the milestone epochs, and whether the clip-length curriculum (`SAMPLE_LENGTHS [2,3,4,5]`) changes within 5 epochs (the iteration counts suggest it stays at length 2).
3. A **protocol** for comparing real-only against real + augmented at **equal optimizer steps**, using the existing checkpoints: Arm 1 `checkpoint_3` and Arm 2 `checkpoint_0` (both 10,112 steps, both before any LR drop), and a statement of the limits of that comparison (the LR drop falls at a different point of each run's total length; one validation sequence; single seed unless Part D is launched). State whether a 20-epoch real-only run is still needed, and what it would cost (about 50,560 steps). The existing evidence is that Arm 2 degrades past about 20,000 steps, so say what a 20-epoch run would add.

---

## Part F. Tuning plan (document only; do not launch the search)

Produce the proposed plan with measured costs:

| Parameter | Current | Candidates | Needs retraining? | Cost per setting | Selection criterion |
|---|---|---|---|---|---|

Cover:

- **A. Training length / checkpoint selection:** from the existing checkpoint curves, the best step range for each arm, and a **checkpoint-selection rule written before further runs** (for example: the checkpoint with the highest HOTA among the first five epochs, at the default thresholds).
- **B. Non-backbone learning rate:** current 2e-5; candidates 1e-5 and 5e-5. Recommend the smallest search the measured variability can resolve.
- **C. Inference thresholds:** the tracker's confidence column is a constant 1, so scores cannot be filtered after the fact; each threshold needs inference again. Give the measured cost per run, and propose a bounded sweep (for example 0.5, 0.6, 0.7), reusing the existing 0.7 and 0.8 results where valid. State that a threshold sweep is not a training run, and that choosing a threshold on `seq_02` is still tuning on the validation sequence.
- **D. C1 and C2:** keep out of the initial search. Report as read-only findings, with code references:
  - **C1:** whether the perspective calibration exists and which frames and sequences it used (it must use original frames only; see `split_manifest_v2.json`); the direction of the PositionMLP weighting in the code (extra weight for near-camera boxes or for distant, small ones), and note that the project report contradicts itself on this
  - **C2:** how the training target of the adapter is computed (quote the label-generation code). State whether the target depends only on brightness, in which case the adapter cannot learn any altitude effect. State the tau range in code (the reports disagree: [0.3, 0.7] or [0.2, 0.7])
  - the prerequisites that remain for each, as PASS, BLOCKED or NEEDS APPROVAL
- **Not to prioritise:** augmentation strength.

### Validation limits

State that there is one validation sequence (744 frames, 50 tracks, night) and no test set; repeated tuning on it is not independent test performance. Estimate the cost of **leave-one-sequence-out** (train on two sequences, validate on the third, three folds) from the measured step time, per setting and for the full plan, and recommend whether it should be mandatory, optional for the final candidate only, or skipped. Do not change the split.

---

## Part G. Final report (then stop)

Write `data/IOTD/check/hyperparameter_tuning_readiness.md` with:

1. Experiment inventory and exact Arm 1 definition (Part A), including contradicted claims
2. Annotation-audit findings and status, with counts and the B4 conclusion; the user's GATE 1 decision
3. Re-scored predictions and updated metrics (if branch X), original beside corrected
4. Seed-repeat status and measured variability (or "not launched")
5. Step and scheduler analysis (Part E)
6. The fair comparison protocol
7. The proposed hyperparameter table
8. Run-budget estimate from measured step time
9. Checkpoint-selection rule and validation limitations
10. A status for each prerequisite: **PASS**, **BLOCKED** or **NEEDS APPROVAL**, with the reason

Definition of ready: the annotation and evaluation protocol is settled; the metrics are reproducible enough to interpret (variability measured); training exposure is understood; the proposed search fits the compute budget.

Then stop. Do not start the search. Do not delete v1 or v2 folders, modify unrelated code, change the architecture, or overwrite any existing output. Required caveats at the end of the report: one validation sequence, a small number of repeats, and augmented frames are training samples, not additional annotated data (real annotated data: 3,274 frames).
