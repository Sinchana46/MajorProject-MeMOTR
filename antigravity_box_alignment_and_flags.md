# Antigravity Instructions: Box Alignment and Pre-Training Flags (read-only diagnostic)

Written 7 Oct 2026. Context: Parts A to D of `antigravity_closeout_final.md` are finished. The user opened `data/IOTD/check/dataset_samples/` and the boxes **do not sit on the vehicles**. Your earlier "PASS" for that step only meant the images were saved; alignment was never checked. The earlier overlays of the raw and augmented frames (`check/aug_overlays_v2/`) were correct, so the frames and `gt.txt` are fine. The fault is either in how the samples were drawn or in the loader's transforms. Find out which.

---

## 0. Rules

1. **Read-only diagnostic. Do not start training.** Do not modify any existing file: MeMOTR code, existing configs, the two new configs, original IOTD data, v1 or v2 outputs, junctions. Do not delete anything.
2. You may create only: new scripts named `tools/diag_*.py`, and new folders `data/IOTD/check/box_alignment_diagnostic/`, `data/IOTD/check/dataset_samples_v2/`, `data/IOTD/check/logs/`.
3. Your recent runs hung or went silent. So: run **each step as its own small script** with `python -u`, output redirected to `check/logs/<step>.txt`, and a hard 300 s timeout (`Start-Process ... -PassThru`, `WaitForExit(300000)`, kill and print `TIMEOUT` otherwise). Do not poll in loops, do not read `transcript.jsonl`, do not run echo or latency tests, do not retry a failed command more than once. After each step read the log and paste it.
4. Paste **raw output**. Label every claim **verified** (you ran it or read the code, with file and line) or **assumed**.
5. If anything differs from this document, or a step fails, stop that step, report the exact error and the line, and go on to the next independent step.

---

## Part A. Box alignment (do this first)

### A1. How the old samples were drawn

Read `tools/run_partC_and_partD.py`, the code that writes `check/dataset_samples/`. Quote the lines and answer:

- Which boxes were drawn: the raw `gt.txt` boxes, or `target["boxes"]` returned by the loader?
- What box format did the drawing code assume (pixel xyxy, pixel xywh, normalized cxcywh)?
- Was the image tensor de-normalized (mean/std) and converted from CHW RGB to HWC BGR before saving?

### A2. The real transform pipeline

Trace the complete path from `VehicleMOT.__getitem__` in `MeMOTR/data/vehiclemot.py` through `MeMOTR/data/transforms.py`. Instantiate the dataset for `split="train"` using `configs/train_iotd_real.yaml` and **print the exact transform list** (class names and arguments, in order).

### A3. Per-transform table

Fill one row per transform in the list:

| # | Transform | Geometric? | Updates boxes with the image? (file:line) | Box format in -> out |
|---|---|---|---|---|

Include every resize, random size crop, random crop, horizontal flip, scale, shift, rotation, colour transform, `ToTensor` and `Normalize`. State at which step boxes are converted (for example to **normalized cxcywh**, as in DETR-style code) and therefore **what format the loader finally returns**. State whether the same random parameters are applied to every frame in a clip.

### A4. Candidate causes (test in this order)

1. **Format:** the old code drew normalized cxcywh boxes as if they were pixel xyxy. If the returned format is normalized, decode with `cx,cy,w,h = box * [W,H,W,H]`, then `x1 = cx - w/2`, `y1 = cy - h/2`, `x2 = cx + w/2`, `y2 = cy + h/2`, using the **transformed** image's `W,H`.
2. **Wrong boxes:** raw `gt.txt` boxes were drawn on transformed images.
3. **Image conversion:** missing de-normalization or channel order that changed what was shown.
4. **Frame/box mismatch:** the target for frame t was paired with a different image in the clip (an indexing error).
5. **A real transform bug:** a crop, resize or flip that does not move the boxes consistently.

### A5. Diagnostic script `tools/diag_box_alignment.py`

Use the actual dataset object and its actual train transforms. Fixed seeds. Never edit the repo files; to substitute images for the paint test below, wrap or monkeypatch inside your own script only.

1. **Control:** one raw frame with its raw GT boxes (pixel xywh to xyxy), saved as `raw_control_*.jpg`.
2. **Loader output, drawn correctly:** 12 samples (4 from `seq_01_1min`, 4 from `seq_03_30sec_b`, 2 from an `aug` copy of each), **every frame of each clip**. For each: de-normalize the image, decode the returned boxes per the format you verified in A3, draw them, and write a header with ORIGINAL image shape, TRANSFORMED image shape, number of boxes. Save under `check/box_alignment_diagnostic/`.
3. **Old method for comparison:** for 3 of those samples, also save what the previous visualisation did (the wrong method you identified in A1), so the failure is visible next to the correct drawing.
4. **Numeric paint test (independent of the eye):** take the raw frame, paint a solid, distinctive colour (for example pure magenta) inside every GT box, and run it through the **same** train transform and normalization. De-normalize the output, build a mask of the painted colour, and compare with the decoded output boxes. For every output box at least 4 px on each side, compute (a) the share of the box area covered by the mask, and (b) overall, the share of mask pixels that fall outside all boxes. Run 200 draws spanning both orientations, real and aug copies, and both frames of the clip. Report the pass rate and the worst case. **Pass rule:** each evaluated box has coverage of at least 0.85, and at most 10 percent of mask pixels lie outside all boxes.
5. Print for the first 3 boxes of 3 samples: ORIGINAL image shape, TRANSFORMED image shape, ORIGINAL box coordinates (raw pixels), TRANSFORMED box coordinates (as returned, and decoded to pixels).

### A6. Decision

- **If the loader's boxes pass the paint test and the correctly drawn images look aligned:** the only fault was the old visualisation. Do **not** touch the training pipeline. Write a corrected drawing script as a **new** file, regenerate the six samples into `check/dataset_samples_v2/` (do not overwrite `dataset_samples/`), and state the root cause in one sentence.
- **If the paint test fails or the boxes are visibly off:** **STOP.** Report the exact transform that causes it (name, file and line), the shapes and coordinates before and after, and the failing share. Do not fix it, do not continue to B or C until the user replies.

Also report (information only, do not change anything): which transforms are geometric, and whether the training pipeline includes a random crop, a scale change or a **horizontal flip**. The offline augmentation deliberately avoided these because C1 depends on box height against vertical position; say whether the loader's own crop, scale or flip would disturb that relationship.

---

## Part B. Flag 1: fair ablation (read-only)

The verified clip-start counts are **2528** (real only) and **10112** (real + aug), both configs at 5 epochs. Do not accept "20 epochs vs 5 epochs" without checking the following.

### B1. Read the schedule and curriculum

Print every config key (with file and line) that controls: epochs, batch size, accumulation steps, learning rate and backbone LR, LR scheduler type, LR drop epochs or milestones, `SAMPLE_LENGTHS`, `SAMPLE_INTERVALS`, the epochs at which the sample length changes, sampling mode, and whether `sample_begin_frames` is rebuilt each epoch. Then read `train_engine.py` and `vehiclemot.py` and state:

- optimizer steps per epoch = clip starts / (batch size x accumulation)
- how the clip length (`sample_length`) changes with the epoch, and how clip starts change with it
- whether the LR schedule and the curriculum are indexed by **epoch number** (so changing the epoch count changes where the LR drops and where clip length increases)

### B2. Compute a table

For real-only and real + aug, at 5, 10 and 20 epochs: clip starts per epoch, optimizer steps per epoch, **total optimizer steps**, total frames seen (clip starts x sample length), and the epochs at which the LR drops and the clip length changes.

### B3. Recommend a fair setup

Explain whether **equal optimizer steps** or equal epochs is the appropriate control, and whether the 20-epoch run would **also** reach longer clips or an LR drop at a different relative point than the 5-epoch run (a second difference between the arms that is not about augmentation). Propose exact YAML values for both configs so that total optimizer steps match **and** the LR schedule and clip-length curriculum fall at the same fraction of training in both. If that is impossible with epoch-indexed keys, say so and give the closest option. **Do not apply any change.** Also state that both arms should use the same random seed.

---

## Part C. Flag 3: starting checkpoint (read-only)

Trace from `configs/train_iotd_real.yaml` and `configs/train_iotd_real_aug.yaml` through `MeMOTR/main.py` to model construction and weight loading. Print every config key whose name contains PRETRAIN, RESUME, CHECKPOINT, MODEL, WEIGHT or LOAD (note that `USE_CHECKPOINT` and `CHECKPOINT_LEVEL` may mean activation checkpointing, not weights; say which). Report:

1. The effective path of the weights loaded at start-up, and the code path that loads them (file:line)
2. Whether the file exists; its size and modification time
3. Metadata from the file: epoch, keys present (model, optimizer, scheduler), number of parameters, and a few parameter-key names
4. Whether it is the UA-DETRAC + VisDrone fine-tuned checkpoint (list the candidate files under `results/phase5_interim_baseline/` with size, epoch and time) or only the original base MeMOTR checkpoint
5. Whether the loader restores **only the model weights**, or also the optimizer, scheduler and **start epoch**. If it resumes at the saved epoch, say what would happen with `EPOCHS: 5` and a checkpoint saved at epoch 4 (it could train zero epochs)
6. Whether any IOTD output or checkpoint exists or would be overwritten (`results/train_iotd_real`, `results/train_iotd_real_aug`)

If the current config is ambiguous or points to the wrong file, **propose** the exact key change and value, but do not apply it.

---

## Part D. Evaluation path on the validation sequence (read-only, do not run inference)

The earlier report found no validation loop during training; evaluation is `--mode submit` then TrackEval. Report, with file and line numbers:

- The exact command and the config keys needed to run submit on `IOTD_val` using a given checkpoint
- The folder layout and files the submit and TrackEval code expect (`img1/`, `gt/gt.txt`, `seqinfo.ini`, sequence map, split name) and whether `data/IOTD_val/` (a directory junction) satisfies it
- The inference-time resize: is it fixed shortest side with `max_size`? What does a 1080x2336 portrait sequence become, and is it the same as in training?
- Which metrics are computed (HOTA, AssA, IDF1, MOTA, IDSW or others), and whether the detection score threshold is a config key

Do not execute inference or evaluation.

---

## Part E. Final report (compact, then stop)

```
1. BOX ALIGNMENT: PASS / FAIL
   Root cause (one sentence):
   Paint test: pass rate, worst case:
   Diagnostic images: <path>   Corrected samples: <path or "not created">
2. TRANSFORMS: exact list; which are geometric; are boxes transformed
   correctly: YES/NO; returned box format; flip present: YES/NO
3. FLAG 1: steps real-only = ?  aug = ?  at 5 epochs.
   Recommended fair setup (exact keys and values, not applied):
   Curriculum / LR-drop confound found: YES/NO
4. FLAG 3: actual starting checkpoint = ?
   Intended UA-DETRAC + VisDrone checkpoint: YES/NO
   Weights-only or full resume: ?
   Proposed change (not applied):
5. EVAL PATH: command, keys, layout compatible with IOTD_val: YES/NO
6. Anything unverified or assumed:
```

Then stop and wait for the user. Do not start training, do not delete v1, do not modify original IOTD data, do not silently change the augmentation, dataset or configs.
