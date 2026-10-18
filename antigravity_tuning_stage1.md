# Antigravity Instructions: Tuning Stage 1 (seeds + two training variants, unattended queue)

Written 11 Oct 2026. Read with `data/IOTD/check/hyperparameter_tuning_readiness.md` and `archive/pre_tuning_baseline_v1/` (the frozen baseline package, if it exists).

## What this stage does

Four training runs, one after another, each followed by the evaluation of all five checkpoints, started by **one detached queue script** so that nothing depends on you polling:

| Job | Config | Change from Arm 1 | Why |
|---|---|---|---|
| `seed1` | `configs/train_iotd_real_seed1.yaml` (seed 1234) | seed only | measure run-to-run noise |
| `seed2` | `configs/train_iotd_real_seed2.yaml` (seed 2024) | seed only | measure run-to-run noise |
| `assoc_lr0` | new `configs/train_iotd_real_assoc_lr0.yaml` (seed 42) | learning rate of the **query updater** (association / memory update) set to 0, everything else as Arm 1 | Fine-tuning raised DetA about 3 points but lowered AssA about 4 and raised ID switches from 1 to 9. Test whether freezing the association module removes that loss |
| `lr_half` | new `configs/train_iotd_real_lr_half.yaml` (seed 42) | every non-zero learning rate halved, ratios kept (2e-5 becomes 1e-5) | test whether less drift from the Phase 5 weights helps |

Reference run for all comparisons: **Arm 1** (`configs/train_iotd_real.yaml`, seed 42, results already evaluated). Its scores are in the baseline registry; do not retrain it.

**Out of scope:** the inference-time sweep (a later stage), C1, C2, augmentation changes, any change to data, split, code or existing configs.

---

## 0. Rules

1. **Do not delete, move, rename or overwrite anything.** Do not modify: any dataset file, annotation file, existing config, existing code, existing checkpoint, existing result or report, and **nothing under `archive/`** (the frozen pre-tuning package is read-only). New outputs only under `results/tuning_stage1/`, `configs/` (the two new files), `tools/tune_*.py`, `check/logs/`.
2. Do not change the model architecture, the backbone or the encoder. The backbone stays frozen (LR 0) in every job.
3. **No resume.** Each job starts from `results/phase5_interim_baseline/checkpoint_4.pth` as weights only, exactly like Arm 1. If a job's output folder already exists, stop and ask.
4. **Do not poll.** After launching the queue and doing one start-up check (section 4), **end your turn**. Do not write loops that wait for completion, do not run echo or latency tests, do not read `transcript.jsonl`.
5. Run every step as its own script with `python -u`, output to `check/logs/tune1_<step>.txt`, hard timeout 300 s (except the queue itself). Retry a failed command at most once. Paste raw output. Label claims **verified** (you read the file or ran the check) or **assumed**.
6. Re-hash all real IOTD files against `check/hashes_before.json` before launching and when reporting (thread pool). `changed`, `missing`, `added` must be 0.
7. Do not change Windows power settings. The queue driver itself must keep the machine awake while it runs (section 3, item 4).

---

## 1. Verify before building anything (read-only)

1. Free GPU memory (`nvidia-smi`), no other training or inference process running (`Get-Process python`), free disk. Each job writes five checkpoints of about 575 MB, so four jobs need about 12 GB plus evaluation outputs. **Stop if free disk is under 30 GB.**
2. Find how Arm 1 was launched (the exact command line, environment variables such as `PYTHONPATH`, and working directory) in the logs and the `manage_task` history. Print it. The queue uses the same command with only the config changed.
3. Diff `configs/train_iotd_real_seed1.yaml` and `..._seed2.yaml` against `configs/train_iotd_real.yaml`: only the seed, the output directory and the label may differ. Print the diffs. Confirm that the seed key is actually read by `main.py`, and state precisely what it seeds (Python, NumPy, torch CPU and CUDA, DataLoader workers) and what it does not. Confirm the start checkpoint is `results/phase5_interim_baseline/checkpoint_4.pth` and that loading is weights only (`RESUME: None`). Confirm `OUTPUTS_DIR` in each of the three files is different from Arm 1's and from each other.
4. **How the learning rates are built.** Read `main.py` and the optimizer code. Print which config keys set `lr`, `lr_backbone`, `lr_points`, `lr_query_updater` (use the real names), the scheduler, its milestones and gamma, the weight decay, and **which parameter names fall in each group**, with the parameter count per group. Then do a **read-only optimizer dry check**: build the optimizer from the Arm 1 config without training (CPU if the model fits, otherwise by reading the code) and print each parameter group's name, parameter count and learning rate. State whether the "query updater" group is the memory-update module that carries the track queries between frames, and whether the decoder (which is shared by detection and association) sits in another group. If the groups do not match this description, say so and stop; do not guess.

---

## 2. Create the two new configs (and only those)

Copy `configs/train_iotd_real.yaml` (seed 42):

- `configs/train_iotd_real_assoc_lr0.yaml`: set the query-updater learning rate to 0 and nothing else; new `OUTPUTS_DIR = D:/MajorProject-MeMOTR/results/tuning_stage1/assoc_lr0` and a new label
- `configs/train_iotd_real_lr_half.yaml`: multiply every non-zero learning rate by 0.5, keeping their ratios (for example 2e-5 to 1e-5, and the points group scaled the same way); the backbone stays 0; new `OUTPUTS_DIR = D:/MajorProject-MeMOTR/results/tuning_stage1/lr_half` and a new label

Set `OUTPUTS_DIR` for the two seed jobs by passing them to new copies only if their existing configs point to a clashing folder; **never edit the existing seed configs in place**. If an output directory must change, create `configs/train_iotd_real_seed1_s1.yaml` and `..._seed2_s1.yaml` as copies with only the output directory changed, and use those.

Print a diff of every config used by the queue against `train_iotd_real.yaml`. Run the optimizer dry check (section 1.4) on `assoc_lr0` and `lr_half` and print the groups and learning rates; the query-updater group must show 0 in the first and half of its Arm 1 value in the second.

---

## 3. Build and launch the queue

Write `tools/tune_queue.py` (a plain Python driver, no new packages). It runs the four jobs **sequentially** in the order `seed1`, `seed2`, `assoc_lr0`, `lr_half`, and for each job:

1. confirm free disk and an idle GPU; refuse to start if the job's output folder exists and is not empty
2. run training with Arm 1's exact launch command and the job's config, stdout and stderr to `results/tuning_stage1/logs/<job>_train.log`
3. verify that five checkpoints `checkpoint_0.pth` to `checkpoint_4.pth` exist and each is about 575 MB, and that the log shows 2,528 iterations per epoch and `seq_02_30sec_a` absent; record the result
4. evaluate **all five checkpoints** on `data/IOTD_val/seq_02_30sec_a` with exactly the protocol of the baseline package: same inference tool and the config `configs/train_iotd_real.yaml` for inference in every job, detection and tracking threshold 0.5, miss tolerance 30, `use_c1=False`, `use_c2=False`, the same evaluation-only class-1 GT, TrackEval with `DO_PREPROC=False`. Scoring **S0**; if the corrected-GT scoring S1 exists, score S1 from the same predictions as well. Outputs under `results/tuning_stage1/eval/<job>/ckpt_<N>/`
5. append the metrics to `results/tuning_stage1/stage1_results.csv` using the **same columns as the baseline registry** (run, checkpoint, steps, epochs completed, seed, config, scoring, threshold, GT SHA256, HOTA, DetA, AssA, MOTA, IDF1, IDSW, TP, FP, FN, recall, precision, summary path, summary SHA256)
6. compute the parameter-difference check against Phase 5 `checkpoint_4` for the final checkpoint: backbone maximum absolute difference (must be exactly 0); for `assoc_lr0`, also the maximum absolute difference of the **query-updater parameters** (must be exactly 0); for every job, the mean absolute difference over the identical set of non-backbone parameters
7. update `results/tuning_stage1/status.json` after every phase (job, phase, start and end time, result, error text if any)

Failure handling: if a phase fails, record the error in `status.json` and the log, **do not retry endlessly, continue with the next job**. A crash in training is recorded and the job's evaluation is skipped. The driver must never delete a partial output. On restart it skips jobs whose five checkpoints and evaluation rows already exist.

The driver must also:

- write its PID to `results/tuning_stage1/queue.pid` and a heartbeat timestamp to `status.json` at least every minute
- **keep the machine awake while running** by calling `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` through `ctypes` at start and clearing it on exit (this does not change any system setting)
- print, at the very start, the measured steps-per-second from the earlier Arm 1 log, and an estimate of the total queue time (training about 0.66 s per step times 12,640 steps is about 2h20m per job; add the measured inference time per checkpoint times 20)

Launch it detached so it survives your session: `Start-Process -FilePath <python> -ArgumentList '-u','tools/tune_queue.py' -WindowStyle Hidden -RedirectStandardOutput results/tuning_stage1/logs/queue.out -RedirectStandardError results/tuning_stage1/logs/queue.err -PassThru`, with the same `PYTHONPATH` as before.

---

## 4. One start-up check, then end your turn

Two minutes after launch, read the first lines of the first job's training log once and report: the PID, the log paths, `Loaded N sequences` (expected 2), the weights-loaded line, the first logged losses, and the estimated completion time of each job. Then **stop and end your turn**. Do not poll, do not wait. The user will come back and ask for the report.

---

## 5. Report (the user will ask for it later; run only when asked)

When the user says the queue has finished, do **not** retrain anything. Read `status.json` and `stage1_results.csv`, re-hash the originals, and write `results/tuning_stage1/stage1_report.md` with:

1. A job table: status, wall time, any failures
2. **Seed spread.** For each checkpoint (steps 2,528, 5,056, 7,584, 10,112, 12,640), the HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN of Arm 1 (seed 42, from the baseline registry), `seed1` and `seed2`, and their mean, standard deviation and range. State that three runs give only a rough estimate
3. **Each variant against that spread**, at every checkpoint: value, Arm 1 value (paired, same seed 42), difference, and difference expressed in seed standard deviations. Also the best checkpoint of each run under the written rule (highest HOTA among the five checkpoints at 0.5) and its metrics
4. **Association check for `assoc_lr0`:** AssA, IDF1 and IDSW against Arm 1 and against Phase 5 at equal steps, and whether DetA changed. Confirm the query-updater difference to Phase 5 is exactly 0
5. **Drift check for `lr_half`:** the mean absolute parameter difference to Phase 5 against Arm 1's, and the metrics
6. FP, FN, precision and recall trends for all runs
7. **Decision rule, applied mechanically:** a variant counts as better than Arm 1 only if its HOTA exceeds the **Arm 1 seed mean by more than two seed standard deviations at more than one checkpoint**, without a worse AssA or IDSW. Otherwise the default stays and the result is reported as negative. State the outcome for each variant, and a recommendation for the next step (the inference-time sweep, then whether any training variant deserves a repeat)
8. Caveats: one validation sequence, three seeds for the spread and a single run per variant, tuning on the validation sequence, augmented frames are not additional annotated data (real annotated data 3,274 frames, 2,530 real training frames)

Then stop. Do not start the inference sweep, C1, C2 or any further training.
