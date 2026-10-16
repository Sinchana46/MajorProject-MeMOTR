# REPRODUCE.md — Environment Setup & Patch Application

This file documents everything needed to reproduce the development environment
from a fresh clone of this repository.

## 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-org>/MajorProject-MeMOTR.git
cd MajorProject-MeMOTR
```

Or if already cloned without submodules:

```bash
git submodule update --init --recursive
```

This will check out:

| Submodule | Upstream repository | Pinned commit |
|-----------|--------------------|----|
| `MeMOTR/` | https://github.com/MCG-NJU/MeMOTR.git | (see `.gitmodules`) |
| `TrackEval/` | https://github.com/JonathonLuiten/TrackEval.git | (see `.gitmodules`) |

---

## 2. Apply required patches

Patches live in `patches/`. Apply them **after** submodule init, before running anything.

### Patch 001 — TrackEval NumPy ≥ 1.24 compatibility (required)

```bash
cd TrackEval
git apply ../patches/001-trackeval-numpy-compat.diff
cd ..
```

**What it fixes:** `np.float`, `np.int` were removed in NumPy 1.24 (deprecated since 1.20).
Without this patch, every TrackEval evaluation call fails immediately with
`AttributeError: module 'numpy' has no attribute 'float'`.

**Files changed:**
- `trackeval/datasets/mot_challenge_2d_box.py` — 4 replacements
- `trackeval/metrics/hota.py` — 5 replacements
- `trackeval/metrics/identity.py` — 3 replacements

**Verified against:** NumPy 1.24+, Python 3.10, MOT17-02-FRCNN val, DanceTrack val.

---

### Patches 002 and 003 — not used

The numbers were reserved for C1/C2 before they were implemented. The final C1 and C2
changes are patches 007 and 008 below; no files named 002 or 003 exist.

### Patch 004 — MeMOTR optional `LR_QUERY_UPDATER` key (required for tuning variant `assoc_lr0`)

```bash
cd MeMOTR
git apply ../patches/004-memotr-lr-query-updater.diff
cd ..
```

In `train_engine.py:get_param_groups`, the query-updater parameter group reads
`config.get("LR_QUERY_UPDATER", config["LR"])` instead of `config["LR"]`.
When the key is absent (every config except `configs/train_iotd_real_assoc_lr0.yaml`)
behaviour is unchanged. `assoc_lr0` sets it to `0.0` to freeze the query updater.

### Patch 005 — MeMOTR VehicleMOT dataset + Python 3.9 compatibility (required for every IOTD / Phase 5 run)

```bash
cd MeMOTR
git apply ../patches/005-memotr-vehiclemot-py39.diff
cd ..
```

Adds the `VehicleMOT` dataset (new file `data/vehiclemot.py`, registered in `data/__init__.py`, one class in
`models/criterion.py` and `models/memotr.py`) and `from __future__ import annotations` to the eight upstream files
that use `X | Y` type syntax, which Python 3.9 rejects otherwise (`data/transforms.py`, `log/log.py`, `log/logger.py`,
`models/criterion.py`, `models/memotr.py`, `structures/track_instances.py`, `train_engine.py`; `models/query_updater.py`
gets it in patch 007). Required by every config with `DATASET: VehicleMOT` (Phase 5, Arm 1, Arm 2, Tuning Stage 1).
Touches `train_engine.py` at a different place than patch 004; the two apply in either order.

### Patch 006 — MeMOTR Deformable Attention build on Windows (required to build the CUDA op on Windows)

```bash
cd MeMOTR
git apply ../patches/006-memotr-ops-setup-windows.diff
cd ..
```

`models/ops/setup.py` passes `-ccbin=<MSVC Hostx64/x64>` and `-allow-unsupported-compiler` to nvcc on Windows
(MSVC 14.42 / VS 2022 with CUDA 13.0; the MSVC path comes from `VCToolsInstallDir`). Required before section 4
on Windows; no effect on Linux.

### Patch 007 — MeMOTR C1 flag in the query updater (required for every run; C1 itself is off by default)

```bash
cd MeMOTR
git apply ../patches/007-memotr-c1-query-updater.diff
cd ..
```

`models/query_updater.py` gets a `use_c1` flag (config key `USE_C1`, default `False`). With it on, the long-memory
update uses a per-track lambda from `modules/position_decay.py` (perspective zone + box height) instead of the fixed
`LONG_MEMORY_LAMBDA`. With it off (every experiment so far) no parameters are added and the update is identical to
upstream. Also adds `from __future__ import annotations` for Python 3.9. Imports `modules/` from the main repo,
so it is needed even with C1 off.

### Patch 008 — MeMOTR C2 flag in the runtime tracker (required for every inference run; C2 itself is off by default)

```bash
cd MeMOTR
git apply ../patches/008-memotr-c2-runtime-tracker.diff
cd ..
```

`models/runtime_tracker.py` gets a `use_c2` flag (default `False`). With it on, the detection threshold for new tracks
comes from `modules/threshold_adapter.py` applied to the `modules/scene_analyzer.py` scene vector, clamped to
[0.3, 0.7]. With it off, the fixed `DET_SCORE_THRESH` is used, as upstream. Imports `modules/` from the main repo,
so it is needed even with C2 off. Required by `tools/run_inference_mot.py` and the Phase 6 C2 experiments.

---

## MeMOTR submodule state

- **Upstream commit the submodule is pinned to:** `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` (MCG-NJU/MeMOTR `main`, "Fix a typo (#34)").
- **Patches to apply after cloning:** TrackEval `001`; MeMOTR `004`, `005`, `006`, `007`, `008`.
- Nothing is ever committed inside the submodules; all changes live in `patches/`.
- Verified 2026-10-16: applying 004–008 to a pristine `git archive` of `eb7a177` reproduces the working copy
  file-for-file (`check/logs/submodule_audit_3_4_verify.txt`).

Restore the working state from scratch:

```bash
git submodule update --init MeMOTR TrackEval
cd TrackEval
git apply ../patches/001-trackeval-numpy-compat.diff
cd ../MeMOTR
git apply ../patches/004-memotr-lr-query-updater.diff
git apply ../patches/005-memotr-vehiclemot-py39.diff
git apply ../patches/006-memotr-ops-setup-windows.diff
git apply ../patches/007-memotr-c1-query-updater.diff
git apply ../patches/008-memotr-c2-runtime-tracker.diff
cd ..
```

---

## 3. Create and activate the Conda environment

```bash
conda create -n memotr_project python=3.10
conda activate memotr_project
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install opencv-python
```

---

## 4. Build Deformable Attention CUDA ops

**Windows (VS 2022 + CUDA 13.0):**

```bat
build_cuda_ops.bat
```

This script uses `%CONDA_PREFIX%\python.exe` automatically when a Conda
environment is active, and `%~dp0` to find the project root — no hardcoded
paths.

**Linux/macOS:**

```bash
cd MeMOTR/models/ops
sh make.sh
python test.py   # optional: verify the op
```

---

## 5. Download pretrained checkpoints

| Checkpoint | URL | Destination |
|---|---|---|
| `memotr_dancetrack.pth` | [Google Drive](https://drive.google.com/file/d/1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2) | `checkpoints/pretrained/` |
| `memotr_mot17.pth` | [Google Drive](https://drive.google.com/file/d/1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew) | `checkpoints/pretrained/` |

Checkpoints are excluded from git (`.gitignore` rule: `*.pth`).

---

## 6. Data

Dataset images are excluded from git. Ground-truth annotation files
(`gt/gt.txt`, `seqinfo.ini`) **are** tracked for the sequences included in
this repo:

- `data/DanceTrack/val/dancetrack0004/`
- `data/DanceTrack/val/dancetrack0005/`
- `data/MOT17/train/MOT17-02-FRCNN/`

Raw IOTD video clips go in `data/IOTD/raw_unannotated/` (gitignored).

---

## 7. Sanity check

Run the quick end-to-end smoke test to confirm everything is wired up:

```bash
python smoke_test.py
```

Expected: model loads without `missing_keys`, Deformable Attention forward
pass executes, no import errors.
