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

### Patch 002 — MeMOTR modifications (C1/C2 — to be added)

> **Placeholder.** Once the Indian On-road Traffic Detection (IOTD) domain
> adaptation changes (C1: perspective-aware attention, C2: multi-class head)
> are implemented, their diffs will be placed here as
> `patches/002-memotr-c1-perspective.diff` and
> `patches/003-memotr-c2-multiclass.diff`.

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
