# MeMOTR for Indian Traffic Surveillance

Transformer-based multi-object tracking of vehicles in **oblique, fixed-camera Indian traffic video**.

This project adapts [MeMOTR](https://github.com/MCG-NJU/MeMOTR) (a long-term-memory transformer tracker built on
DAB-Deformable-DETR) to vehicle tracking. It fine-tunes the tracker on a custom dataset, **IOTD (Indian Oblique
Traffic Dataset)**, recorded from bridges and flyovers. It also adds two optional extensions:

| | Contribution | Module | Status |
|---|---|---|---|
| **C1** | Perspective-zone-conditioned memory decay: replaces MeMOTR's fixed memory decay λ with a per-track value based on where the vehicle is in the frame (far vs. near) | `modules/position_decay.py` | Implemented and unit-tested; **not yet trained or evaluated** (off by default) |
| **C2** | Scene-conditioned detection threshold: replaces the fixed τ_det = 0.5 with a per-frame value in [0.3, 0.7] computed from brightness, contrast and density | `modules/scene_analyzer.py`, `modules/threshold_adapter.py` | Implemented and evaluated; it **lowered HOTA** on the night sequences, so it is off |

The repo also includes a **Streamlit web app**. You upload a video and get back the tracked, annotated video, a gallery
of keyframes and a CSV of tracks.

---

## Results

All results below are on IOTD validation sequence `seq_02_30sec_a` (744 frames, 50 tracks, night, 1920×1080). They
were scored with TrackEval (HOTA, CLEAR and Identity metrics) using the upstream MeMOTR output filter (score > 0.5,
box area > 100 px).

| Model | Training data | Best ckpt | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|---|---|
| Phase 5 baseline | UA-DETRAC + VisDrone (vehicles) | 4 | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | 0 |
| **Arm 1** | Phase 5 + real IOTD frames | 3 | **64.12** | **60.54** | **68.21** | **68.15** | **84.64** | 3 |
| Arm 2 | Phase 5 + real + augmented IOTD | 1 | 61.61 | 58.10 | 65.51 | 67.43 | 82.83 | 3 |

- **Fine-tuning on IOTD helps:** +6.38 HOTA over the baseline, and recall rises from 73.8 % to 88.3 %.
- **Results across seeds:** with seeds 42 / 1234 / 2024, Arm 1 checkpoint 3 scores 64.12 / 61.07 / 59.74
  (61.64 ± 2.25), so every seed beats the baseline.
- **Photometric augmentation (Arm 2) did not help.**
- **Learning-rate variants made no clear difference:** halved learning rates or a frozen query updater gave no
  improvement beyond the seed-to-seed variation.
- **Selected setup:** Arm 1 `checkpoint_3`, detection 0.5, tracking 0.5, miss tolerance 30, C1 and C2 off. See
  `results_after_tuning/final_review/selected_settings.csv`.

**Limitations:** only one validation sequence, which is at night, and only 3 seeds. No statistical significance is
claimed. For details, see `results_after_tuning/final_results_report.md` and
`results_after_tuning/final_review/final_review_report.md`.

---

## Repository layout

```
MeMOTR/                  git submodule – upstream MeMOTR (patched, see patches/)
TrackEval/               git submodule – upstream TrackEval (patched, see patches/)
patches/                 all changes made to the two submodules (apply after cloning)
modules/                 C1 and C2 modules (position_decay, scene_analyzer, threshold_adapter)
configs/                 training / inference configs (Phase 5, Arm 1, Arm 2, seed and LR variants)
tools/                   data conversion, inference, evaluation, tuning and report scripts
tests/                   unit tests for modules/ and a zero-regression check
data/                    ground-truth annotations only (frames are not in git)
  IOTD/                  IOTD sequences + augmented_train_v2 (accepted augmentation)
  MOT17/, DanceTrack/    small sanity-check slices
results/                 tuning close-out: figures, tables, narrative
results_prior_to_tuning/ baseline results before tuning
results_after_tuning/    corrected sweep, seed check, final review, presentation figures
archive/                 frozen copy of the pre-tuning baseline
check/                   annotation audit and diagnostic renders
reports/                 project report (HTML / PDF)
app.py                   Streamlit web app
run_app.py               launches app.py (with a Windows SSL fix)
run_dashboard.py         launches app.py (alternative SSL fix using certifi)
smoke_test.py            end-to-end import + forward-pass check
build_cuda_ops.bat       builds the Deformable Attention CUDA op on Windows
REPRODUCE.md             detailed environment and patch notes
```

---

## Setup

You need an NVIDIA GPU with CUDA and Python 3.9. The project was developed on Windows with VS 2022 and the CUDA 13.0
toolkit. Linux works too: use `make.sh` to build the CUDA op instead of the `.bat` file.

**1. Clone with submodules**

```bash
git clone --recurse-submodules https://github.com/Sinchana46/MajorProject-MeMOTR.git
cd MajorProject-MeMOTR
```

**2. Apply the patches** (in this order; `REPRODUCE.md` explains each one)

```bash
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

**3. Create the environment and install dependencies**

```bash
conda create -n memotr_project python=3.9
conda activate memotr_project
# Install the torch + torchvision build that matches your CUDA version from pytorch.org first, then:
pip install -r requirements.txt
```

**4. Build the Deformable Attention CUDA op**

```bat
:: Windows
build_cuda_ops.bat
```
```bash
# Linux
cd MeMOTR/models/ops && sh make.sh && cd ../../..
```

**5. Download the pretrained MeMOTR checkpoint** into `checkpoints/pretrained/`:
- [`memotr_dancetrack.pth`](https://drive.google.com/file/d/1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2)
- [`memotr_mot17.pth`](https://drive.google.com/file/d/1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew) (optional)

The fine-tuned checkpoints (`results/phase5_interim_baseline/`, `results/train_iotd_real/` etc.) are **not** in git
(`*.pth` is ignored). Get them from the project team, or produce them by training (see below).

**6. Check the install**

```bash
python smoke_test.py
python tests/test_modules.py
```

---

## Data

Only the annotations (`gt/gt.txt`, `gt/labels.txt`, `seqinfo.ini`) are in git. To run training or inference, put the
extracted frames in each sequence's `img1/` folder:

```
data/IOTD/seq_01_1min/       train  (1,599 frames)
data/IOTD/seq_03_30sec_b/    train  (931 frames)
data/IOTD/seq_02_30sec_a/    validation  (744 frames, night)
data/IOTD/augmented_train_v2/seq_*_aug_0{1,2,3}/   photometric augmentations of the two training sequences
```

The configs read IOTD through three folders that **link to** the folders above, so the frames aren't stored twice.
Create these links once, after the frames are in place:

```bash
python tools/create_junctions_and_verify.py   # Windows: creates data/IOTD_train_real, IOTD_train_aug, IOTD_val
```

On Linux, create the same layout with `ln -s`. For example:
`mkdir -p data/IOTD_val && ln -s ../IOTD/seq_02_30sec_a data/IOTD_val/seq_02_30sec_a`.

The Phase 5 baseline also needs UA-DETRAC and VisDrone-MOT. Use `tools/convert_detrac.py` and
`tools/convert_visdrone.py` to convert them into `data/UA_DETRAC/` and `data/VisDrone_Vehicles/`.

> **Paths in configs:** every file in `configs/` uses absolute paths under `D:/MajorProject-MeMOTR/`
> (`DATA_ROOT`, `OUTPUTS_DIR`, `PRETRAINED_MODEL`). If your clone is somewhere else, change them first.

---

## Usage

**Web app**: upload an `.mp4` and get back the annotated video, keyframes and a CSV of tracks. It needs
`results/train_iotd_real/checkpoint_3.pth`.

```bash
python run_app.py            # or: streamlit run app.py
# open http://localhost:8501
```

**Track a new video from the command line**

```bash
python -u tools/test_new_video.py --video path/to/video.mp4
# output: results/manual_test/<name>/tracker/<name>.txt (MOT format)
```

**Training** (example: Arm 1). Run from the project root, with `MeMOTR` and `MeMOTR/models/ops` on `PYTHONPATH`:

```bash
python MeMOTR/main.py --config-path configs/train_iotd_real.yaml
```

**Inference + evaluation on the validation sequence**

```bash
# 1. (once, from the project root) build the evaluation-only GT: all vehicle classes mapped to 1
python tools/eval_build_gt.py          # writes results/eval_iotd_val/gt_eval/

# 2. inference — from MeMOTR/, with PYTHONPATH=<repo>/MeMOTR/models/ops;<repo>
python ../tools/run_inference_mot_filtered.py \
    --checkpoint ../results/train_iotd_real/checkpoint_3.pth \
    --split-dirs ../data/IOTD_val \
    --output-dir ../results/my_eval/tracker \
    --config     ../configs/train_iotd_real.yaml

# 3. scoring — from the project root
python tools/run_trackeval.py \
    --tracker-dir results/my_eval/tracker \
    --gt-dirs     results/eval_iotd_val/gt_eval \
    --output-dir  results/my_eval \
    --label       MY_EVAL
```

Use `run_inference_mot_filtered.py`, not the older `run_inference_mot.py`. The older script writes boxes that
upstream MeMOTR filters out, which lowers HOTA by 7–17 points (see `results_after_tuning/final_results_report.md`).

---

## Acknowledgements

- [MeMOTR](https://github.com/MCG-NJU/MeMOTR), Gao & Wang, ICCV 2023: the base tracker.
- [TrackEval](https://github.com/JonathonLuiten/TrackEval): HOTA / CLEAR / Identity evaluation.
- [UA-DETRAC](https://detrac-db.rit.albany.edu/) and [VisDrone](https://github.com/VisDrone/VisDrone-Dataset): public vehicle datasets.
