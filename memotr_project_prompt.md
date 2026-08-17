# MeMOTR Indian Traffic Surveillance — Research Project Context Prompt

> **Purpose:** Give this file as context to any AI assistant (Claude, ChatGPT, Copilot, etc.) at the start of every session. It contains the full project background, finalized decisions, and execution plan so you never repeat yourself.

---

## 1. Project Identity

| Field | Detail |
|---|---|
| Project type | B.E. Final Year Research Project targeting journal publication |
| Target venues | IEEE T-ITS / Pattern Recognition / Engineering Applications of AI / Journal of Visual Communication and Image Representation |
| Paper title | "Perspective-Aware and Scene-Adaptive Transformer-Based Multi-Object Tracking for Oblique Fixed-Camera Indian Traffic Surveillance" |
| Team hardware | NVIDIA GPU 24GB VRAM, 16GB RAM |
| AI tools in use | Claude (code, analysis), other AI tools for assistance |

---

## 2. Core Architecture

```
PRE-TRAINED DAB-Deformable-DETR (COCO weights)
              ↓
    MeMOTR (MCG-NJU/MeMOTR on GitHub)
    initialized from DAB-Deformable-DETR weights
              ↓
    Fine-tuned on combined dataset
    (UA-DETRAC + VisDrone-Vehicles + IOTD)
              ↓
    + Contribution 1: Perspective-Zone Memory Decay
    + Contribution 2: Scene-Conditioned Threshold Adapter
              ↓
    Evaluated on all three datasets separately
```

**Key MeMOTR facts you must know:**
- Built on DAB-Deformable-DETR with ResNet-50 backbone
- Uses fixed memory decay rate λ = 0.01 for ALL tracked objects
- Uses fixed detection confidence threshold τ_det = 0.5
- Memory update equation: `M̃(t+1) = (1 - λ) × M(t) + λ × O(t)`
- The Temporal Interaction Module (TIM) is where λ is applied
- Main file to modify: `models/memorizing_motr.py`
- GitHub: https://github.com/MCG-NJU/MeMOTR

---

## 3. Datasets

### 3.1 UA-DETRAC (Public)
- 100 video sequences, 140,000+ frames, fixed overhead/side cameras
- Chinese traffic, 4 classes: car, bus, van, other
- Annotation format: XML per sequence
- Download: https://detrac-db.rit.albany.edu/
- Train/test split: 60 train / 40 test (official)
- Camera: Fixed, overhead or low-angle side view, 960×540 @ 25 FPS

### 3.2 VisDrone-MOT (Public, filtered to vehicles)
- Drone aerial footage, diverse urban scenes
- Keep only classes: car (4), van (5), truck (6), bus (9)
- Download: https://github.com/VisDrone/VisDrone-Dataset
- MUST download MOT version not DET version
- Split: official train/val splits

### 3.3 IOTD — Indian Oblique Traffic Dataset (Custom, Novel)
- **Camera setup:** Smartphone on tripod/mount on bridge or flyover
- **Camera height:** 5–15 meters above road level
- **Camera angle:** 30–60 degrees from horizontal (oblique, NOT top-down)
- **Resolution:** 1080p minimum, 4K preferred
- **Frame rate:** 25 FPS (matches UA-DETRAC standard)
- **Clip duration:** 60–90 seconds raw, 500–600 usable frames per sequence
- **Vehicle classes:** car, bus, truck, auto_rickshaw (novel), two_wheeler
- **Lighting conditions:** daytime (7 seq), dusk/evening (7 seq), night (6 seq)
- **Total target:** 20 sequences, minimum 10,000 annotated frames
- **Annotation tool:** CVAT (https://github.com/opencv/cvat)
- **Annotation format:** MOT Challenge format (txt files)
- **Occlusion attributes per box:** none / partial / heavy
- **Split:** 12 train / 4 val / 4 test (by sequence, NEVER by frame)

**What makes IOTD novel (three independent axes):**
1. Camera geometry: oblique fixed-camera with strong perspective gradient — no existing dataset has this
2. Vehicle classes: auto-rickshaw is absent from ALL standard benchmarks
3. Traffic behavior: non-lane Indian traffic with chaotic occlusion patterns

---

## 4. Finalized Contributions

### Contribution 1: Perspective-Zone-Conditioned Memory Decay Module

**Problem it solves:**
MeMOTR uses λ = 0.01 for every tracked object regardless of position in frame. In an oblique camera view, vehicles at the top of frame are far away (small pixels, weak features, frequent occlusion) while vehicles at bottom are close (large, well-lit, reliable features). They should NOT have the same memory retention strategy.

**How it works:**
1. One-time calibration: divide image vertically into 3 perspective zones using bounding box height vs vertical position regression from annotated data
2. Zone boundaries stored as fixed constants (not learned)
3. PositionMLP (tiny 3-layer network, ~1000 parameters) maps zone + normalized box height to λ_adaptive
4. Memory update becomes: `M̃(t+1) = (1 - λ_adaptive) × M(t) + λ_adaptive × O(t)`

```
Top 30% of frame    → FAR zone   → λ low  (rely on memory, features weak)
Middle 40% of frame → MID zone   → λ mid
Bottom 30% of frame → NEAR zone  → λ high (trust current obs, features strong)
```

**Why novel vs STFN-ADM-MTHS (closest competitor, J. Supercomputing 2025):**
- STFN-ADM-MTHS conditions decay on motion stability (trajectory state)
- Ours conditions decay on geometric position in the perspective field
- Completely orthogonal conditioning signals
- STFN-ADM-MTHS is for highway gantry cameras with vibration; ours is for oblique fixed cameras with perspective gradient

**Files to modify:**
- `models/memorizing_motr.py` → TemporalInteractionModule class
- Add new file: `modules/position_decay.py`
- Add new file: `tools/calibrate_perspective.py`

**Training strategy:**
- Load fine-tuned MeMOTR checkpoint
- Freeze backbone and encoder
- Train TIM + PositionMLP on combined dataset, 15–20 epochs
- LR: 2e-4, decay at epoch 12

---

### Contribution 2: Scene-Conditioned Adaptive Detection Confidence Module

**Problem it solves:**
MeMOTR uses fixed τ_det = 0.5 during inference (tuned on DanceTrack/MOT17 pedestrians in controlled lighting). In oblique bridge-camera footage at dusk/night, genuine vehicles score 0.35–0.45 and get discarded. No existing transformer MOT paper adapts this threshold based on scene conditions.

**How it works:**
1. SceneAnalyzer: computes 3 scalars from current frame — brightness (mean pixel), contrast (std pixel), density (active track count normalized)
2. ThresholdAdapter: 2-layer MLP (~500 parameters), maps scene vector to τ_adaptive ∈ [0.3, 0.7]
3. Replaces fixed `if score > 0.5` with `if score > τ_adaptive` in MeMOTR inference

**Why novel:**
- No reviewed paper adapts detection threshold by scene state in transformer MOT
- ByteTrack uses low-confidence detections post-detection (different paradigm)
- Ours operates pre-decision, not post-detection

**Files to modify:**
- `models/memorizing_motr.py` → inference forward pass
- Add new file: `modules/scene_analyzer.py`
- Add new file: `modules/threshold_adapter.py`

**Training strategy:**
- Freeze ALL MeMOTR weights
- Train only ThresholdAdapter on IOTD training sequences
- Use sequences across all 3 lighting conditions
- LR: 1e-3, 5 epochs, fast convergence

---

### Contribution 3: IOTD Dataset

**Minimum requirements for journal credibility:**

| Statistic | Minimum | Recommended |
|---|---|---|
| Total sequences | 20 | 25–30 |
| Frames per sequence | 500 | 600 |
| Total annotated frames | 10,000 | 15,000+ |
| Unique vehicle track IDs | 500 | 800+ |
| Vehicle classes | 5 | 5 |
| Lighting conditions | 3 | 3 |
| Sequences per lighting | 5 minimum | 7 |
| Occlusion attribute | Required | Required |
| Perspective gradient R² | Report | > 0.5 expected |

**Annotation time estimate:**
```
500 frames ÷ 10 (CVAT keyframe interval) = 50 keyframes per sequence
50 keyframes × 3 minutes = 2.5 hours per sequence
20 sequences × 2.5 hours = 50 total hours
With 3 annotators = ~17 hours each = 3–4 weeks part-time
```

---

## 5. Baseline Comparisons Required

Your results tables must include these methods:

| Method | Type | Why included |
|---|---|---|
| MeMOTR (pre-trained, no fine-tune) | Backbone | Shows domain gap |
| MeMOTR (fine-tuned on combined data) | Strong baseline | Shows fine-tuning gain |
| STFN-ADM-MTHS | Direct competitor | Closest architecture |
| ByteTrack | Tracking-by-detection | Standard baseline |
| OC-SORT | Tracking-by-detection | Standard baseline |
| Ours (Contribution 2 only) | Ablation | Shows C2 value |
| Ours (Contribution 1 only) | Ablation | Shows C1 value |
| Ours (C1 + C2, full model) | Proposed | Final result |

---

## 6. Evaluation Metrics

**Primary metrics (report all):**
- HOTA ↑ (Higher Order Tracking Accuracy — primary metric)
- DetA ↑ (Detection Accuracy component)
- AssA ↑ (Association Accuracy component — most sensitive to memory changes)
- MOTA ↑ (Multiple Object Tracking Accuracy)
- IDF1 ↑ (ID F1 Score)
- ID Switches ↓

**Per-condition metrics (for IOTD, novel evaluation):**
- HOTA / IDF1 / ID Switches broken down by: daytime / dusk / night
- DetA broken down by vehicle class (shows auto-rickshaw gap)
- AssA broken down by perspective zone (far / mid / near)

**Evaluation toolkit:** TrackEval (https://github.com/JonathonLuiten/TrackEval)

---

## 7. What Has Been Confirmed as NOT Novel

Do NOT claim novelty for these — reviewers will reject immediately:

- Using DAB-Deformable-DETR as backbone (MeMOTR already does this)
- Combining multiple datasets for training (standard practice)
- Collecting a dataset without characterizing the geometry gap (insufficient alone)
- Adding CLAHE or manual preprocessing rules (heuristic engineering, not research)
- Adapting MeMOTR to vehicles in general (STFN-ADM-MTHS does this)
- Trajectory-motion-conditioned memory decay (STFN-ADM-MTHS does this)
- Camera vibration compensation (STFN-ADM-MTHS SCFA module does this)

---

## 8. Key Papers to Know and Cite

| Paper | Year | Venue | Role in your paper |
|---|---|---|---|
| MeMOTR | 2023 | ICCV | Your backbone — cite extensively |
| DAB-Deformable-DETR | 2022 | ICLR | Pre-training source |
| MOTR | 2022 | ECCV | MeMOTR predecessor |
| MOTRv2 | 2023 | CVPR | Hybrid architecture comparison |
| STFN-ADM-MTHS | 2025 | J. Supercomputing | Closest competitor — must beat on AssA |
| RT-DETR + DeepSORT | 2025 | HSI | Tracking-by-detection baseline |
| UA-DETRAC | 2020 | CVIU | Dataset you use |
| ByteTrack | 2022 | ECCV | Baseline tracker |
| OC-SORT | 2023 | CVPR | Baseline tracker |
| TrackEval / HOTA | 2021 | IJCV | Your evaluation metric |

---

## 9. Execution Phases Summary

```
Phase 0: Foundation Setup          → Weeks 1–2
Phase 1: Dataset Creation (IOTD)   → Weeks 3–8 (runs in parallel throughout)
Phase 2: Data Preparation Pipeline → Weeks 3–4
Phase 3: Baseline Establishment    → Weeks 5–6
Phase 4: Module Implementation     → Weeks 7–10
Phase 5: Ablation Experiments      → Weeks 11–12
Phase 6: Full Evaluation           → Weeks 13–14
Phase 7: Paper Writing             → Weeks 15–18
Phase 8: Internal Review & Submit  → Weeks 19–20
```

---

## 10. Critical Implementation Files

### Files to MODIFY in MeMOTR:

```
models/memorizing_motr.py
  → TemporalInteractionModule: replace fixed λ with PositionMLP output
  → Inference forward: replace fixed τ_det with ThresholdAdapter output
  → __init__: add new modules to model

models/dab_deformable_detr/
  → No changes needed here
```

### New files to CREATE:

```
modules/
  scene_analyzer.py         → SceneAnalyzer class (no learned params)
  threshold_adapter.py      → ThresholdAdapter class (2-layer MLP, ~500 params)
  position_decay.py         → PerspectiveZoneEncoder + PositionMLP (~1000 params)

tools/
  calibrate_perspective.py  → One-time script to measure zone boundaries from data
  convert_detrac.py         → UA-DETRAC XML → MOT format
  convert_visdrone.py       → VisDrone → MOT format, filter vehicle classes
  visualize_zones.py        → Draw zone boundaries on sample frames (for paper figure)
```

### Training scripts:

```
train_c2_only.py   → Train ThresholdAdapter with MeMOTR frozen
train_full.py      → Fine-tune TIM + PositionMLP + ThresholdAdapter
eval_per_condition.py → HOTA/IDF1 broken down by lighting condition
eval_per_zone.py      → AssA broken down by perspective zone
```

---

## 11. Data Directory Structure

```
project_root/
├── data/
│   ├── UA_DETRAC/
│   │   ├── train/
│   │   │   └── MVI_XXXXX/
│   │   │       ├── img1/
│   │   │       │   ├── 000001.jpg
│   │   │       ├── gt/
│   │   │       │   └── gt.txt
│   │   │       └── seqinfo.ini
│   │   └── test/
│   ├── VisDrone_Vehicles/
│   │   ├── train/
│   │   └── val/
│   └── IOTD/
│       ├── train/    (12 sequences)
│       ├── val/      (4 sequences)
│       └── test/     (4 sequences — DO NOT TOUCH until final eval)
├── checkpoints/
│   ├── pretrained/
│   │   └── memotr_dab_deformable_detr.pth
│   ├── memotr_finetuned/
│   └── memotr_full_proposed/
├── results/
│   ├── baseline/
│   ├── ablation/
│   └── final/
├── modules/
├── tools/
└── MeMOTR/          (cloned from GitHub)
```

---

## 12. Ablation Study Table (Target Structure)

| Configuration | HOTA | DetA | AssA | MOTA | IDF1 | ID-SW |
|---|---|---|---|---|---|---|
| MeMOTR pretrained | - | - | - | - | - | - |
| MeMOTR fine-tuned (Baseline B) | - | - | - | - | - | - |
| Baseline B + C2 (threshold adapt) | - | - | - | - | - | - |
| Baseline B + C1 (perspective decay) | - | - | - | - | - | - |
| Baseline B + C1 + C2 (Full proposed) | - | - | - | - | - | - |

*Fill values after experiments. AssA improvement from C1 is the key number to highlight.*

---

## 13. Per-Lighting Condition Table (Target Structure for IOTD)

| Method | Daytime HOTA | Dusk HOTA | Night HOTA | Avg ID-SW |
|---|---|---|---|---|
| MeMOTR fine-tuned | - | - | - | - |
| + C2 (threshold adapt) | - | - | - | - |
| Full proposed | - | - | - | - |

*Night improvement from C2 is the key number to highlight here.*

---

## 14. Per-Zone AssA Table (Target Structure for IOTD)

| Method | FAR zone AssA | MID zone AssA | NEAR zone AssA |
|---|---|---|---|
| MeMOTR fine-tuned | - | - | - |
| + C1 (perspective decay) | - | - | - |
| Full proposed | - | - | - |

*FAR zone improvement from C1 is the key number to highlight here.*

---

## 15. Perspective Calibration Output (Fill After Running Script)

```
Camera location: [YOUR LOCATION NAME]
Image resolution: [WIDTH] × [HEIGHT]
Perspective gradient slope: [FILL AFTER CALIBRATION]
R² value: [FILL — expected > 0.5]
Zone boundaries:
  FAR/MID boundary (normalized y): [FILL]
  MID/NEAR boundary (normalized y): [FILL]
  FAR zone: y < [FILL] (top [X]% of frame)
  MID zone: [FILL] ≤ y < [FILL]
  NEAR zone: y ≥ [FILL] (bottom [X]% of frame)
```

---

## 16. Environment Setup Commands

```bash
# Create environment
conda create -n memotr_project python=3.9
conda activate memotr_project

# PyTorch (adjust cu118 to match your CUDA version)
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Clone MeMOTR
git clone https://github.com/MCG-NJU/MeMOTR
cd MeMOTR
pip install -r requirements.txt

# Evaluation toolkit
git clone https://github.com/JonathonLuiten/TrackEval
pip install -r TrackEval/requirements.txt

# Experiment tracking
pip install wandb

# Annotation tool (local CVAT)
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d
# Access at http://localhost:8080

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 17. Novelty One-Liner (Use This to Explain to Guide)

> "We propose a homography-guided perspective-aware memory decay module and a scene-conditioned detection threshold adapter for transformer-based multi-object tracking, evaluated on IOTD, a novel Indian oblique fixed-camera traffic dataset that introduces the auto-rickshaw vehicle class absent from all existing MOT benchmarks. Unlike STFN-ADM-MTHS which conditions memory decay on trajectory motion state for vibrating highway cameras, our decay is conditioned on geometric perspective position, addressing an orthogonal and previously unaddressed failure mode in fixed oblique-camera deployments."

---

## 18. Immediate Next Actions (In Order)

- [ ] Set up conda environment and verify GPU
- [ ] Clone MeMOTR and run inference on a sample UA-DETRAC sequence
- [ ] Download UA-DETRAC and VisDrone-MOT datasets
- [ ] Record first 5 IOTD sequences (daytime: 2, dusk: 2, night: 1)
- [ ] Install CVAT and start annotating first 3 sequences
- [ ] Read MeMOTR paper: first pass (overview), second pass (draw architecture), third pass (match paper to code)
- [ ] In `models/memorizing_motr.py`, find and comment the line that sets λ = 0.01
- [ ] In the same file, find where `tau_det` threshold is applied during inference
- [ ] Run `tools/calibrate_perspective.py` once first 3 sequences are annotated
- [ ] Write one-page architecture summary in your own words (for guide presentation)

---

## 19. Things NOT to Do

- Do NOT split dataset by frame (only by sequence)
- Do NOT touch IOTD test sequences until final evaluation
- Do NOT modify MeMOTR backbone or encoder (too risky, not your contribution)
- Do NOT claim vibration compensation as novelty (STFN-ADM-MTHS does this)
- Do NOT report only MOTA (it overweights detection, obscures memory contribution)
- Do NOT implement all contributions simultaneously (implement C2 first, verify it works, then add C1)
- Do NOT generate code you cannot explain line by line to your guide

---

## 20. Session Instruction for AI Assistant

When starting a new session with this context file, tell the AI:

> "This is my research project context. I am working on [specific task] today. The relevant section from this document is Section [X]. Help me with [specific ask]."

For code generation tasks, always specify:
- Which file to modify or create
- Which class or function to work within
- What the input and output should be
- Which section of this document the code relates to

This ensures the AI generates code that fits your existing architecture rather than generic implementations.

---

*Last updated: Based on full conversation history covering project scoping, literature review, novelty validation, contribution finalization, and execution planning.*
*Hardware confirmed: 24GB VRAM GPU — sufficient for MeMOTR fine-tuning with gradient checkpointing and mixed precision.*
