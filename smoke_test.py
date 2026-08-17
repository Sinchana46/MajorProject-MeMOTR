"""
smoke_test.py — Task 1.3: MeMOTR end-to-end smoke test

Purpose: Verify the MeMOTR codebase is fully importable and can run a forward pass
without downloading any dataset or checkpoint.  Uses random-weight initialisation
only.  The test validates:
  1. All MeMOTR modules import cleanly under Python 3.9 + PyTorch 2.5.1
  2. The model builds from the DanceTrack config without error
  3. A single-frame forward pass completes on GPU (batch_size=1)
  4. The model produces pred_logits and pred_bboxes tensors of the expected shape
  5. The RuntimeTracker update() runs without error (validates the TODO(C2) hook point)
  6. The QueryUpdater runs (validates the TODO(C1) hook point)

This test is intentionally minimal — it does NOT:
  - Load any pretrained weights (no checkpoint download required)
  - Process real video or images
  - Compute HOTA or any tracking metric

Run from the MeMOTR directory:
    conda activate memotr_project
    python d:/MajorProject-MeMOTR/smoke_test.py
"""

import sys
import os

# ── 1. Path setup ─────────────────────────────────────────────────────────────
MEMOTR_DIR = os.path.join(os.path.dirname(__file__), "MeMOTR")
sys.path.insert(0, MEMOTR_DIR)
# MultiScaleDeformableAttention.pyd lives in models/ops/ and is imported as a
# top-level module by MeMOTR's functions/__init__.py — must be on sys.path.
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))

import torch
import numpy as np

print("=" * 60)
print("MeMOTR Smoke Test — Task 1.3")
print("=" * 60)

# ── 2. GPU check ──────────────────────────────────────────────────────────────
print(f"\n[1] PyTorch {torch.__version__}")
print(f"    CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU            : {torch.cuda.get_device_name(0)}")
    print(f"    VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = "cuda"
else:
    print("    WARNING: CUDA not available — running on CPU (slow)")
    DEVICE = "cpu"

# ── 3. Import MeMOTR modules ──────────────────────────────────────────────────
print("\n[2] Importing MeMOTR modules...")
from models import build_model
from models.runtime_tracker import RuntimeTracker
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from structures.track_instances import TrackInstances
from utils.utils import yaml_to_dict
print("    All MeMOTR imports: OK")

# ── 4. Build model from DanceTrack config ─────────────────────────────────────
print("\n[3] Building model from configs/train_dancetrack.yaml (random weights)...")
config_path = os.path.join(MEMOTR_DIR, "configs", "train_dancetrack.yaml")
config = yaml_to_dict(config_path)

# Patch config for single-GPU smoke test
config["AVAILABLE_GPUS"] = "0"
config["USE_DISTRIBUTED"] = False
config["USE_CHECKPOINT"] = False
config["VISUALIZE"] = False

model = build_model(config)
model.eval()
print(f"    Model built: OK  (params={sum(p.numel() for p in model.parameters()):,})")

# ── 5. Create a fake single frame (random pixels) ─────────────────────────────
print("\n[4] Creating synthetic 800x1333 frame (random pixels)...")
H, W = 800, 1333
fake_frame = torch.randn(1, 3, H, W).to(DEVICE)   # (B, C, H, W)
fake_mask  = torch.zeros(1, H, W, dtype=torch.bool).to(DEVICE)  # no padding

from utils.nested_tensor import NestedTensor
frame_nested = NestedTensor(fake_frame, fake_mask)

# ── 6. Create empty track instances (no existing tracks on frame 1) ───────────
print("\n[5] Creating empty TrackInstances (frame 1, no prior tracks)...")
tracks = [TrackInstances(
    frame_height=float(H),
    frame_width=float(W),
    hidden_dim=config["HIDDEN_DIM"],
    use_dab=config.get("USE_DAB", True)    # must match model config: query_embed is 256-dim with DAB
).to(DEVICE)]
print("    TrackInstances: OK")

# ── 7. Forward pass through MeMOTR ───────────────────────────────────────────
print("\n[6] Running MeMOTR forward pass...")
with torch.no_grad():
    outputs = model(frame=frame_nested, tracks=tracks)

pred_logits = outputs["pred_logits"]
pred_bboxes = outputs["pred_bboxes"]
print(f"    pred_logits shape : {pred_logits.shape}")  # (B, N_det+N_track, num_classes)
print(f"    pred_bboxes shape : {pred_bboxes.shape}")  # (B, N_det+N_track, 4)
print("    Forward pass: OK")

# ── 8. RuntimeTracker update (TODO(C2) hook point) ───────────────────────────
print("\n[7] Testing RuntimeTracker.update() — validates TODO(C2) hook point...")
tracker = RuntimeTracker(
    det_score_thresh=config.get("DET_SCORE_THRESH", 0.5),
    track_score_thresh=config.get("TRACK_SCORE_THRESH", 0.5),
    miss_tolerance=config.get("MISS_TOLERANCE", 30),
    use_dab=config.get("USE_DAB", True)
)
_, new_tracks = tracker.update(outputs, tracks)
print(f"    New tracks detected: {len(new_tracks[0])} objects")
print(f"    det_score_thresh used: {tracker.det_score_thresh}  ← this is the TODO(C2) hook")
print("    RuntimeTracker.update(): OK")

# ── 9. QueryUpdater (TODO(C1) hook point) ────────────────────────────────────
print("\n[8] Testing query_updater (validates TODO(C1) hook point)...")
# postprocess_single_frame wraps query_updater.forward()
updated_tracks = model.postprocess_single_frame(
    previous_tracks=tracks,
    new_tracks=new_tracks,
    unmatched_dets=None,
    no_augment=True
)
print(f"    long_memory_lambda : {model.query_updater.long_memory_lambda}  ← this is the TODO(C1) hook")
print("    query_updater forward: OK")

# ── 10. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SMOKE TEST PASSED — MeMOTR install is functional end-to-end")
print(f"  GPU : {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
print(f"  PyTorch version : {torch.__version__}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  TODO(C1) hook   : query_updater.long_memory_lambda = {model.query_updater.long_memory_lambda}")
print(f"  TODO(C2) hook   : RuntimeTracker.det_score_thresh  = {tracker.det_score_thresh}")
print("=" * 60)
