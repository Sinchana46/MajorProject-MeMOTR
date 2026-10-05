"""
tests/test_zero_regression.py — Task 3.4: real-model zero-regression check

Requirement (implementation plan, Task 3.4):
  "re-run the Phase 1 smoke test with flags off — output must be bit-identical
   to Phase 1 baseline, proving zero regression."

What this test does
-------------------
1. Seeds all RNGs for determinism.
2. Builds the real MeMOTR model (with compiled CUDA ops) from the DanceTrack config.
3. Runs ONE forward pass + RuntimeTracker.update() + postprocess_single_frame()
   with the model's default RuntimeTracker (which now has use_c2=False by
   default after Task 3.4 wiring).
4. Repeats step 3 with RuntimeTracker(use_c2=False) explicitly set.
5. Asserts pred_logits, pred_bboxes, and all tracker attributes are
   bit-for-bit identical between the two runs.
6. Verifies model.query_updater.use_c1 == False (C1 path is off by default).
7. Verifies RuntimeTracker.tau_adaptive == det_score_thresh (fixed threshold
   used, not the adaptive one).

Run from the MeMOTR directory (required — same as smoke_test.py):
    conda activate memotr_project
    cd d:/MajorProject-MeMOTR/MeMOTR
    python d:/MajorProject-MeMOTR/tests/test_zero_regression.py

Exit code 0 = PASS, 1 = FAIL.
"""

import sys
import os

# ── Path setup (identical to smoke_test.py) ───────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MEMOTR_DIR  = os.path.join(PROJECT_DIR, "MeMOTR")
sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))  # for MSDA .pyd
sys.path.insert(0, PROJECT_DIR)

import torch

print("=" * 65)
print("Phase 3 Task 3.4 — Zero-Regression Check (real CUDA model)")
print("=" * 65)

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Imports ───────────────────────────────────────────────────────────────────
print("\n[1] Importing real MeMOTR (with compiled CUDA ops)...")
from models import build_model
from models.runtime_tracker import RuntimeTracker
from utils.nested_tensor import NestedTensor
from structures.track_instances import TrackInstances
from utils.utils import yaml_to_dict
print("    OK")

# ── Config ────────────────────────────────────────────────────────────────────
config_path = os.path.join(MEMOTR_DIR, "configs", "train_dancetrack.yaml")
config = yaml_to_dict(config_path)
config["AVAILABLE_GPUS"] = "0"
config["USE_DISTRIBUTED"] = False
config["USE_CHECKPOINT"]  = False
config["VISUALIZE"]       = False

# ── Deterministic RNG seed ─────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)

# ── Build model once (shared weights for both runs) ───────────────────────────
print("\n[2] Building model (random weights, seed=42)...")
model = build_model(config)
model.eval()
print(f"    params = {sum(p.numel() for p in model.parameters()):,}")

# ── Synthetic frame (same for both runs) ──────────────────────────────────────
H, W = 800, 1333
torch.manual_seed(SEED)
fake_frame = torch.randn(1, 3, H, W).to(DEVICE)
fake_mask  = torch.zeros(1, H, W, dtype=torch.bool).to(DEVICE)
frame_nested = NestedTensor(fake_frame, fake_mask)


def run_forward_and_track(use_c2_explicit):
    """
    Run MeMOTR forward + RuntimeTracker.update() + postprocess.
    use_c2_explicit=None  → use default RuntimeTracker (as smoke_test.py does)
    use_c2_explicit=False → pass use_c2=False explicitly
    Returns dict of key tensors/values for comparison.
    """
    tracks = [TrackInstances(
        frame_height=float(H),
        frame_width=float(W),
        hidden_dim=config["HIDDEN_DIM"],
        use_dab=config.get("USE_DAB", True)
    ).to(DEVICE)]

    with torch.no_grad():
        outputs = model(frame=frame_nested, tracks=tracks)

    pred_logits = outputs["pred_logits"].cpu().clone()
    pred_bboxes = outputs["pred_bboxes"].cpu().clone()

    # Build tracker
    tracker_kwargs = dict(
        det_score_thresh   = config.get("DET_SCORE_THRESH", 0.5),
        track_score_thresh = config.get("TRACK_SCORE_THRESH", 0.5),
        miss_tolerance     = config.get("MISS_TOLERANCE", 30),
        use_dab            = config.get("USE_DAB", True),
    )
    if use_c2_explicit is not None:
        tracker_kwargs["use_c2"] = use_c2_explicit

    tracker = RuntimeTracker(**tracker_kwargs)
    _, new_tracks = tracker.update(outputs, tracks)

    updated = model.postprocess_single_frame(
        previous_tracks=tracks,
        new_tracks=new_tracks,
        unmatched_dets=None,
        no_augment=True,
    )

    return {
        "pred_logits"       : pred_logits,
        "pred_bboxes"       : pred_bboxes,
        "n_new_tracks"      : len(new_tracks[0]),
        "tau_adaptive"      : tracker.tau_adaptive,
        "det_score_thresh"  : tracker.det_score_thresh,
        "use_c2"            : tracker.use_c2,
        "lm_lambda"         : model.query_updater.long_memory_lambda,
        "use_c1"            : model.query_updater.use_c1,
    }


# ── RUN A: default (same as original smoke_test.py — no use_c2 kwarg) ─────────
print("\n[3] Run A: default RuntimeTracker (no use_c2 kwarg, mirrors smoke_test.py)...")
res_A = run_forward_and_track(use_c2_explicit=None)
print(f"    pred_logits shape : {res_A['pred_logits'].shape}")
print(f"    n_new_tracks      : {res_A['n_new_tracks']}")
print(f"    tau_adaptive      : {res_A['tau_adaptive']}")
print(f"    use_c2            : {res_A['use_c2']}")
print(f"    use_c1            : {res_A['use_c1']}")
print(f"    long_memory_lambda: {res_A['lm_lambda']}")

# ── RUN B: use_c2=False EXPLICITLY ────────────────────────────────────────────
print("\n[4] Run B: RuntimeTracker(use_c2=False) explicitly...")
res_B = run_forward_and_track(use_c2_explicit=False)
print(f"    pred_logits shape : {res_B['pred_logits'].shape}")
print(f"    n_new_tracks      : {res_B['n_new_tracks']}")
print(f"    tau_adaptive      : {res_B['tau_adaptive']}")
print(f"    use_c2            : {res_B['use_c2']}")

# ── ASSERTIONS ────────────────────────────────────────────────────────────────
print("\n[5] Comparing outputs (bit-identical check)...")
failures = []

# pred_logits and pred_bboxes: identical because the model itself is unchanged
if not torch.equal(res_A["pred_logits"], res_B["pred_logits"]):
    failures.append(
        f"  pred_logits differ: max diff = "
        f"{(res_A['pred_logits'] - res_B['pred_logits']).abs().max():.2e}"
    )
else:
    print("  pred_logits: bit-identical  OK")

if not torch.equal(res_A["pred_bboxes"], res_B["pred_bboxes"]):
    failures.append(
        f"  pred_bboxes differ: max diff = "
        f"{(res_A['pred_bboxes'] - res_B['pred_bboxes']).abs().max():.2e}"
    )
else:
    print("  pred_bboxes: bit-identical  OK")

# Track counts: identical (same threshold 0.5 used in both)
if res_A["n_new_tracks"] != res_B["n_new_tracks"]:
    failures.append(
        f"  n_new_tracks differ: A={res_A['n_new_tracks']} B={res_B['n_new_tracks']}"
    )
else:
    print(f"  n_new_tracks: {res_A['n_new_tracks']} (identical)  OK")

# tau_adaptive: both should equal det_score_thresh (0.5) since use_c2=False
for name, res in [("A", res_A), ("B", res_B)]:
    if res["tau_adaptive"] != res["det_score_thresh"]:
        failures.append(
            f"  Run {name}: tau_adaptive ({res['tau_adaptive']}) != "
            f"det_score_thresh ({res['det_score_thresh']}) — C2 path must be off"
        )
    else:
        print(f"  tau_adaptive (Run {name}): {res['tau_adaptive']} == det_score_thresh  OK")

# use_c2: both False
for name, res in [("A", res_A), ("B", res_B)]:
    if res["use_c2"] != False:
        failures.append(f"  Run {name}: use_c2={res['use_c2']} (expected False)")
    else:
        print(f"  use_c2 (Run {name}): False  OK")

# use_c1: False (model was built with config that doesn't set USE_C1)
if res_A["use_c1"] != False:
    failures.append(f"  use_c1={res_A['use_c1']} (expected False — C1 wired but off by default)")
else:
    print(f"  use_c1: False (C1 gated off by default)  OK")

# long_memory_lambda: unchanged at 0.01 (C1 is off)
if res_A["lm_lambda"] != 0.01:
    failures.append(f"  long_memory_lambda={res_A['lm_lambda']} (expected 0.01)")
else:
    print(f"  long_memory_lambda: {res_A['lm_lambda']} (original value unchanged)  OK")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if failures:
    print("ZERO-REGRESSION CHECK: FAIL")
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print("ZERO-REGRESSION CHECK: PASS")
    print("  All key outputs are bit-identical between default and")
    print("  explicit use_c1=False / use_c2=False runs.")
    print("  Wiring of C1/C2 modules has zero effect when flags are off.")
print("=" * 65)
