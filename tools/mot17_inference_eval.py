"""
mot17_inference_eval.py
=======================
Step 1 : Verify real MOT17-02-FRCNN gt.txt (print first 10 lines verbatim, confirm format).
Step 2 : Run official pretrained MeMOTR checkpoint on frames 1-50 of MOT17-02-FRCNN.
Step 3 : Write tracker output in strict MOT format.
Step 4 : Run TrackEval against real GT. Print raw stdout/stderr verbatim.
Step 5 : Print pedestrian_summary.txt verbatim if it exists; report exact error if not.

RULES ENFORCED IN THIS SCRIPT:
- No fallback values. No default dict values substituted as metrics.
- If TrackEval errors, we print the exact error and exit. No numbers reported.
- gt.txt is NEVER written or modified by this script.
"""

import os
import sys
import subprocess
import torch
import numpy as np
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMOTR_DIR  = os.path.join(ROOT, "MeMOTR")
TRACKEVAL_DIR = os.path.join(ROOT, "TrackEval")

SEQ_NAME    = "MOT17-02-FRCNN"
MOT17_TRAIN = os.path.join(ROOT, "data", "MOT17", "train")
SEQ_DIR     = os.path.join(MOT17_TRAIN, SEQ_NAME)
IMG_DIR     = os.path.join(SEQ_DIR, "img1")
GT_FILE     = os.path.join(SEQ_DIR, "gt", "gt.txt")
SEQINFO     = os.path.join(SEQ_DIR, "seqinfo.ini")
CKPT_PATH   = os.path.join(ROOT, "checkpoints", "pretrained", "memotr_dancetrack.pth")

# TrackEval layout:
#   GT   : MOT17_TRAIN / SEQ_NAME / gt / gt.txt   (existing real file, untouched)
#   Pred : TRACKER_FOLDER / memotr / data / MOT17-02-FRCNN.txt
TRACKER_FOLDER = os.path.join(ROOT, "results", "mot17_eval")
TRACKER_NAME   = "memotr"
PRED_DIR       = os.path.join(TRACKER_FOLDER, TRACKER_NAME, "data")
PRED_FILE      = os.path.join(PRED_DIR, SEQ_NAME + ".txt")
OUTPUT_FOLDER  = os.path.join(ROOT, "results", "mot17_eval_output")

NUM_FRAMES = 600  # full sequence — must match real GT which covers all 600 frames

# ── Step 0: Sanity-check required inputs exist ────────────────────────────────
print("=" * 72)
print("Step 0: Checking required inputs")
print("=" * 72)
for path, label in [(GT_FILE, "GT file"), (SEQINFO, "seqinfo.ini"),
                    (CKPT_PATH, "MeMOTR checkpoint")]:
    exists = os.path.isfile(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"  {'OK' if exists else 'MISSING':7s}  {label}: {path}"
          + (f" ({size/1024/1024:.2f} MB)" if exists else ""))
    if not exists:
        print(f"\nFATAL: {label} not found. Stopping.")
        sys.exit(1)

imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
print(f"  OK       Images in img1/: {len(imgs)} frames "
      f"(first={imgs[0]}, last={imgs[-1]})")

# ── Step 1: Print first 10 lines of real gt.txt verbatim ─────────────────────
print("\n" + "=" * 72)
print(f"Step 1: First 10 lines of REAL gt.txt — {GT_FILE}")
print("=" * 72)
with open(GT_FILE, "r") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        print(f"  {line}", end="")
print()

# Basic format check: must have 9 comma-separated columns, col[7] != -1 for all
print("Format check:")
with open(GT_FILE, "r") as f:
    rows = [l.strip().split(",") for l in f if l.strip()]
cols_ok  = all(len(r) == 9 for r in rows[:20])
class_ok = any(r[7] != "-1" for r in rows[:200])
vis_ok   = len(set(r[8] for r in rows[:200])) > 1
print(f"  9 columns in first 20 rows : {'YES' if cols_ok else 'FAIL'}")
print(f"  Class column not all -1    : {'YES' if class_ok else 'FAIL'}")
print(f"  Visibility varies          : {'YES' if vis_ok else 'FAIL'}")

if not (cols_ok and class_ok):
    print("\nFATAL: gt.txt does not look like real MOT ground truth. Stopping.")
    sys.exit(1)

# ── Step 2: Load model ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Step 2: Loading MeMOTR (DanceTrack pretrained) on GPU")
print("=" * 72)

sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))

import torchvision.models as _tvm
_orig_resnet50 = _tvm.resnet50
_tvm.resnet50 = lambda weights=None, norm_layer=None, **kw: _orig_resnet50(
    weights=None, norm_layer=norm_layer, **kw)

from models import build_model
from models.runtime_tracker import RuntimeTracker
from utils.nested_tensor import NestedTensor
from utils.box_ops import box_cxcywh_to_xyxy
from structures.track_instances import TrackInstances
from utils.utils import yaml_to_dict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device : {device}")
if device == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

config_path = os.path.join(MEMOTR_DIR, "configs", "train_dancetrack.yaml")
config = yaml_to_dict(config_path)
config["AVAILABLE_GPUS"] = "0"
config["USE_DISTRIBUTED"] = False

model = build_model(config)
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
sd    = ckpt["model"] if "model" in ckpt else ckpt
miss, unex = model.load_state_dict(sd, strict=False)
model = model.to(device).eval()
print(f"  Checkpoint loaded  missing={len(miss)}  unexpected={len(unex)}")

# ── Step 3: Inference on frames 1..NUM_FRAMES ─────────────────────────────────
print("\n" + "=" * 72)
print(f"Step 3: MeMOTR inference on frames 1–{NUM_FRAMES} of {SEQ_NAME}")
print("=" * 72)

import configparser
ini = configparser.ConfigParser()
ini.read(SEQINFO)
im_w = int(ini["Sequence"]["imWidth"])
im_h = int(ini["Sequence"]["imHeight"])
print(f"  Resolution : {im_w}x{im_h}")

tracker = RuntimeTracker(
    det_score_thresh=0.5,
    track_score_thresh=0.5,
    miss_tolerance=config.get("MISS_TOLERANCE", 30),
    use_dab=config.get("USE_DAB", True),
)
tracks = [TrackInstances(
    hidden_dim=model.hidden_dim,
    num_classes=model.num_classes,
    use_dab=config.get("USE_DAB", True),
).to(device)]

mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,3,1,1)
std  = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,3,1,1)

os.makedirs(PRED_DIR, exist_ok=True)
pred_lines = []

with torch.no_grad():
    for frame_idx in range(1, NUM_FRAMES + 1):
        img_path = os.path.join(IMG_DIR, f"{frame_idx:06d}.jpg")
        img_pil  = Image.open(img_path).convert("RGB")
        img_t    = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float().to(device) / 255.0
        img_t    = img_t.unsqueeze(0)
        img_t    = (img_t - mean) / std

        mask         = torch.zeros((1, im_h, im_w), dtype=torch.bool, device=device)
        frame_nested = NestedTensor(img_t, mask)
        outputs      = model(frame=frame_nested, tracks=tracks)
        prev_tracks, new_tracks = tracker.update(model_outputs=outputs, tracks=tracks)
        tracks = model.postprocess_single_frame(prev_tracks, new_tracks, None)

        t_cpu = tracks[0].to("cpu")
        n_out = 0
        if len(t_cpu) > 0:
            keep = t_cpu.scores.max(dim=-1).values > 0.5
            t_cpu = t_cpu[keep]
            if len(t_cpu) > 0:
                boxes = box_cxcywh_to_xyxy(t_cpu.boxes) * torch.tensor(
                    [im_w, im_h, im_w, im_h], dtype=torch.float)
                for b, tid, sc in zip(boxes, t_cpu.ids, t_cpu.scores.max(dim=-1).values):
                    x1, y1, x2, y2 = b.tolist()
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    # MOT format: frame,id,x,y,w,h,conf,-1,-1,-1
                    pred_lines.append(
                        f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{sc:.4f},-1,-1,-1\n"
                    )
                    n_out += 1
        print(f"  Frame {frame_idx:03d}/{NUM_FRAMES}  active_tracks={len(tracks[0]):3d}  "
              f"written={n_out:3d}")

with open(PRED_FILE, "w") as f:
    f.writelines(pred_lines)

print(f"\n  Total prediction lines written : {len(pred_lines)}")
print(f"  Prediction file                : {PRED_FILE}")

# ── Step 4: Run TrackEval ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Step 4: Running TrackEval (raw output follows verbatim)")
print("=" * 72)

# We ran inference on all 600 frames so the sequence length matches the real
# gt.txt exactly. No SEQ_INFO override needed — TrackEval reads seqLength
# from seqinfo.ini (600) and the GT spans the same 600 frames.

sys.path.insert(0, TRACKEVAL_DIR)
import trackeval

eval_config = {
    "USE_PARALLEL"      : False,
    "NUM_PARALLEL_CORES": 1,
    "BREAK_ON_ERROR"    : True,
    "RETURN_ON_ERROR"   : False,
    "LOG_ON_ERROR"      : os.path.join(OUTPUT_FOLDER, "trackeval_error.log"),
    "PRINT_RESULTS"     : True,
    "PRINT_ONLY_COMBINED": False,
    "PRINT_CONFIG"      : True,
    "TIME_PROGRESS"     : False,
    "DISPLAY_LESS_PROGRESS": False,
    "OUTPUT_SUMMARY"    : True,
    "OUTPUT_EMPTY_CLASSES": True,
    "OUTPUT_DETAILED"   : True,
    "PLOT_CURVES"       : False,
}

dataset_config = {
    "GT_FOLDER"         : MOT17_TRAIN,       # …/data/MOT17/train — contains SEQ_NAME/gt/gt.txt
    "TRACKERS_FOLDER"   : TRACKER_FOLDER,    # …/results/mot17_eval — contains memotr/data/MOT17-02-FRCNN.txt
    "OUTPUT_FOLDER"     : OUTPUT_FOLDER,
    "TRACKERS_TO_EVAL"  : [TRACKER_NAME],
    "CLASSES_TO_EVAL"   : ["pedestrian"],
    "BENCHMARK"         : "MOT17",
    "SPLIT_TO_EVAL"     : "train",
    "DO_PREPROC"        : True,
    "TRACKER_SUB_FOLDER": "data",
    "OUTPUT_SUB_FOLDER" : "",
    "SKIP_SPLIT_FOL"    : True,              # GT and tracker folders already point at correct level
    "SEQ_INFO"          : {SEQ_NAME: NUM_FRAMES},  # 600 — matches real gt.txt exactly
    "GT_LOC_FORMAT"     : "{gt_folder}/{seq}/gt/gt.txt",
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

evaluator   = trackeval.Evaluator(eval_config)
metrics_list = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR(),
    trackeval.metrics.Identity(),
]
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

print()  # blank line before TrackEval's own output
raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)
print()

# ── Step 5: Print summary file verbatim ──────────────────────────────────────
print("=" * 72)
print("Step 5: pedestrian_summary.txt — verbatim contents")
print("=" * 72)

summary_path = os.path.join(OUTPUT_FOLDER, TRACKER_NAME, "pedestrian_summary.txt")
if os.path.isfile(summary_path):
    print(f"File: {summary_path}\n")
    with open(summary_path, "r") as f:
        contents = f.read()
    print(contents)
else:
    # List what actually was written to OUTPUT_FOLDER
    print(f"BLOCKER: pedestrian_summary.txt NOT found at {summary_path}")
    print("\nContents of output folder tree:")
    for dirpath, dirnames, filenames in os.walk(OUTPUT_FOLDER):
        rel = os.path.relpath(dirpath, OUTPUT_FOLDER)
        for fn in filenames:
            fpath = os.path.join(dirpath, fn)
            print(f"  {os.path.join(rel, fn)}  ({os.path.getsize(fpath)} bytes)")
    print("\nNo metric values are reported. TrackEval did not produce a summary.")
    sys.exit(1)

print("=" * 72)
print("DONE — all numbers above came directly from pedestrian_summary.txt")
print("=" * 72)
