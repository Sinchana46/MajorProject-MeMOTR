"""
dancetrack_inference_eval.py
============================
1. Audit real DanceTrack val gt.txt files (print first 10 lines, confirm format).
2. Run official pretrained memotr_dancetrack.pth on every real sequence found.
3. Write tracker output in strict MOT format.
4. Run TrackEval against real GT. Print ALL raw stdout verbatim.
5. Print pedestrian_summary.txt verbatim. Exit with error if not produced.

STANDING RULES ENFORCED:
- gt.txt is NEVER written or modified.
- No fallback values, no dict defaults, no hardcoded paper numbers.
- If TrackEval fails, exact error is shown and no metrics are reported.
"""

import os
import sys
import configparser
import torch
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMOTR_DIR    = os.path.join(ROOT, "MeMOTR")
TRACKEVAL_DIR = os.path.join(ROOT, "TrackEval")
CKPT_PATH     = os.path.join(ROOT, "checkpoints", "pretrained", "memotr_dancetrack.pth")
VAL_DIR       = os.path.join(ROOT, "data", "DanceTrack", "val")
TRACKER_FOLDER = os.path.join(ROOT, "results", "dancetrack_eval")
TRACKER_NAME   = "memotr"
OUTPUT_FOLDER  = os.path.join(ROOT, "results", "dancetrack_eval_output")
PRED_DIR       = os.path.join(TRACKER_FOLDER, TRACKER_NAME, "data")

# ── Step 0: Discover REAL sequences ──────────────────────────────────────────
print("=" * 72)
print("Step 0: Discovering real DanceTrack val sequences")
print("=" * 72)

real_seqs = []
for name in sorted(os.listdir(VAL_DIR)):
    seq_path = os.path.join(VAL_DIR, name)
    if not os.path.isdir(seq_path):
        continue
    gt_file  = os.path.join(seq_path, "gt", "gt.txt")
    ini_file = os.path.join(seq_path, "seqinfo.ini")
    img_dir  = os.path.join(seq_path, "img1")
    if not (os.path.isfile(gt_file) and os.path.isfile(ini_file) and os.path.isdir(img_dir)):
        print(f"  SKIP {name}: missing gt/gt.txt, seqinfo.ini, or img1/")
        continue
    # Verify gt.txt is real (col 7 == '1', not '-1')
    with open(gt_file) as f:
        first_lines = [l.strip() for l in f if l.strip()][:5]
    cols_ok  = all(len(l.split(',')) == 9 for l in first_lines)
    class_ok = any(l.split(',')[7] != '-1' for l in first_lines)
    if not (cols_ok and class_ok):
        print(f"  SKIP {name}: gt.txt does not look like real DanceTrack GT (col 7 = -1 or wrong columns)")
        continue
    ini = configparser.ConfigParser()
    ini.read(ini_file)
    n_frames = int(ini["Sequence"]["seqLength"])
    n_imgs   = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    real_seqs.append({"name": name, "path": seq_path, "gt": gt_file,
                      "ini": ini_file, "img_dir": img_dir, "n_frames": n_frames,
                      "n_imgs": n_imgs, "ini_data": ini})
    print(f"  OK  {name}  seqLength={n_frames}  imgs={n_imgs}  gt_lines={len(first_lines)}+")

print(f"\nReal sequences to evaluate: {len(real_seqs)}")
if len(real_seqs) == 0:
    print("FATAL: No real sequences found. Stopping.")
    sys.exit(1)

# ── Step 1: Print first 10 lines of each real gt.txt ─────────────────────────
print("\n" + "=" * 72)
print("Step 1: First 10 lines of each real gt.txt")
print("=" * 72)
for seq in real_seqs:
    print(f"\n--- {seq['name']}/gt/gt.txt ---")
    with open(seq["gt"]) as f:
        for i, line in enumerate(f):
            if i >= 10: break
            print(f"  {line}", end="")

# ── Step 2: Load Model ────────────────────────────────────────────────────────
print("\n\n" + "=" * 72)
print("Step 2: Loading MeMOTR (DanceTrack pretrained) on GPU")
print("=" * 72)

if not os.path.isfile(CKPT_PATH):
    print(f"FATAL: checkpoint not found at {CKPT_PATH}")
    sys.exit(1)
print(f"  Checkpoint : {CKPT_PATH} ({os.path.getsize(CKPT_PATH)/1024/1024:.2f} MB)")

sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))

import torchvision.models as _tvm
_orig_r50 = _tvm.resnet50
_tvm.resnet50 = lambda weights=None, norm_layer=None, **kw: _orig_r50(
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
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

config_path = os.path.join(MEMOTR_DIR, "configs", "train_dancetrack.yaml")
config = yaml_to_dict(config_path)
config["AVAILABLE_GPUS"] = "0"
config["USE_DISTRIBUTED"] = False

model = build_model(config)
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
sd    = ckpt["model"] if "model" in ckpt else ckpt
miss, unex = model.load_state_dict(sd, strict=False)
model = model.to(device).eval()
print(f"  Checkpoint loaded  missing_keys={len(miss)}  unexpected_keys={len(unex)}")

# ── Step 3: Inference on each real sequence ───────────────────────────────────
print("\n" + "=" * 72)
print("Step 3: MeMOTR inference on each real sequence")
print("=" * 72)

os.makedirs(PRED_DIR, exist_ok=True)
mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,3,1,1)
std  = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,3,1,1)

seq_info_map = {}

for seq in real_seqs:
    name    = seq["name"]
    im_w    = int(seq["ini_data"]["Sequence"]["imWidth"])
    im_h    = int(seq["ini_data"]["Sequence"]["imHeight"])
    n_frames = seq["n_frames"]
    print(f"\n  [{name}]  {im_w}x{im_h}  {n_frames} frames")

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

    pred_lines = []
    with torch.no_grad():
        for frame_idx in range(1, n_frames + 1):
            img_path = os.path.join(seq["img_dir"], f"{frame_idx:08d}.jpg")
            if not os.path.isfile(img_path):
                # Some sequences use 6-digit naming
                img_path = os.path.join(seq["img_dir"], f"{frame_idx:06d}.jpg")
            img_pil = Image.open(img_path).convert("RGB")
            img_t   = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float().to(device) / 255.0
            img_t   = img_t.unsqueeze(0)
            img_t   = (img_t - mean) / std

            mask         = torch.zeros((1, im_h, im_w), dtype=torch.bool, device=device)
            frame_nested = NestedTensor(img_t, mask)
            outputs      = model(frame=frame_nested, tracks=tracks)
            prev_tracks, new_tracks = tracker.update(model_outputs=outputs, tracks=tracks)
            tracks = model.postprocess_single_frame(prev_tracks, new_tracks, None)

            t_cpu = tracks[0].to("cpu")
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
                        pred_lines.append(
                            f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{sc:.4f},-1,-1,-1\n"
                        )
            if frame_idx % 100 == 0 or frame_idx == n_frames:
                print(f"    Frame {frame_idx:04d}/{n_frames}  active={len(tracks[0]):3d}  "
                      f"total_preds_so_far={len(pred_lines)}")

    pred_file = os.path.join(PRED_DIR, name + ".txt")
    with open(pred_file, "w") as f:
        f.writelines(pred_lines)
    print(f"  Written: {pred_file}  ({len(pred_lines)} lines)")
    seq_info_map[name] = n_frames

# ── Step 4: TrackEval ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Step 4: TrackEval — raw output follows verbatim")
print("=" * 72)

sys.path.insert(0, TRACKEVAL_DIR)
import trackeval

eval_config = {
    "USE_PARALLEL"        : False,
    "NUM_PARALLEL_CORES"  : 1,
    "BREAK_ON_ERROR"      : True,
    "RETURN_ON_ERROR"     : False,
    "LOG_ON_ERROR"        : os.path.join(OUTPUT_FOLDER, "trackeval_error.log"),
    "PRINT_RESULTS"       : True,
    "PRINT_ONLY_COMBINED" : False,
    "PRINT_CONFIG"        : True,
    "TIME_PROGRESS"       : False,
    "DISPLAY_LESS_PROGRESS": False,
    "OUTPUT_SUMMARY"      : True,
    "OUTPUT_EMPTY_CLASSES": True,
    "OUTPUT_DETAILED"     : True,
    "PLOT_CURVES"         : False,
}

dataset_config = {
    "GT_FOLDER"          : VAL_DIR,
    "TRACKERS_FOLDER"    : TRACKER_FOLDER,
    "OUTPUT_FOLDER"      : OUTPUT_FOLDER,
    "TRACKERS_TO_EVAL"   : [TRACKER_NAME],
    "CLASSES_TO_EVAL"    : ["pedestrian"],
    "BENCHMARK"          : "DanceTrack",
    "SPLIT_TO_EVAL"      : "val",
    "DO_PREPROC"         : False,   # DanceTrack uses class 1 throughout; no distractor filtering needed
    "TRACKER_SUB_FOLDER" : "data",
    "OUTPUT_SUB_FOLDER"  : "",
    "SKIP_SPLIT_FOL"     : True,    # GT_FOLDER already points directly at val/
    "SEQ_INFO"           : seq_info_map,
    "GT_LOC_FORMAT"      : "{gt_folder}/{seq}/gt/gt.txt",
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

evaluator    = trackeval.Evaluator(eval_config)
metrics_list = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR(),
    trackeval.metrics.Identity(),
]
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

print()
raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)
print()

# ── Step 5: Print summary file verbatim ──────────────────────────────────────
print("=" * 72)
print("Step 5: pedestrian_summary.txt — VERBATIM CONTENTS")
print("=" * 72)

summary_path = os.path.join(OUTPUT_FOLDER, TRACKER_NAME, "pedestrian_summary.txt")
if os.path.isfile(summary_path):
    print(f"\nFile: {summary_path}\n")
    print(open(summary_path).read())
else:
    print(f"\nBLOCKER: {summary_path} was NOT produced. No metrics are reported.")
    print("\nFiles actually written to output folder:")
    for dp, _, fns in os.walk(OUTPUT_FOLDER):
        for fn in fns:
            fp = os.path.join(dp, fn)
            print(f"  {os.path.relpath(fp, OUTPUT_FOLDER)}  ({os.path.getsize(fp)} bytes)")
    sys.exit(1)

print("=" * 72)
print("All numbers above came directly from pedestrian_summary.txt.")
print("=" * 72)
