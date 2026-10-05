"""
tools/run_inference_mot.py — Phase 5/6 inference script

Purpose: Run a trained MeMOTR checkpoint on all sequences in a split directory,
writing MOT-format per-sequence tracker output files.

Designed to work with our VehicleMOT data layout:
  <split_dir>/<seq>/img1/<frames>
  <split_dir>/<seq>/seqinfo.ini   (for frame count / size)

Output layout (MOT format):
  <output_dir>/<seq>.txt  — one file per sequence,
                             columns: frame, id, left, top, w, h, 1, -1, -1, -1

Usage (run from MeMOTR/):
  set PYTHONPATH=D:/MajorProject-MeMOTR/MeMOTR/models/ops;D:/MajorProject-MeMOTR
  python ../tools/run_inference_mot.py ^
      --checkpoint  ../results/phase5_interim_baseline/checkpoint_4.pth ^
      --split-dirs  ../data/UA_DETRAC/test ../data/VisDrone_Vehicles/val ^
      --output-dir  ../results/phase5_interim_baseline/tracker_val ^
      --config      ../configs/train_vehicle_mot_interim.yaml ^
      [--label INTERIM_BASELINE]
"""

from __future__ import annotations
import argparse
import os
import sys
import glob
import configparser

import torch
import numpy as np
from PIL import Image

# ── Path setup (same as smoke_test.py) ──────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MEMOTR_DIR  = os.path.join(PROJECT_DIR, "MeMOTR")
sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))
sys.path.insert(0, PROJECT_DIR)

import torchvision.transforms.functional as TF

from utils.utils import yaml_to_dict
from utils.nested_tensor import NestedTensor
from models import build_model
from models.utils import load_checkpoint
from models.runtime_tracker import RuntimeTracker
from structures.track_instances import TrackInstances
from modules.scene_analyzer import SceneAnalyzer
from modules.threshold_adapter import ThresholdAdapter


# ── Image filename auto-detection (mirrors vehiclemot.py) ─────────────────────

def _detect_img_format(img1_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(img1_dir, "*.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(img1_dir, "*.png")))
    if not files:
        raise FileNotFoundError(f"No images in {img1_dir}")
    basename = os.path.basename(files[0])
    ext = os.path.splitext(basename)[1]
    stem = os.path.splitext(basename)[0]
    prefix, digits = "", stem
    for k, ch in enumerate(stem):
        if ch.isdigit():
            prefix, digits = stem[:k], stem[k:]
            break
    n = len(digits)
    return (f"{prefix}{{:0{n}d}}{ext}") if prefix else (f"{{:0{n}d}}{ext}")


# ── Preprocessing (same as DanceTrack eval transforms) ────────────────────────

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])
MAX_SIZE = 1333
EVAL_SIZE = 800


def preprocess_frame(img: Image.Image, device: str):
    """Resize (keep AR, short side = EVAL_SIZE, long side <= MAX_SIZE), normalise."""
    w, h = img.size
    scale = EVAL_SIZE / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    if max(new_h, new_w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    t = TF.to_tensor(img)                          # (3, H, W) float [0, 1]
    t = TF.normalize(t, mean=MEAN, std=STD)
    # Build NestedTensor (no padding needed — batch size = 1)
    mask = torch.zeros(1, t.shape[1], t.shape[2], dtype=torch.bool)
    nested = NestedTensor(t.unsqueeze(0).to(device), mask.to(device))
    return nested, (w, h), (new_w, new_h)


# ── Per-sequence inference ────────────────────────────────────────────────────

def infer_sequence(seq_dir: str, model, config: dict, device: str,
                   det_thresh: float, track_thresh: float,
                   miss_tolerance: int,
                   scene_analyzer=None, threshold_adapter=None) -> list[str]:
    """
    Run MeMOTR on one sequence.  Returns list of MOT-format lines.
    """
    img1_dir  = os.path.join(seq_dir, "img1")
    img_fmt   = _detect_img_format(img1_dir)

    # Frame list — sorted numerically by extracting digits from filename
    all_files = sorted(
        glob.glob(os.path.join(img1_dir, "*.jpg")) +
        glob.glob(os.path.join(img1_dir, "*.png")),
        key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0')
    )
    n_frames = len(all_files)

    # Read seqinfo for frame size
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(seqinfo_path):
        cp = configparser.ConfigParser()
        cp.read(seqinfo_path)
        orig_w = int(cp["Sequence"].get("imwidth",  "1280"))
        orig_h = int(cp["Sequence"].get("imheight", "720"))
    else:
        orig_w, orig_h = 1280, 720

    use_c2 = config.get("USE_C2", False)
    tracker = RuntimeTracker(
        det_score_thresh   = det_thresh,
        track_score_thresh = track_thresh,
        miss_tolerance     = miss_tolerance,
        use_dab            = config.get("USE_DAB", True),
        use_c2             = use_c2,
    )

    # If C2 enabled, load adapter weights into tracker's own ThresholdAdapter
    if use_c2 and threshold_adapter is not None:
        tracker.threshold_adapter.load_state_dict(threshold_adapter.state_dict())
        tracker.threshold_adapter.eval()
        if scene_analyzer is not None:
            # SceneAnalyzer is analytic — no weights to copy, just ensure it's ready
            pass

    tracks = [TrackInstances(
        frame_height = float(orig_h),
        frame_width  = float(orig_w),
        hidden_dim   = config["HIDDEN_DIM"],
        use_dab      = config.get("USE_DAB", True),
    ).to(device)]

    mot_lines = []

    with torch.no_grad():
        for frame_idx, img_path in enumerate(all_files, start=1):
            img = Image.open(img_path).convert("RGB")
            frame_nested, (orig_w_f, orig_h_f), (new_w, new_h) = preprocess_frame(img, device)

            outputs = model(frame=frame_nested, tracks=tracks)

            # C2: inject raw (un-normalized) frame so RuntimeTracker.update() can compute tau_adaptive
            if use_c2:
                raw_t = TF.to_tensor(img).to(device)   # (3,H,W) float [0,1]
                outputs["frame"] = raw_t

            tracks, new_tracks = tracker.update(outputs, tracks)

            # Postprocess: merge new tracks back
            tracks = model.postprocess_single_frame(
                previous_tracks = tracks,
                new_tracks      = new_tracks,
                unmatched_dets  = None,
                no_augment      = True,
            )

            # Scale predicted boxes back to original image size
            # Model outputs: cx,cy,w,h in [0,1] relative to the PROCESSED frame
            scale_x = orig_w_f / new_w
            scale_y = orig_h_f / new_h

            all_track_instances = tracks[0]
            if len(all_track_instances) == 0:
                continue

            boxes  = all_track_instances.boxes.cpu()    # (N, 4) cx,cy,w,h in [0,1]
            ids    = all_track_instances.ids.cpu()      # (N,)
            scores = all_track_instances.scores.cpu()   # (N, num_classes)

            for n in range(len(all_track_instances)):
                if ids[n] < 0:
                    continue
                cx, cy, bw, bh = boxes[n].tolist()
                # Convert cxcywh [0,1] -> xyxy in processed-frame pixels -> original pixels
                x1 = (cx - bw / 2) * new_w * scale_x
                y1 = (cy - bh / 2) * new_h * scale_y
                w  = bw * new_w * scale_x
                h  = bh * new_h * scale_y
                score = float(scores[n].max())
                mot_lines.append(
                    f"{frame_idx},{int(ids[n])},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
                )

    return mot_lines


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("MeMOTR VehicleMOT inference")
    p.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint")
    p.add_argument("--split-dirs",  nargs="+",     required=True,
                   help="One or more directories containing MOT sequences")
    p.add_argument("--output-dir",  required=True, help="Where to write tracker .txt files")
    p.add_argument("--config",      required=True, help="YAML config path")
    p.add_argument("--label",       default="",    help="Label string written to output_dir/label.txt")
    p.add_argument("--c2-adapter-checkpoint", default="",
                   help="Path to c2_adapter_epoch_N.pth for C2 inference (optional)")
    p.add_argument("--det-thresh",  type=float,    default=0.5)
    p.add_argument("--track-thresh",type=float,    default=0.5)
    p.add_argument("--miss-tol",    type=int,      default=30)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load config
    config = yaml_to_dict(args.config)
    config["USE_DISTRIBUTED"] = False
    config["AVAILABLE_GPUS"]  = "0"
    config["USE_CHECKPOINT"]  = False
    config["VISUALIZE"]       = False

    # Build model
    print("Building model...")
    model = build_model(config=config)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    states = {}
    load_checkpoint(model=model, path=args.checkpoint, states=states)
    model.to(device)
    print(f"  Checkpoint epoch: {states.get('start_epoch', '?')}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write label file
    if args.label:
        with open(os.path.join(args.output_dir, "label.txt"), "w") as f:
            f.write(args.label + "\n")

    # C2: load ThresholdAdapter if requested
    scene_analyzer    = None
    threshold_adapter = None
    if args.c2_adapter_checkpoint and os.path.isfile(args.c2_adapter_checkpoint):
        print(f"Loading C2 adapter: {args.c2_adapter_checkpoint}")
        ckpt = torch.load(args.c2_adapter_checkpoint, map_location="cpu")
        scene_analyzer    = SceneAnalyzer(max_tracks=50).to(device)
        threshold_adapter = ThresholdAdapter(
            tau_min=ckpt.get("tau_min", 0.3),
            tau_max=ckpt.get("tau_max", 0.7),
        ).to(device)
        threshold_adapter.load_state_dict(ckpt["state_dict"])
        threshold_adapter.eval()
        print(f"  C2 adapter loaded (epoch={ckpt.get('epoch','?')}, "
              f"tau_min={ckpt.get('tau_min',0.3)}, tau_max={ckpt.get('tau_max',0.7)})")
        config["USE_C2"] = True
    elif config.get("USE_C2", False):
        print("WARNING: USE_C2=True in config but no --c2-adapter-checkpoint provided. "
              "Falling back to fixed threshold.")

    # Collect all sequences
    sequences = []
    for split_dir in args.split_dirs:
        if not os.path.isdir(split_dir):
            print(f"  WARNING: {split_dir} not found, skipping")
            continue
        for seq in sorted(os.listdir(split_dir)):
            seq_path = os.path.join(split_dir, seq)
            img1_path = os.path.join(seq_path, "img1")
            if os.path.isdir(img1_path):
                sequences.append((seq, seq_path))

    print(f"Found {len(sequences)} sequences across {len(args.split_dirs)} split dir(s)")

    for i, (seq_name, seq_dir) in enumerate(sequences):
        print(f"  [{i+1}/{len(sequences)}] {seq_name} ...", end=" ", flush=True)
        lines = infer_sequence(
            seq_dir       = seq_dir,
            model         = model,
            config        = config,
            device        = device,
            det_thresh    = args.det_thresh,
            track_thresh  = args.track_thresh,
            miss_tolerance= args.miss_tol,
            scene_analyzer    = scene_analyzer,
            threshold_adapter = threshold_adapter,
        )
        out_path = os.path.join(args.output_dir, f"{seq_name}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        print(f"{len(lines)} detections")

        # Reset tracker state between sequences
        # (tracker is re-created per sequence inside infer_sequence)

    print(f"\nInference complete. Tracker files in: {args.output_dir}")


if __name__ == "__main__":
    main()
