"""
tools/train_c2_only.py — Phase 6 C2: ThresholdAdapter training

Purpose: Train ONLY the ThresholdAdapter (209 params) on top of a frozen
MeMOTR baseline. All MeMOTR weights are frozen and verified via assertion.

Contribution 2 (C2): Scene-conditioned detection threshold.
  - SceneAnalyzer (analytic, 0 params): extracts [brightness, contrast, density]
  - ThresholdAdapter (209 params): 3->16->8->1 MLP, output in [tau_min, tau_max]
  - tau_adaptive replaces fixed det_score_thresh in RuntimeTracker.update()

Training signal / proxy loss
-----------------------------
We don't have IOTD-specific night/dusk scene labels yet, so we use a brightness-
based proxy: for each batch, we compute the scene brightness from the current
frame and define a target tau:
  - bright frame (brightness > 0.6): target tau = tau_max * 0.9  (high threshold)
  - mid frame   (0.3 < brightness <= 0.6): target tau = (tau_min + tau_max) / 2
  - dark frame  (brightness <= 0.3): target tau = tau_min * 1.1  (low threshold)

Loss: L2(tau_predicted, tau_target)

This teaches the adapter the directional prior: dark -> lower tau.
Full scene-specific calibration will happen in Phase 6 proper with IOTD data.

Label: "C2_INTERIM_no_IOTD_2026-08-29"

Usage (run from MeMOTR/):
  set PYTHONPATH=D:/MajorProject-MeMOTR/MeMOTR/models/ops;D:/MajorProject-MeMOTR
  python ../tools/train_c2_only.py ^
      --baseline-checkpoint ../results/phase5_interim_baseline/checkpoint_4.pth ^
      --config              ../configs/train_vehicle_mot_interim.yaml ^
      --output-dir          ../results/phase6_c2_interim ^
      --epochs              5 ^
      --lr                  1e-3 ^
      --batch-size          1

Checkpoints saved to: <output-dir>/c2_adapter_epoch_N.pth
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import glob

import torch
import torch.nn as nn
from torch.optim import Adam
from PIL import Image
import torchvision.transforms.functional as TF

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MEMOTR_DIR  = os.path.join(PROJECT_DIR, "MeMOTR")
sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))
sys.path.insert(0, PROJECT_DIR)

from utils.utils import yaml_to_dict
from utils.nested_tensor import NestedTensor
from models import build_model
from models.utils import load_checkpoint
from structures.track_instances import TrackInstances

from modules.scene_analyzer    import SceneAnalyzer
from modules.threshold_adapter import ThresholdAdapter


# ── Preprocessing (same as eval) ──────────────────────────────────────────────

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])


def preprocess_frame(img: Image.Image, device: str):
    scale = 800 / min(img.size)
    new_w = int(round(img.width  * scale))
    new_h = int(round(img.height * scale))
    if max(new_w, new_h) > 1333:
        scale = 1333 / max(img.size)
        new_w = int(round(img.width  * scale))
        new_h = int(round(img.height * scale))
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    t = TF.to_tensor(img_r)
    t = TF.normalize(t, mean=MEAN, std=STD)
    mask   = torch.zeros(1, t.shape[1], t.shape[2], dtype=torch.bool)
    nested = NestedTensor(t.unsqueeze(0).to(device), mask.to(device))
    return nested, t   # t is the raw (unnormalized-domain) tensor for SceneAnalyzer


def raw_frame_tensor(img: Image.Image) -> torch.Tensor:
    """Return (3, H, W) float [0,1] for SceneAnalyzer (no normalization)."""
    t = TF.to_tensor(img)   # (3, H, W) float [0, 1]
    return t


# ── Target tau from brightness ─────────────────────────────────────────────────

TAU_MIN = 0.3
TAU_MAX = 0.7


def brightness_target_tau(brightness: float) -> float:
    """Proxy supervision: dark frames should have lower tau."""
    if brightness <= 0.3:
        return TAU_MIN * 1.1                         # ~0.33
    elif brightness >= 0.6:
        return TAU_MAX * 0.9                         # ~0.63
    else:
        return (TAU_MIN + TAU_MAX) / 2.0             # 0.50


# ── Collect training frames ────────────────────────────────────────────────────

def collect_frames(data_root: str, max_per_seq: int = 50) -> list[str]:
    """
    Collect up to max_per_seq evenly-spaced frame paths from each sequence
    under data_root/UA_DETRAC/train and data_root/VisDrone_Vehicles/train.
    Returns a flat list of image paths.
    """
    frame_paths = []
    for dataset in ["UA_DETRAC/train", "VisDrone_Vehicles/train"]:
        dataset_dir = os.path.join(data_root, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"  [C2] WARNING: {dataset_dir} not found")
            continue
        for seq in sorted(os.listdir(dataset_dir)):
            img1 = os.path.join(dataset_dir, seq, "img1")
            if not os.path.isdir(img1):
                continue
            files = sorted(
                glob.glob(os.path.join(img1, "*.jpg")) +
                glob.glob(os.path.join(img1, "*.png")),
                key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0')
            )
            if not files:
                continue
            step = max(1, len(files) // max_per_seq)
            frame_paths.extend(files[::step][:max_per_seq])
    return frame_paths


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Phase 6 C2: ThresholdAdapter training")
    p.add_argument("--baseline-checkpoint", required=True,
                   help="Phase 5 MeMOTR checkpoint (.pth)")
    p.add_argument("--config",   required=True, help="YAML config path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs",   type=int,   default=5)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Frames per gradient step (currently only 1 supported)")
    p.add_argument("--max-frames-per-seq", type=int, default=50)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "c2_train_log.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("=" * 65)
    log("Phase 6 C2: ThresholdAdapter training (interim, no IOTD)")
    log(f"Label: C2_INTERIM_no_IOTD_2026-08-29")
    log(f"Epochs: {args.epochs}, LR: {args.lr}")
    log("=" * 65)

    # ── Build & freeze MeMOTR ─────────────────────────────────────────────────
    config = yaml_to_dict(args.config)
    config["USE_DISTRIBUTED"] = False
    config["USE_C2"] = True   # enable C2 path in model / tracker

    log("\nBuilding MeMOTR model...")
    model = build_model(config=config)
    states = {}
    load_checkpoint(model=model, path=args.baseline_checkpoint, states=states)
    model.to(device)
    model.eval()

    # Freeze ALL MeMOTR parameters
    for name, param in model.named_parameters():
        param.requires_grad_(False)

    # Verify freeze — must be 0 trainable params in MeMOTR
    trainable_memotr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_memotr == 0, \
        f"ASSERTION FAILED: {trainable_memotr} MeMOTR params still trainable after freeze!"
    log(f"MeMOTR frozen: {sum(p.numel() for p in model.parameters()):,} params, "
        f"0 trainable (verified).")

    # ── Instantiate ThresholdAdapter + SceneAnalyzer ──────────────────────────
    scene_analyzer    = SceneAnalyzer(max_tracks=50).to(device)
    threshold_adapter = ThresholdAdapter(tau_min=TAU_MIN, tau_max=TAU_MAX).to(device)
    threshold_adapter.train()

    trainable_c2 = sum(p.numel() for p in threshold_adapter.parameters() if p.requires_grad)
    log(f"ThresholdAdapter: {trainable_c2} trainable params")

    optimizer = Adam(threshold_adapter.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ── Collect training frames ───────────────────────────────────────────────
    data_root = config["DATA_ROOT"]
    log(f"\nCollecting frames from {data_root}...")
    frame_paths = collect_frames(data_root, max_per_seq=args.max_frames_per_seq)
    log(f"  Total frames for C2 training: {len(frame_paths)}")

    if len(frame_paths) == 0:
        log("ERROR: No frames found. Check DATA_ROOT in config.")
        sys.exit(1)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        n_steps    = 0

        # Shuffle frames each epoch
        import random
        random.shuffle(frame_paths)

        optimizer.zero_grad()

        for step, img_path in enumerate(frame_paths):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                log(f"  WARNING: could not open {img_path}: {e}")
                continue

            # Raw tensor for scene analyzer (un-normalised, (3,H,W) float [0,1])
            raw_t = raw_frame_tensor(img)   # CPU

            # Compute scene vector (analytic, no grad) — always returns CPU tensor
            with torch.no_grad():
                scene_vec = scene_analyzer(raw_t, active_track_count=0)
                brightness = scene_vec[0].item()

            # Move to device before ThresholdAdapter forward (adapter lives on GPU)
            scene_vec = scene_vec.to(device)

            # Predict tau
            tau_pred = threshold_adapter(scene_vec.detach())   # scalar

            # Compute proxy target and loss
            tau_target = torch.tensor(
                brightness_target_tau(brightness),
                dtype=torch.float32,
                device=device
            )
            loss = criterion(tau_pred, tau_target)
            loss.backward()

            total_loss += loss.item()
            n_steps    += 1

            if (step + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % args.log_every == 0:
                avg = total_loss / n_steps
                log(f"  [Epoch={epoch}, Step={step+1}/{len(frame_paths)}] "
                    f"avg_loss={avg:.6f}, last_brightness={brightness:.3f}, "
                    f"tau_pred={tau_pred.item():.4f}, tau_target={tau_target.item():.4f}")

        # Final optimizer step for remaining frames
        optimizer.step()
        optimizer.zero_grad()

        epoch_min = (time.time() - epoch_start) / 60
        avg_loss  = total_loss / max(n_steps, 1)
        log(f"\n[Epoch {epoch}] avg_loss={avg_loss:.6f}  ({epoch_min:.1f} min)")

        # Save ThresholdAdapter checkpoint
        ckpt_path = os.path.join(args.output_dir, f"c2_adapter_epoch_{epoch}.pth")
        torch.save({
            "epoch"              : epoch,
            "state_dict"         : threshold_adapter.state_dict(),
            "avg_loss"           : avg_loss,
            "tau_min"            : TAU_MIN,
            "tau_max"            : TAU_MAX,
            "label"              : "C2_INTERIM_no_IOTD_2026-08-29",
            "baseline_checkpoint": args.baseline_checkpoint,
        }, ckpt_path)
        log(f"  Checkpoint saved: {ckpt_path}")

    log("\nPhase 6 C2 training complete.")
    log(f"Final adapter checkpoint: c2_adapter_epoch_{args.epochs - 1}.pth")


if __name__ == "__main__":
    main()
