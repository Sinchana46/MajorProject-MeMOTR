"""
Check 3: Numeric paint test for box alignment in the actual training path.
"""
import sys, os, random
import numpy as np
import torch

sys.path.insert(0, "MeMOTR")
sys.path.insert(0, "MeMOTR/models/ops")

from data.vehiclemot import build as build_vehiclemot

def yaml_to_dict(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    from models.utils import load_config
    return load_config(path)

try:
    cfg = yaml_to_dict("configs/train_iotd_real_aug.yaml")
except Exception as e:
    # Fallback: manual config parse
    import yaml
    with open("configs/train_iotd_real_aug.yaml") as f:
        cfg = yaml.safe_load(f)

cfg["DATA_ROOT"]    = "D:/MajorProject-MeMOTR/data"
cfg["DATASET_DIRS"] = ["IOTD_train_real", "IOTD_train_aug"]

print("=== CHECK 3.2: VehicleMOT transform list for train split ===")
import inspect
from data.vehiclemot import build as build_vehiclemot
src = inspect.getsource(build_vehiclemot)
# Print transform-related lines
for i, line in enumerate(src.split("\n")):
    if any(x in line for x in ["transform", "Resize", "Normalize", "Flip", "RandomHorizontalFlip",
                                "ColorJitter", "RandomPhotometric", "ToTensor", "Compose"]):
        print(f"  {line}")

print("\n=== CHECK 3.3: PAINT TEST ===")
random.seed(999); torch.manual_seed(999); np.random.seed(999)

ds = build_vehiclemot(config=cfg, split="train")
print(f"Dataset: {len(ds.sample_begin_frames)} clip starts")

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)
TOL = 30  # color tolerance

pass_count = 0
fail_count = 0
total_evaluated = 0
worst_box_cover = 1.0
worst_outside_frac = 0.0

save_dir = "check/box_alignment_diagnostic_v2"
os.makedirs(save_dir, exist_ok=True)
saved = 0

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image, ImageDraw

# Sample 200 draws
N = min(200, len(ds.sample_begin_frames))
step = max(1, len(ds.sample_begin_frames) // N)
indices = list(range(0, len(ds.sample_begin_frames), step))[:N]

for sample_idx in indices:
    try:
        item = ds[sample_idx]
    except Exception as e:
        continue

    imgs = item.get("imgs") or item.get("images")
    infos = item.get("infos") or item.get("targets")
    if imgs is None or infos is None:
        imgs = item[0] if isinstance(item, (list, tuple)) else None
        infos = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else None
    if imgs is None:
        continue

    # Process first frame of clip
    for fi in range(min(2, len(imgs))):
        img_t = imgs[fi]  # [3, H, W] float tensor normalized
        info = infos[fi] if isinstance(infos, (list, tuple)) else infos
        boxes = info.get("boxes", info.get("gt_boxes", None))
        if boxes is None or len(boxes) == 0:
            continue

        H, W = img_t.shape[-2], img_t.shape[-1]

        # De-normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnorm = (img_t * std + mean).clamp(0, 1)
        np_img = (unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()

        # Paint magenta inside GT boxes
        painted = np_img.copy()
        pixel_boxes = []
        for box in boxes:
            b = box.tolist()
            cx, cy, bw, bh = b
            x1 = int(round(max(0, (cx - bw/2) * W)))
            y1 = int(round(max(0, (cy - bh/2) * H)))
            x2 = int(round(min(W-1, (cx + bw/2) * W)))
            y2 = int(round(min(H-1, (cy + bh/2) * H)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            painted[y1:y2, x1:x2] = MAGENTA
            pixel_boxes.append((x1, y1, x2, y2))

        if not pixel_boxes:
            continue

        # Build mask of painted pixels
        mask = np.all(np.abs(painted.astype(int) - MAGENTA.astype(int)) <= TOL, axis=2)
        total_mask = mask.sum()
        if total_mask == 0:
            continue

        # For each box: coverage = mask pixels inside box / box area
        outside_mask = total_mask
        all_pass = True
        for (x1, y1, x2, y2) in pixel_boxes:
            box_pixels = mask[y1:y2, x1:x2].sum()
            box_area = (x2-x1) * (y2-y1)
            coverage = box_pixels / box_area if box_area > 0 else 0
            outside_mask -= box_pixels
            total_evaluated += 1
            if coverage < worst_box_cover:
                worst_box_cover = coverage
            if coverage < 0.85:
                all_pass = False
                fail_count += 1
            else:
                pass_count += 1

        outside_frac = outside_mask / total_mask if total_mask > 0 else 0
        if outside_frac > worst_outside_frac:
            worst_outside_frac = outside_frac

        if outside_frac > 0.10:
            all_pass = False

        # Save diagnostic images (max 10)
        if saved < 10:
            if HAS_CV2:
                vis = painted.copy()
                for (x1, y1, x2, y2) in pixel_boxes:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                out_path = os.path.join(save_dir, f"sample_{sample_idx:04d}_f{fi}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            saved += 1

print(f"Boxes evaluated: {total_evaluated}")
print(f"Pass (coverage >= 0.85): {pass_count}")
print(f"Fail (coverage < 0.85): {fail_count}")
pass_rate = pass_count / total_evaluated if total_evaluated > 0 else 0
print(f"Pass rate: {pass_rate:.1%}")
print(f"Worst box coverage: {worst_box_cover:.4f}")
print(f"Worst outside-box mask fraction: {worst_outside_frac:.4f}")
print(f"Diagnostic images saved: {saved} -> {save_dir}/")

if pass_rate >= 0.95 and worst_outside_frac <= 0.10:
    print("\nCHECK 3 (PAINT TEST): PASS")
else:
    print(f"\nCHECK 3 (PAINT TEST): FAIL  pass_rate={pass_rate:.1%} outside={worst_outside_frac:.4f}")
    sys.exit(1)
