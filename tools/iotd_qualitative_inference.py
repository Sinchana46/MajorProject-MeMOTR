"""
iotd_qualitative_inference.py
==============================
Runs official pretrained memotr_dancetrack.pth (zero changes, zero fine-tuning)
on each clip in data/IOTD/raw_unannotated/.
Writes output videos with predicted bounding boxes and track IDs overlaid.
NO metrics are computed — no ground truth exists for these clips.

Rendering details:
  - Each unique track ID gets a consistent colour (hash-based, bright palette).
  - Box drawn as solid-border rectangle. Track ID printed above box.
  - Frame number printed top-left. Active track count top-right.
  - Score threshold: 0.5 (same as DanceTrack eval run).
"""

import os, sys, cv2, torch, numpy as np
from PIL import Image

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMOTR_DIR   = os.path.join(ROOT, "MeMOTR")
CKPT_PATH    = os.path.join(ROOT, "checkpoints", "pretrained", "memotr_dancetrack.pth")
INPUT_DIR    = os.path.join(ROOT, "data", "IOTD", "raw_unannotated")
OUTPUT_DIR   = os.path.join(ROOT, "results", "iotd_qualitative")
SCORE_THRESH = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model Setup ───────────────────────────────────────────────────────────────
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
print(f"Device : {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

config_path = os.path.join(MEMOTR_DIR, "configs", "train_dancetrack.yaml")
config = yaml_to_dict(config_path)
config["AVAILABLE_GPUS"] = "0"
config["USE_DISTRIBUTED"] = False

model = build_model(config)
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
sd    = ckpt["model"] if "model" in ckpt else ckpt
miss, unex = model.load_state_dict(sd, strict=False)
model = model.to(device).eval()
print(f"Checkpoint loaded — missing={len(miss)}  unexpected={len(unex)}")

mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,3,1,1)
std  = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,3,1,1)

# ── Colour palette (bright, consistent per track ID) ─────────────────────────
def track_colour(tid):
    """Return a bright BGR colour consistent for this track ID."""
    np.random.seed(int(tid) * 2654435761 & 0xFFFFFFFF)
    h = np.random.randint(0, 180)
    hsv = np.array([[[h, 230, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return tuple(int(x) for x in bgr)

def draw_box(frame, x1, y1, x2, y2, tid, score, colour, thickness=2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    label = f"ID{tid} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 4, th + 2)
    cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 2, ty + 2), colour, -1)
    cv2.putText(frame, label, (x1 + 1, ty - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

# ── Per-clip inference + render ───────────────────────────────────────────────
clips = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4"))
print(f"\nClips found: {clips}\n")

for clip_name in clips:
    clip_path   = os.path.join(INPUT_DIR, clip_name)
    stem        = os.path.splitext(clip_name)[0]
    out_path    = os.path.join(OUTPUT_DIR, f"{stem}_tracked.mp4")

    cap = cv2.VideoCapture(clip_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{'='*60}")
    print(f"Processing: {clip_name}  {width}x{height}  {fps:.1f}fps  {total} frames")
    print(f"Output    : {out_path}")
    print(f"{'='*60}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Reset tracker state for each clip
    tracker = RuntimeTracker(
        det_score_thresh=SCORE_THRESH,
        track_score_thresh=SCORE_THRESH,
        miss_tolerance=config.get("MISS_TOLERANCE", 30),
        use_dab=config.get("USE_DAB", True),
    )
    tracks = [TrackInstances(
        hidden_dim=model.hidden_dim,
        num_classes=model.num_classes,
        use_dab=config.get("USE_DAB", True),
    ).to(device)]

    frame_idx   = 0
    total_dets  = 0
    seen_ids    = set()

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1

            # BGR→RGB PIL→tensor
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil   = Image.fromarray(frame_rgb)
            img_t     = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float().to(device) / 255.0
            img_t     = img_t.unsqueeze(0)
            img_t     = (img_t - mean) / std

            mask         = torch.zeros((1, height, width), dtype=torch.bool, device=device)
            frame_nested = NestedTensor(img_t, mask)
            outputs      = model(frame=frame_nested, tracks=tracks)
            prev_tracks, new_tracks = tracker.update(model_outputs=outputs, tracks=tracks)
            tracks = model.postprocess_single_frame(prev_tracks, new_tracks, None)

            t_cpu = tracks[0].to("cpu")
            render = frame_bgr.copy()
            n_drawn = 0

            if len(t_cpu) > 0:
                keep = t_cpu.scores.max(dim=-1).values > SCORE_THRESH
                t_cpu_filt = t_cpu[keep]
                if len(t_cpu_filt) > 0:
                    boxes = box_cxcywh_to_xyxy(t_cpu_filt.boxes) * torch.tensor(
                        [width, height, width, height], dtype=torch.float)
                    for b, tid, sc in zip(boxes, t_cpu_filt.ids,
                                          t_cpu_filt.scores.max(dim=-1).values):
                        x1, y1, x2, y2 = b.tolist()
                        # Clamp to frame bounds
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(width, x2); y2 = min(height, y2)
                        tid_int = int(tid)
                        seen_ids.add(tid_int)
                        draw_box(render, x1, y1, x2, y2,
                                 tid_int, float(sc), track_colour(tid_int))
                        n_drawn += 1
                total_dets += n_drawn

            # HUD: frame number + active track count
            cv2.putText(render, f"Frame {frame_idx}/{total}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(render, f"Active: {n_drawn}", (width - 130, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(render)

            if frame_idx % 200 == 0 or frame_idx == total:
                print(f"  Frame {frame_idx:04d}/{total}  active={n_drawn:3d}  "
                      f"unique_ids_so_far={len(seen_ids)}")

    cap.release()
    writer.release()
    print(f"\n  DONE: {frame_idx} frames rendered  "
          f"total_det_instances={total_dets}  unique_track_ids={len(seen_ids)}")
    print(f"  Saved: {out_path}\n")

print("All clips processed.")
