"""
tools/convert_visdrone.py — VisDrone-MOT → MOT Challenge format converter (Phase 2)

PURPOSE
-------
Convert VisDrone-MOT ground-truth annotation files to MOT Challenge format
(gt/gt.txt + seqinfo.ini per sequence), keeping ONLY the four vehicle classes
and discarding all pedestrian, bicycle, and other non-vehicle classes.

VisDrone-MOT details (Section 3.2 of memotr_project_prompt.md)
---------------------------------------------------------------
  - Drone aerial footage, diverse urban scenes
  - MUST use the MOT version (NOT the DET version — file structure differs)
  - This script verifies the MOT version before proceeding
  - Official train/val splits are preserved (never split by frame)
  - Train: 56 sequences / Val: 7 sequences

VEHICLE CLASSES TO KEEP (Section 3.2)
--------------------------------------
  VisDrone class_id → kept
  4  → car
  5  → van
  6  → truck
  9  → bus
  ALL other class IDs are discarded.

VisDrone-MOT ANNOTATION FORMAT (10 columns, comma-separated)
-------------------------------------------------------------
  frame, track_id, x, y, w, h, score, class_id, truncation, occlusion

  frame      : 1-indexed frame number
  track_id   : object track ID (1-indexed, per-sequence)
  x, y, w, h : bounding box top-left + size (integer pixels)
  score      : 0=ignored, 1=valid annotation
  class_id   : 0=ignored, 1=pedestrian, 2=person, 3=bicycle, 4=car,
               5=van, 6=truck, 7=motor, 8=bicycle, 9=bus, 10=awning-tricycle,
               11=tricycle
  truncation : 0=no, 1=yes
  occlusion  : 0=no, 1=partial, 2=heavy

OUTPUT (MOT Challenge format)
------------------------------
  data/VisDrone_Vehicles/{split}/{seq_name}/
    img1/        ← copied from raw sequences/{seq_name}/
    gt/gt.txt    ← 9-col MOT, only rows with class_id in {4,5,6,9}
    seqinfo.ini

gt.txt column mapping:
  frame, track_id, x, y, w, h → taken directly from annotation
  conf  → 1 (all kept rows are valid annotations, score=1)
  class → 1 (MOT convention: unified object class)
  vis   → 1 (occlusion info not passed through; use 1 for compatibility)
"""

import os
import sys
import shutil
import argparse
import configparser
import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_VISDRONE_ROOT = os.path.join(ROOT, "data", "VisDrone_Vehicles", "raw")
DEFAULT_OUT_ROOT      = os.path.join(ROOT, "data", "VisDrone_Vehicles")
DEFAULT_VERIFY_DIR    = os.path.join(ROOT, "results", "phase2_verify")

# VisDrone class IDs to KEEP — all others are discarded
VEHICLE_CLASS_IDS = {4, 5, 6, 9}
VEHICLE_CLASS_NAMES = {4: "car", 5: "van", 6: "truck", 9: "bus"}

# Class names for logging (all VisDrone classes)
ALL_CLASS_NAMES = {
    0: "ignored", 1: "pedestrian", 2: "person", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "motor",
    8: "bicycle2", 9: "bus", 10: "awning-tricycle", 11: "tricycle",
}

FRAME_RATE = 30  # VisDrone-MOT sequences are 30 fps


# ── Version check ─────────────────────────────────────────────────────────────

def verify_mot_version(split_root):
    """
    Verify the downloaded data is the MOT version, not the DET version.

    MOT version has: sequences/  and  annotations/
    DET version has: images/  and  labels/  (different file structure)

    Raises SystemExit with a clear message if the DET version is detected.
    """
    has_sequences   = os.path.isdir(os.path.join(split_root, "sequences"))
    has_annotations = os.path.isdir(os.path.join(split_root, "annotations"))
    has_images      = os.path.isdir(os.path.join(split_root, "images"))
    has_labels      = os.path.isdir(os.path.join(split_root, "labels"))

    if has_images and has_labels and not has_sequences:
        print(f"FATAL: Detected VisDrone DET version at {split_root}")
        print("       DET version has images/ and labels/ directories.")
        print("       This script requires the MOT version (sequences/ + annotations/).")
        print("       Download: https://github.com/VisDrone/VisDrone-Dataset (MOT task)")
        sys.exit(1)

    if not (has_sequences and has_annotations):
        print(f"FATAL: Cannot confirm VisDrone MOT version at {split_root}")
        print(f"  sequences/   present: {has_sequences}")
        print(f"  annotations/ present: {has_annotations}")
        sys.exit(1)

    print(f"  MOT version confirmed: sequences/ and annotations/ found.")


# ── Core conversion ───────────────────────────────────────────────────────────

def parse_visdrone_annotation(ann_path):
    """
    Parse one VisDrone-MOT annotation file.

    Yields
    ------
    tuple: (frame_idx, track_id, x, y, w, h, class_id)
       for ALL rows (both vehicle and non-vehicle) — filtering happens in caller.

    Skips rows where score == 0 (ignored regions in VisDrone convention).
    """
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue  # malformed row

            frame_idx = int(parts[0])
            track_id  = int(parts[1])
            x         = int(parts[2])
            y         = int(parts[3])
            w         = int(parts[4])
            h         = int(parts[5])
            score     = int(parts[6])
            class_id  = int(parts[7])

            if score == 0:
                continue  # skip ignored annotations

            yield frame_idx, track_id, x, y, w, h, class_id


def count_class_distribution(ann_path):
    """Return a dict of {class_id: count} for all valid rows in ann_path."""
    dist = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            score    = int(parts[6])
            class_id = int(parts[7])
            if score == 0:
                continue
            dist[class_id] = dist.get(class_id, 0) + 1
    return dist


def get_image_dims(img_dir):
    """Read actual dimensions from the first image in img_dir."""
    for fname in sorted(os.listdir(img_dir)):
        if fname.lower().endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(img_dir, fname))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return 1920, 1080  # fallback


def convert_sequence(seq_name, ann_path, img_src_dir, out_seq_dir):
    """
    Convert one VisDrone-MOT sequence to MOT Challenge format.

    Parameters
    ----------
    seq_name    : sequence name (used for seqinfo.ini)
    ann_path    : path to the annotation .txt file
    img_src_dir : path to the sequence image folder
    out_seq_dir : output folder for this sequence

    Returns
    -------
    dict with stats: seq_name, n_frames, n_tracks, n_boxes_before, n_boxes_after,
                     class_dist_before, class_dist_after
    """
    gt_dir   = os.path.join(out_seq_dir, "gt")
    img1_dir = os.path.join(out_seq_dir, "img1")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(img1_dir, exist_ok=True)

    # Count class distribution BEFORE filtering (for verification)
    dist_before = count_class_distribution(ann_path)
    n_before    = sum(dist_before.values())

    # Parse, filter to vehicle classes, write gt.txt
    rows      = []
    track_ids = set()
    frame_ids = set()
    dist_after = {}

    for frame_idx, track_id, x, y, w, h, class_id in parse_visdrone_annotation(ann_path):
        dist_after[class_id] = dist_after.get(class_id, 0) + 1
        if class_id not in VEHICLE_CLASS_IDS:
            continue  # discard non-vehicle
        # Write as MOT 9-column format
        rows.append(f"{frame_idx},{track_id},{x},{y},{w},{h},1,1,1\n")
        track_ids.add(track_id)
        frame_ids.add(frame_idx)

    # dist_after here still has all classes; filter for the kept dist
    dist_kept = {k: v for k, v in dist_after.items() if k in VEHICLE_CLASS_IDS}
    n_after   = sum(dist_kept.values())

    gt_path = os.path.join(gt_dir, "gt.txt")
    with open(gt_path, "w") as f:
        f.writelines(rows)

    n_frames = max(frame_ids) if frame_ids else 0

    # Copy images
    images_available = False
    if os.path.isdir(img_src_dir):
        images_available = True
        for fname in sorted(os.listdir(img_src_dir)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            dst = os.path.join(img1_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(img_src_dir, fname), dst)

    # Get image dimensions
    im_w, im_h = get_image_dims(img_src_dir) if os.path.isdir(img_src_dir) else (1920, 1080)

    # Write seqinfo.ini
    ini = configparser.ConfigParser()
    ini["Sequence"] = {
        "name"      : seq_name,
        "imDir"     : "img1",
        "frameRate" : str(FRAME_RATE),
        "seqLength" : str(n_frames),
        "imWidth"   : str(im_w),
        "imHeight"  : str(im_h),
        "imExt"     : ".jpg",
    }
    ini_path = os.path.join(out_seq_dir, "seqinfo.ini")
    with open(ini_path, "w") as f:
        ini.write(f)

    return {
        "seq_name"        : seq_name,
        "n_frames"        : n_frames,
        "n_tracks"        : len(track_ids),
        "n_boxes_before"  : n_before,
        "n_boxes_after"   : len(rows),
        "dist_before"     : dist_before,
        "dist_kept"       : dist_kept,
        "images_available": images_available,
    }


# ── Verification frame renderer ───────────────────────────────────────────────

def render_sample_frame(seq_dir, out_path, frame_idx=1):
    """Render first frame of a sequence with vehicle-only GT boxes overlaid."""
    gt_path  = os.path.join(seq_dir, "gt", "gt.txt")
    img1_dir = os.path.join(seq_dir, "img1")

    real_imgs = sorted(f for f in os.listdir(img1_dir)
                       if f.lower().endswith(".jpg")) if os.path.isdir(img1_dir) else []
    if real_imgs:
        canvas = cv2.imread(os.path.join(img1_dir, real_imgs[0]))
        if canvas is None:
            canvas = np.full((1080, 1920, 3), 50, dtype=np.uint8)
        label = f"Real frame {frame_idx}"
    else:
        canvas = np.full((1080, 1920, 3), 50, dtype=np.uint8)
        label  = "Blank canvas (no images)"

    n_boxes = 0
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if int(parts[0]) != frame_idx:
                continue
            tid = int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            np.random.seed(tid * 2654435761 & 0xFFFFFF)
            colour = tuple(int(c) for c in np.random.randint(80, 255, 3))
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(canvas, f"ID{tid}", (x, max(y - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            n_boxes += 1

    cv2.putText(canvas, f"VisDrone-MOT vehicle GT | {label}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Vehicles in frame: {n_boxes}", (8, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Seq: {os.path.basename(seq_dir)}", (8, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Verification frame saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def process_split(raw_split_root, out_split_root, split_name, all_stats):
    """Process one split (train or val)."""
    verify_mot_version(raw_split_root)

    ann_dir = os.path.join(raw_split_root, "annotations")
    seq_dir = os.path.join(raw_split_root, "sequences")

    ann_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".txt"))
    print(f"\n--- {split_name.upper()} ({len(ann_files)} sequences) ---")

    first_out_dir = None
    for ann_fname in ann_files:
        seq_name    = os.path.splitext(ann_fname)[0]
        ann_path    = os.path.join(ann_dir, ann_fname)
        img_src_dir = os.path.join(seq_dir, seq_name)
        out_seq_dir = os.path.join(out_split_root, seq_name)

        stats = convert_sequence(seq_name, ann_path, img_src_dir, out_seq_dir)
        stats["split"] = split_name
        all_stats.append(stats)

        if first_out_dir is None:
            first_out_dir = out_seq_dir

        pct = (stats['n_boxes_after'] / stats['n_boxes_before'] * 100
               if stats['n_boxes_before'] > 0 else 0)
        print(f"  {seq_name:30s}  frames={stats['n_frames']:5d}  "
              f"tracks={stats['n_tracks']:3d}  "
              f"boxes={stats['n_boxes_before']:7d}->{stats['n_boxes_after']:7d} "
              f"({pct:.0f}% kept)")

    return first_out_dir


def main(visdrone_root, out_root, verify_dir):
    print("=" * 70)
    print("VisDrone-MOT → MOT Challenge converter (vehicle classes only)")
    print(f"  Source : {visdrone_root}")
    print(f"  Output : {out_root}")
    print("=" * 70)

    all_stats = []
    first_seq_dir = None

    for split, subdir in [("train", "VisDrone2019-MOT-train"),
                           ("val",   "VisDrone2019-MOT-val")]:
        raw_split = os.path.join(visdrone_root, subdir)
        out_split = os.path.join(out_root, split)
        if not os.path.isdir(raw_split):
            print(f"  SKIP {split}: {raw_split} not found")
            continue
        out_dir = process_split(raw_split, out_split, split, all_stats)
        if first_seq_dir is None:
            first_seq_dir = out_dir

    # Aggregate class distribution across all sequences
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION (aggregated across all sequences)")
    print("-" * 70)

    total_before = {}
    total_after  = {}
    for s in all_stats:
        for cid, cnt in s["dist_before"].items():
            total_before[cid] = total_before.get(cid, 0) + cnt
        for cid, cnt in s["dist_kept"].items():
            total_after[cid] = total_after.get(cid, 0) + cnt

    print(f"  {'Class':<20}  {'Before filter':>14}  {'After filter':>13}  Action")
    print(f"  {'-'*20}  {'-'*14}  {'-'*13}  ------")
    for cid in sorted(total_before):
        name   = ALL_CLASS_NAMES.get(cid, f"class_{cid}")
        before = total_before[cid]
        after  = total_after.get(cid, 0)
        action = "KEPT" if cid in VEHICLE_CLASS_IDS else "discarded"
        print(f"  {name:<20}  {before:>14,}  {after:>13,}  {action}")

    print("-" * 70)
    total_b = sum(total_before.values())
    total_a = sum(total_after.values())
    print(f"  {'TOTAL':<20}  {total_b:>14,}  {total_a:>13,}  ({total_a/total_b*100:.1f}% kept)")

    print("\nSEQUENCE SUMMARY")
    train_s = [s for s in all_stats if s["split"] == "train"]
    val_s   = [s for s in all_stats if s["split"] == "val"]
    print(f"  Train sequences : {len(train_s)}")
    print(f"  Val   sequences : {len(val_s)}")
    print(f"  Total frames    : {sum(s['n_frames'] for s in all_stats):,}")
    print(f"  Vehicle boxes   : {total_a:,}")
    print("=" * 70)

    # Render verification frame
    if first_seq_dir:
        verify_path = os.path.join(verify_dir, "visdrone_sample.jpg")
        render_sample_frame(first_seq_dir, verify_path, frame_idx=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VisDrone-MOT to MOT format (vehicle classes only)")
    parser.add_argument("--visdrone-root", default=DEFAULT_VISDRONE_ROOT,
                        help="Root of raw VisDrone-MOT data")
    parser.add_argument("--out-root",      default=DEFAULT_OUT_ROOT,
                        help="Output root (writes train/ and val/ here)")
    parser.add_argument("--verify-dir",    default=DEFAULT_VERIFY_DIR,
                        help="Directory for verification artifacts")
    args = parser.parse_args()

    main(args.visdrone_root, args.out_root, args.verify_dir)
