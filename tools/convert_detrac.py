"""
tools/convert_detrac.py — UA-DETRAC XML → MOT Challenge format converter (Phase 2)

PURPOSE
-------
Convert the official UA-DETRAC ground-truth XML files (one per video sequence)
into MOT Challenge format (gt/gt.txt + seqinfo.ini per sequence), matching the
directory layout defined in Section 11 of memotr_project_prompt.md.

UA-DETRAC details (Section 3.1 of memotr_project_prompt.md)
------------------------------------------------------------
  - 100 video sequences, ~140,000 frames
  - Official 60/40 train/test split (60 train sequences, 40 test sequences)
  - Source annotation format: one XML file per sequence
  - 4 vehicle classes: car, bus, van, other
  - Resolution: 960 × 540 @ 25 FPS
  - Download: https://detrac-db.rit.albany.edu/

XML SCHEMA (observed)
---------------------
  <sequence name="MVI_XXXXX">
    <sequence_attribute camera_state="..." sence_weather="..."/>
    <ignored_region>
      <box left=... top=... width=... height=.../>  ← ignore regions, NOT used
    </ignored_region>
    <frame density="N" num="FRAME_IDX">
      <target_list>
        <target id="TRACK_ID">
          <box left="X" top="Y" width="W" height="H"/>
          <attribute vehicle_type="car|van|bus|other" .../>
        </target>
        ...
      </target_list>
    </frame>
    ...
  </sequence>

OUTPUT (MOT Challenge format per sequence)
------------------------------------------
  data/UA_DETRAC/{split}/{seq_name}/
    img1/          ← symlinked or copied from raw (images are NOT re-encoded)
    gt/gt.txt      ← 9-column MOT: frame,id,x,y,w,h,conf,class,visibility
    seqinfo.ini

gt.txt column meaning:
  frame  : 1-indexed frame number (matches <frame num=...>)
  id     : track ID (matches <target id=...>)
  x,y    : top-left corner (from <box left top>)
  w,h    : width and height (from <box width height>)
  conf   : always 1 (annotation is ground truth)
  class  : always 1 (MOT convention: single pedestrian/object class)
  vis    : always 1 (UA-DETRAC does not provide visibility; set to 1)
"""

import os
import sys
import xml.etree.ElementTree as ET
import configparser
import shutil
import argparse
import cv2
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default paths (can be overridden via CLI)
DEFAULT_DETRAC_ROOT = os.path.join(ROOT, "data", "UA_DETRAC", "raw")
DEFAULT_OUT_ROOT    = os.path.join(ROOT, "data", "UA_DETRAC")
DEFAULT_VERIFY_DIR  = os.path.join(ROOT, "results", "phase2_verify")

# UA-DETRAC standard properties
FRAME_RATE  = 25
IMG_WIDTH   = 960
IMG_HEIGHT  = 540


# ── Core parsing ─────────────────────────────────────────────────────────────

def parse_detrac_xml(xml_path):
    """
    Parse one UA-DETRAC XML file and yield annotation rows.

    Yields
    ------
    tuple: (frame_idx: int, track_id: int, x: float, y: float,
             w: float, h: float, vehicle_type: str)

    Notes
    -----
    - frame_idx comes from <frame num="..."> attribute (1-indexed, as-is)
    - track_id  comes from <target id="..."> attribute
    - x, y, w, h come from <box left top width height> (floating point)
    - <ignored_region> boxes are deliberately skipped
    - Frames with no targets produce no rows (empty frame gap)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for frame_el in root.findall("frame"):
        frame_idx = int(frame_el.get("num"))          # 1-indexed

        target_list = frame_el.find("target_list")
        if target_list is None:
            continue

        for target in target_list.findall("target"):
            track_id = int(target.get("id"))

            box = target.find("box")
            x   = float(box.get("left"))
            y   = float(box.get("top"))
            w   = float(box.get("width"))
            h   = float(box.get("height"))

            attr         = target.find("attribute")
            vehicle_type = attr.get("vehicle_type", "unknown") if attr is not None else "unknown"

            yield frame_idx, track_id, x, y, w, h, vehicle_type


def convert_sequence(xml_path, img_src_dir, out_seq_dir):
    """
    Convert one UA-DETRAC XML sequence to MOT Challenge format.

    Parameters
    ----------
    xml_path    : path to the sequence XML file
    img_src_dir : path to the raw image folder for this sequence
                  (may be None if images are not available)
    out_seq_dir : output root for this sequence
                  (creates img1/, gt/, seqinfo.ini inside)

    Returns
    -------
    dict with stats: seq_name, n_frames, n_tracks, n_boxes
    """
    seq_name = os.path.splitext(os.path.basename(xml_path))[0]

    # Create output directory structure
    gt_dir   = os.path.join(out_seq_dir, "gt")
    img1_dir = os.path.join(out_seq_dir, "img1")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(img1_dir, exist_ok=True)

    # Parse annotations and write gt.txt
    rows      = []
    track_ids = set()
    frame_ids = set()

    for frame_idx, track_id, x, y, w, h, _ in parse_detrac_xml(xml_path):
        # MOT format: frame,id,x,y,w,h,conf,class,visibility
        # Round coords to 2 decimal places; keep as floats (MOT standard allows floats)
        rows.append(f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
        track_ids.add(track_id)
        frame_ids.add(frame_idx)

    gt_path = os.path.join(gt_dir, "gt.txt")
    with open(gt_path, "w") as f:
        f.writelines(rows)

    n_frames = max(frame_ids) if frame_ids else 0

    # Link / copy images if source directory exists
    images_available = False
    if img_src_dir and os.path.isdir(img_src_dir):
        images_available = True
        for fname in sorted(os.listdir(img_src_dir)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            src = os.path.join(img_src_dir, fname)
            dst = os.path.join(img1_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Write seqinfo.ini
    ini = configparser.ConfigParser()
    ini["Sequence"] = {
        "name"      : seq_name,
        "imDir"     : "img1",
        "frameRate" : str(FRAME_RATE),
        "seqLength" : str(n_frames),
        "imWidth"   : str(IMG_WIDTH),
        "imHeight"  : str(IMG_HEIGHT),
        "imExt"     : ".jpg",
    }
    ini_path = os.path.join(out_seq_dir, "seqinfo.ini")
    with open(ini_path, "w") as f:
        ini.write(f)

    return {
        "seq_name"        : seq_name,
        "n_frames"        : n_frames,
        "n_tracks"        : len(track_ids),
        "n_boxes"         : len(rows),
        "images_available": images_available,
    }


# ── Verification frame renderer ───────────────────────────────────────────────

def render_sample_frame(seq_dir, out_path, frame_idx=1):
    """
    Render one frame with GT boxes overlaid.

    If real images exist in img1/, draws on the real frame.
    Otherwise draws on a blank grey canvas (960×540).
    Saves to out_path as a JPEG.
    """
    gt_path  = os.path.join(seq_dir, "gt", "gt.txt")
    img1_dir = os.path.join(seq_dir, "img1")

    # Load or create canvas
    real_imgs = sorted(f for f in os.listdir(img1_dir)
                       if f.lower().endswith(".jpg")) if os.path.isdir(img1_dir) else []
    if real_imgs:
        img_path = os.path.join(img1_dir, real_imgs[frame_idx - 1])
        canvas   = cv2.imread(img_path)
        if canvas is None:
            canvas = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 128, dtype=np.uint8)
        label = f"Real frame {frame_idx}"
    else:
        canvas = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 128, dtype=np.uint8)
        label  = f"Blank canvas (no images) — frame {frame_idx}"

    # Draw GT boxes for this frame
    n_boxes = 0
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if int(parts[0]) != frame_idx:
                continue
            tid = int(parts[1])
            x, y, w, h = [float(v) for v in parts[2:6]]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # Colour per track (deterministic)
            np.random.seed(tid * 2654435761 & 0xFFFFFF)
            colour = tuple(int(c) for c in np.random.randint(80, 255, 3))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(canvas, f"ID{tid}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            n_boxes += 1

    # HUD
    cv2.putText(canvas, label, (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"GT boxes: {n_boxes}", (8, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    seq_name = os.path.basename(seq_dir)
    cv2.putText(canvas, f"Seq: {seq_name}", (8, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Verification frame saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def find_xml_dir(raw_root, subdir_candidates):
    """Return the first existing candidate path under raw_root."""
    for cand in subdir_candidates:
        p = os.path.join(raw_root, cand)
        if os.path.isdir(p):
            return p
    return None


def main(detrac_root, out_root, verify_dir):
    print("=" * 70)
    print("UA-DETRAC -> MOT Challenge converter")
    print(f"  Source  : {detrac_root}")
    print(f"  Output  : {out_root}")
    print("=" * 70)

    # Locate annotation and image directories (handle one-level nesting)
    train_xml_dir = find_xml_dir(detrac_root, [
        os.path.join("DETRAC-Train-Annotations-XML", "DETRAC-Train-Annotations-XML"),
        "DETRAC-Train-Annotations-XML",
    ])
    test_xml_dir = find_xml_dir(detrac_root, [
        os.path.join("DETRAC-Test-Annotations-XML", "DETRAC-Test-Annotations-XML"),
        "DETRAC-Test-Annotations-XML",
    ])
    img_root = find_xml_dir(detrac_root, [
        os.path.join("DETRAC-Images", "DETRAC-Images"),
        "DETRAC-Images",
    ])

    if train_xml_dir is None:
        print("FATAL: Could not find DETRAC-Train-Annotations-XML under", detrac_root)
        sys.exit(1)
    if test_xml_dir is None:
        print("FATAL: Could not find DETRAC-Test-Annotations-XML under", detrac_root)
        sys.exit(1)

    print(f"  Train XML dir : {train_xml_dir}")
    print(f"  Test  XML dir : {test_xml_dir}")
    print(f"  Images dir    : {img_root or '(NOT FOUND — blank canvas mode)'}\n")

    # Track overall stats
    all_stats  = []
    first_seq_dir = None  # for verification frame

    for split, xml_dir in [("train", train_xml_dir), ("test", test_xml_dir)]:
        xml_files = sorted(f for f in os.listdir(xml_dir) if f.endswith(".xml"))
        print(f"--- {split.upper()} ({len(xml_files)} sequences) ---")

        for xml_fname in xml_files:
            xml_path = os.path.join(xml_dir, xml_fname)
            seq_name = os.path.splitext(xml_fname)[0]

            # Find matching image directory (if images present)
            img_seq_dir = os.path.join(img_root, seq_name) if img_root else None

            out_seq_dir = os.path.join(out_root, split, seq_name)
            stats = convert_sequence(xml_path, img_seq_dir, out_seq_dir)
            stats["split"] = split
            all_stats.append(stats)

            if first_seq_dir is None and split == "train":
                first_seq_dir = out_seq_dir

            print(f"  {seq_name:16s}  frames={stats['n_frames']:5d}  "
                  f"tracks={stats['n_tracks']:3d}  boxes={stats['n_boxes']:7d}  "
                  f"imgs={'yes' if stats['images_available'] else 'no'}")

    # Summary
    train_stats = [s for s in all_stats if s["split"] == "train"]
    test_stats  = [s for s in all_stats if s["split"] == "test"]

    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print(f"  Train sequences : {len(train_stats)}")
    print(f"  Test  sequences : {len(test_stats)}")
    print(f"  Total sequences : {len(all_stats)}")
    print(f"  Total frames    : {sum(s['n_frames'] for s in all_stats):,}")
    print(f"  Total boxes     : {sum(s['n_boxes']  for s in all_stats):,}")
    print(f"  Total tracks    : {sum(s['n_tracks'] for s in all_stats):,}")
    print("=" * 70)

    # Render verification frame from first train sequence
    if first_seq_dir:
        verify_path = os.path.join(verify_dir, "detrac_sample.jpg")
        render_sample_frame(first_seq_dir, verify_path, frame_idx=1)
    else:
        print("  WARNING: No sequence converted — no verification frame rendered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC to MOT format")
    parser.add_argument("--detrac-root", default=DEFAULT_DETRAC_ROOT,
                        help="Root of raw UA-DETRAC data")
    parser.add_argument("--out-root",    default=DEFAULT_OUT_ROOT,
                        help="Output root (will write train/ and test/ here)")
    parser.add_argument("--verify-dir",  default=DEFAULT_VERIFY_DIR,
                        help="Directory for verification artifacts")
    args = parser.parse_args()

    main(args.detrac_root, args.out_root, args.verify_dir)
