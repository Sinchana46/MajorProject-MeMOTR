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
  - MUST download the MOT version (NOT the DET version — file structure differs)
  - Download: https://github.com/VisDrone/VisDrone-Dataset
  - Official train/val splits (use these — do NOT create custom frame-level splits)

VEHICLE CLASSES TO KEEP (Section 3.2)
--------------------------------------
  VisDrone class id → kept name
  4  → car
  5  → van
  6  → truck
  9  → bus
  ALL other class IDs are discarded (pedestrians, bicycles, awnings, etc.)

INPUT (raw VisDrone-MOT structure expected by this script)
----------------------------------------------------------
  <visdrone_root>/
    VisDrone2019-MOT-train/
      sequences/
        uav0000013_00000_v/   (each folder = one sequence)
          000001.jpg, 000002.jpg, ...
      annotations/
        uav0000013_00000_v.txt   (one annotation file per sequence)
    VisDrone2019-MOT-val/
      sequences/ ...
      annotations/ ...

  VisDrone annotation format (comma-separated, 1-indexed frame_id):
      frame_id, track_id, x, y, w, h, score, class_id, truncation, occlusion
      (class_id here is 1-indexed VisDrone class; we filter to {4,5,6,9})

OUTPUT (MOT Challenge format, written under data/VisDrone_Vehicles/)
--------------------------------------------------------------------
  data/VisDrone_Vehicles/
    train/
      uav0000013_00000_v/
        img1/   (symlink or copy)
        gt/gt.txt
        seqinfo.ini
      ...
    val/
      ...

  In the output gt.txt, class IDs are REMAPPED:
      car=1, van=2, truck=3, bus=4  (consistent with convert_detrac.py mapping)

VERIFICATION STEP (from Task 2.2 verification artifact)
--------------------------------------------------------
  1. Print class distribution BEFORE filtering (all VisDrone classes present).
  2. Print class distribution AFTER filtering (only {car, van, truck, bus}).
  3. Render one sample frame with boxes — confirm only vehicle classes appear.

USAGE (Phase 2, Task 2.2)
--------------------------
    conda activate memotr_project
    python tools/convert_visdrone.py \\
        --visdrone_root  /path/to/raw/VisDrone-MOT \\
        --output_root    data/VisDrone_Vehicles \\
        --copy_images

  IMPORTANT: Confirm before running that the source directory contains
  a "sequences/" and "annotations/" subfolder structure — this confirms
  you have the MOT version and NOT the DET version.
"""

import argparse
import sys
from pathlib import Path

# VisDrone class IDs to KEEP (all others are discarded)
VEHICLE_CLASS_IDS = {4, 5, 6, 9}

# Remapping to a unified class ID scheme shared with convert_detrac.py
VISDRONE_TO_UNIFIED_CLASS = {
    4: 1,   # car   → 1
    5: 2,   # van   → 2
    6: 3,   # truck → 3
    9: 4,   # bus   → 4
}


def convert_sequence(
    annotation_path: Path,
    img_dir: Path,
    output_seq_dir: Path,
    copy_images: bool,
) -> dict:
    """
    Convert one VisDrone-MOT sequence to MOT Challenge format, filtering
    to vehicle classes only.

    Parameters
    ----------
    annotation_path : Path  Path to the raw VisDrone annotation .txt file.
    img_dir         : Path  Path to the sequence's raw image folder.
    output_seq_dir  : Path  Path to the output sequence folder.
    copy_images     : bool  Copy images if True, symlink if False.

    Returns
    -------
    stats : dict
        {
          "sequence": str,
          "frames": int,
          "tracks_before_filter": int,
          "tracks_after_filter": int,
          "class_dist_before": dict,
          "class_dist_after": dict,
        }
    """
    raise NotImplementedError(
        "convert_sequence() is not yet implemented. "
        "Implement in Phase 2 (Task 2.2) of the project execution plan."
    )


def main(visdrone_root: str, output_root: str, copy_images: bool) -> None:
    """
    Entry point — iterates over all VisDrone-MOT sequences and converts them.

    Parameters
    ----------
    visdrone_root : str  Root directory of the raw VisDrone-MOT download.
    output_root   : str  Output root (data/VisDrone_Vehicles/).
    copy_images   : bool Copy images if True, symlink if False.
    """
    raise NotImplementedError(
        "convert_visdrone.main() is not yet implemented. "
        "Implement in Phase 2 (Task 2.2) of the project execution plan."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VisDrone-MOT annotations to MOT Challenge format, vehicle classes only."
    )
    parser.add_argument("--visdrone_root", type=str, required=True,
                        help="Path to raw VisDrone-MOT download root.")
    parser.add_argument("--output_root",   type=str, default="data/VisDrone_Vehicles",
                        help="Output root (default: data/VisDrone_Vehicles).")
    parser.add_argument("--copy_images",   action="store_true",
                        help="Copy images into output tree (default: symlink).")
    args = parser.parse_args()
    main(args.visdrone_root, args.output_root, args.copy_images)
