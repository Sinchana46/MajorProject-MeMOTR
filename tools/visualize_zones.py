"""
tools/visualize_zones.py — Draw perspective zone boundaries on sample frames (Phase 4)

PURPOSE
-------
Produce a figure showing the three perspective zones (FAR / MID / NEAR) overlaid
on a sample IOTD frame, along with all annotated bounding boxes colour-coded by zone.
This figure is used directly in the paper (Section 4, Contribution 1 figure).

Also used in Phase 2 as a lightweight verification tool: render boxes from a
converted gt.txt onto the corresponding image to visually confirm that the
convert_detrac.py and convert_visdrone.py coordinate mappings are correct.

TWO MODES OF OPERATION
-----------------------

Mode 1 — Zone visualization (main use, Phase 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Draws two horizontal lines on a frame at:
      y = BOUNDARY_FAR_MID  × frame_height   (top of MID zone)
      y = BOUNDARY_MID_NEAR × frame_height   (top of NEAR zone)
  and colours bounding boxes by zone:
      FAR  zone → blue   (small, distant)
      MID  zone → yellow
      NEAR zone → green  (large, close)
  Labels each box with its track ID.
  Outputs a PNG suitable for inclusion as a paper figure.

Mode 2 — Annotation sanity-check (Phase 2 verification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Given a gt.txt and a frame number, renders all boxes for that frame
  onto the image and saves a PNG.  No zone lines drawn.  Used to confirm
  convert_detrac.py / convert_visdrone.py produce correctly-placed boxes.

ZONE BOUNDARIES used at call time
----------------------------------
  Read from the calibration JSON produced by calibrate_perspective.py
  (Phase 4, Task 4.2).  If the JSON does not exist yet, fall back to the
  placeholder constants from modules/position_decay.py (0.30 / 0.70).

USAGE
------
  # Mode 1 — zone visualization (Phase 4)
  conda activate memotr_project
  python tools/visualize_zones.py \\
      --mode zones \\
      --image_path  data/IOTD/train/seq001/img1/000001.jpg \\
      --gt_path     data/IOTD/train/seq001/gt/gt.txt \\
      --frame_id    1 \\
      --calib_json  tools/perspective_calibration_result.json \\
      --output_png  results/figures/zone_visualization.png

  # Mode 2 — annotation sanity-check (Phase 2 verification)
  python tools/visualize_zones.py \\
      --mode boxes \\
      --image_path  data/UA_DETRAC/train/MVI_20011/img1/000001.jpg \\
      --gt_path     data/UA_DETRAC/train/MVI_20011/gt/gt.txt \\
      --frame_id    1 \\
      --output_png  results/figures/detrac_sanity_check.png

RELATED FILES
-------------
  tools/calibrate_perspective.py  — produces the calibration JSON this reads
  modules/position_decay.py       — shares the BOUNDARY_FAR_MID / BOUNDARY_MID_NEAR constants
  data/IOTD/train/*/gt/gt.txt     — input MOT-format ground truth
"""

import argparse
import sys
from pathlib import Path


def visualize_zones(
    image_path: Path,
    gt_path: Path,
    frame_id: int,
    boundary_far_mid: float,
    boundary_mid_near: float,
    output_png: Path,
) -> None:
    """
    Mode 1: Draw horizontal zone boundaries and colour-coded boxes on a frame.

    Parameters
    ----------
    image_path        : Path to the source image file.
    gt_path           : Path to the MOT-format gt.txt file.
    frame_id          : Which frame to visualize (1-indexed, matches gt.txt).
    boundary_far_mid  : Normalized y boundary between FAR and MID zones.
    boundary_mid_near : Normalized y boundary between MID and NEAR zones.
    output_png        : Where to save the output PNG.
    """
    raise NotImplementedError(
        "visualize_zones() is not yet implemented. "
        "Implement in Phase 4 (Task 4.2) of the project execution plan."
    )


def visualize_boxes(
    image_path: Path,
    gt_path: Path,
    frame_id: int,
    output_png: Path,
) -> None:
    """
    Mode 2: Draw raw annotation boxes on a frame (no zone colouring).
    Used in Phase 2 as a dataset-conversion sanity check.

    Parameters
    ----------
    image_path : Path to the source image file.
    gt_path    : Path to the MOT-format gt.txt file.
    frame_id   : Which frame to visualize (1-indexed).
    output_png : Where to save the output PNG.
    """
    raise NotImplementedError(
        "visualize_boxes() is not yet implemented. "
        "Implement in Phase 2 (Task 2.1 verification) or Phase 4 (Task 4.2)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize perspective zones or annotation boxes on a sample frame."
    )
    parser.add_argument("--mode",      type=str, choices=["zones", "boxes"],
                        required=True, help="'zones' = zone visualization; 'boxes' = annotation check.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--gt_path",    type=str, required=True)
    parser.add_argument("--frame_id",   type=int, default=1)
    parser.add_argument("--calib_json", type=str, default="tools/perspective_calibration_result.json",
                        help="Calibration JSON from calibrate_perspective.py. "
                             "Used in 'zones' mode only.")
    parser.add_argument("--output_png", type=str, default="results/figures/visualization.png")
    args = parser.parse_args()

    if args.mode == "zones":
        visualize_zones(
            image_path=Path(args.image_path),
            gt_path=Path(args.gt_path),
            frame_id=args.frame_id,
            boundary_far_mid=0.30,   # placeholder — will be read from calib_json once implemented
            boundary_mid_near=0.70,
            output_png=Path(args.output_png),
        )
    else:
        visualize_boxes(
            image_path=Path(args.image_path),
            gt_path=Path(args.gt_path),
            frame_id=args.frame_id,
            output_png=Path(args.output_png),
        )
