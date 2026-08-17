"""
tools/calibrate_perspective.py — One-time perspective calibration script (Phase 4)

PURPOSE
-------
Derive the FAR / MID / NEAR zone boundaries for Contribution 1 from REAL
annotated IOTD data (not assumed top-30%/mid-40%/bottom-30% placeholders).

The calibration regresses bounding-box HEIGHT against normalized vertical
POSITION across all annotated IOTD training boxes.  In an oblique camera
geometry the relationship is strongly linear: far vehicles (top of frame)
are small; near vehicles (bottom) are large.  The regression slope and R²
characterize how strong the perspective gradient is.

ALGORITHM
---------
1. Load all MOT-format gt.txt files from data/IOTD/train/
2. For each annotation row: extract cy_norm (box vertical center / frame height)
   and h_norm (box height / frame height).
3. Fit a linear regression: h_norm ~ slope * cy_norm + intercept
4. Report slope and R² — expected R² > 0.5 for a genuine perspective gradient.
5. Split cy_norm range [0, 1] into three zones such that the EXPECTED h_norm
   at each zone boundary matches the physical size thresholds:
       FAR  zone : small  vehicles, frequent occlusion → use low  λ
       MID  zone : medium vehicles                     → use mid  λ
       NEAR zone : large  vehicles, reliable features  → use high λ
6. Print the calibrated boundaries and write them to a JSON config file that
   position_decay.py can load (replacing BOUNDARY_FAR_MID / BOUNDARY_MID_NEAR).

OUTPUTS
-------
  Console: slope, R², zone boundaries (fills Section 15 of memotr_project_prompt.md)
  File:    tools/perspective_calibration_result.json
              {
                "slope": <float>,
                "intercept": <float>,
                "r_squared": <float>,
                "boundary_far_mid": <float>,
                "boundary_mid_near": <float>,
                "source_sequences": <int>,
                "total_boxes": <int>
              }

USAGE (run once after first IOTD sequences are annotated — Phase 4, Task 4.2)
-------------------------------------------------------------------------------
    conda activate memotr_project
    python tools/calibrate_perspective.py \\
        --iotd_train_dir data/IOTD/train \\
        --output_json   tools/perspective_calibration_result.json

    After running, update BOUNDARY_FAR_MID and BOUNDARY_MID_NEAR in
    modules/position_decay.py with the values from the output JSON.

DEPENDENCY (Phase 4 gate)
--------------------------
Do NOT run this script until at least the first 3 IOTD sequences are
annotated and exported from CVAT in MOT Challenge format.  Running on
fewer sequences will give unreliable regression statistics.

RELATED FILES
-------------
  modules/position_decay.py         — consumes the calibrated boundaries
  tools/visualize_zones.py          — draws zone boundaries on sample frames
  data/IOTD/train/*/gt/gt.txt       — input annotation files
"""

import argparse
import json
import sys
from pathlib import Path


def main(iotd_train_dir: str, output_json: str) -> None:
    """
    Entry point for the perspective calibration script.

    Parameters
    ----------
    iotd_train_dir : str
        Path to data/IOTD/train/ — must contain at least one sequence folder
        with gt/gt.txt in MOT Challenge format.
    output_json : str
        Path where the calibration result JSON will be written.
    """
    raise NotImplementedError(
        "calibrate_perspective.main() is not yet implemented. "
        "Implement in Phase 4 (Task 4.2) of the project execution plan.\n"
        "Prerequisite: at least 3 IOTD sequences annotated and in data/IOTD/train/."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time perspective zone calibration from IOTD annotations."
    )
    parser.add_argument(
        "--iotd_train_dir",
        type=str,
        default="data/IOTD/train",
        help="Path to the IOTD training split directory.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="tools/perspective_calibration_result.json",
        help="Where to write the calibration output JSON.",
    )
    args = parser.parse_args()
    main(args.iotd_train_dir, args.output_json)
