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

INPUT (raw UA-DETRAC structure expected by this script)
-------------------------------------------------------
  <detrac_root>/
    DETRAC-Train-Annotations-XML/
      MVI_20011.xml, MVI_20012.xml, ...   (60 sequences)
    DETRAC-Test-Annotations-XML/
      MVI_20032.xml, MVI_20051.xml, ...   (40 sequences)
    Insight-MOT_DETRAC-train-data/
      MVI_20011/img1/000001.jpg ...       (images for train sequences)
    Insight-MOT_DETRAC-test-data/
      MVI_20032/img1/000001.jpg ...       (images for test sequences)

OUTPUT (MOT Challenge format, written under data/UA_DETRAC/)
------------------------------------------------------------
  data/UA_DETRAC/
    train/
      MVI_20011/
        img1/   (symlink or copy — see --copy_images flag)
        gt/gt.txt
        seqinfo.ini
      ...
    test/
      MVI_20032/
        img1/ ...
        gt/gt.txt
        seqinfo.ini
      ...

MOT Challenge gt.txt column schema (1-indexed)
----------------------------------------------
  frame_id, track_id, x, y, w, h, conf, class_id, visibility
  conf  = 1 (all ground-truth boxes are considered valid)
  class_id: car=1, van=2, bus=3, other=4  (remapped from UA-DETRAC names)
  visibility: from UA-DETRAC <attribute> occlusion field if available, else 1.0

USAGE (Phase 2, Task 2.1 — run after downloading UA-DETRAC)
-------------------------------------------------------------
    conda activate memotr_project
    python tools/convert_detrac.py \\
        --detrac_root  /path/to/raw/UA-DETRAC \\
        --output_root  data/UA_DETRAC \\
        --copy_images          # or --symlink_images for faster operation

VERIFICATION STEP (from Task 2.1 verification artifact)
--------------------------------------------------------
After running, use tools/visualize_zones.py (or a simple cv2 script) to draw
boxes from a converted gt.txt onto the corresponding image — this visually
confirms the coordinate mapping is correct, not just that the file was created.
"""

import argparse
import sys
from pathlib import Path


def convert_sequence(xml_path: Path, img_dir: Path, output_seq_dir: Path, copy_images: bool) -> dict:
    """
    Convert one UA-DETRAC XML sequence to MOT Challenge format.

    Parameters
    ----------
    xml_path       : Path to the sequence XML file.
    img_dir        : Path to the sequence's img1/ folder in the raw download.
    output_seq_dir : Path to the output sequence folder under data/UA_DETRAC/.
    copy_images    : If True, copy images; if False, create symlinks.

    Returns
    -------
    stats : dict  {"sequence": str, "frames": int, "tracks": int}
    """
    raise NotImplementedError(
        "convert_sequence() is not yet implemented. "
        "Implement in Phase 2 (Task 2.1) of the project execution plan."
    )


def main(detrac_root: str, output_root: str, copy_images: bool) -> None:
    """
    Entry point — iterates over all UA-DETRAC sequences and converts them.

    Parameters
    ----------
    detrac_root : str  Root directory of the raw UA-DETRAC download.
    output_root : str  Root output directory (data/UA_DETRAC/).
    copy_images : bool If True, copy images into output tree; else symlink.
    """
    raise NotImplementedError(
        "convert_detrac.main() is not yet implemented. "
        "Implement in Phase 2 (Task 2.1) of the project execution plan."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert UA-DETRAC XML annotations to MOT Challenge format."
    )
    parser.add_argument("--detrac_root",   type=str, required=True,
                        help="Path to raw UA-DETRAC download root.")
    parser.add_argument("--output_root",   type=str, default="data/UA_DETRAC",
                        help="Output root (default: data/UA_DETRAC).")
    parser.add_argument("--copy_images",   action="store_true",
                        help="Copy images into output tree (default: symlink).")
    args = parser.parse_args()
    main(args.detrac_root, args.output_root, args.copy_images)
