"""
tools/run_trackeval.py — Phase 5/6 TrackEval wrapper

Generates a seqmap file dynamically from our sequence directories,
runs HOTA + CLEAR + Identity via TrackEval's run_mot_challenge.py,
and copies the raw pedestrian_summary.txt to a results directory.

Usage (from project root):
  python tools/run_trackeval.py ^
      --tracker-dir   results/phase5_interim_baseline/tracker_val ^
      --gt-dirs       data/UA_DETRAC/test data/VisDrone_Vehicles/val ^
      --output-dir    results/phase5_interim_baseline ^
      --label         INTERIM_BASELINE

Raw TrackEval output (pedestrian_summary.txt) is preserved exactly
as written by TrackEval — not reformatted.
"""

from __future__ import annotations
import argparse
import configparser
import os
import sys
import shutil
import re

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TRACKEVAL_SCRIPT = os.path.join(PROJECT_DIR, "TrackEval", "scripts", "run_mot_challenge.py")
PYTHON = sys.executable


def build_combined_gt(gt_dirs: list[str], combined_gt_dir: str,
                      tracker_dir: str) -> list[str]:
    """
    Build a single combined GT directory in MOT Challenge layout:
        combined_gt_dir/<seq>/gt/gt.txt
        combined_gt_dir/<seq>/seqinfo.ini

    Returns list of sequence names that have both GT and tracker output.
    """
    os.makedirs(combined_gt_dir, exist_ok=True)
    seqs = []
    for gt_dir in gt_dirs:
        if not os.path.isdir(gt_dir):
            print(f"  WARNING: GT dir not found: {gt_dir}")
            continue
        for seq in sorted(os.listdir(gt_dir)):
            seq_path = os.path.join(gt_dir, seq)
            gt_txt   = os.path.join(seq_path, "gt", "gt.txt")
            tracker_txt = os.path.join(tracker_dir, f"{seq}.txt")
            if not os.path.isfile(gt_txt):
                continue
            if not os.path.isfile(tracker_txt):
                print(f"  WARNING: no tracker output for {seq}, skipping")
                continue
            # Link/copy GT into combined layout
            dst_seq = os.path.join(combined_gt_dir, seq)
            dst_gt  = os.path.join(dst_seq, "gt")
            os.makedirs(dst_gt, exist_ok=True)
            dst_gt_txt = os.path.join(dst_gt, "gt.txt")
            if not os.path.exists(dst_gt_txt):
                shutil.copy2(gt_txt, dst_gt_txt)
            # Copy seqinfo.ini if present
            src_seqinfo = os.path.join(seq_path, "seqinfo.ini")
            dst_seqinfo = os.path.join(dst_seq, "seqinfo.ini")
            if os.path.exists(src_seqinfo) and not os.path.exists(dst_seqinfo):
                shutil.copy2(src_seqinfo, dst_seqinfo)
            seqs.append(seq)
    return seqs


def write_seqmap(seqmap_path: str, seqs: list[str]):
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        for s in sorted(seqs):
            f.write(s + "\n")


def parse_args():
    p = argparse.ArgumentParser("TrackEval wrapper for VehicleMOT evaluation")
    p.add_argument("--tracker-dir",  required=True, help="Dir with per-seq tracker .txt files")
    p.add_argument("--gt-dirs",      nargs="+",     required=True,
                   help="Ground-truth split directories (MOT layout)")
    p.add_argument("--output-dir",   required=True,
                   help="Where to write seqmap, combined_gt, and summary copy")
    p.add_argument("--label",        default="eval",
                   help="Label for this evaluation run (used in summary filename)")
    p.add_argument("--num-cores",    type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== TrackEval Wrapper for VehicleMOT ===")
    print(f"Tracker dir : {args.tracker_dir}")
    print(f"GT dirs     : {args.gt_dirs}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Label       : {args.label}")

    # Build combined GT layout in output_dir/combined_gt/
    combined_gt_dir = os.path.join(args.output_dir, "combined_gt")
    print("\nBuilding combined GT directory...")
    seqs = build_combined_gt(
        gt_dirs        = args.gt_dirs,
        combined_gt_dir = combined_gt_dir,
        tracker_dir    = args.tracker_dir,
    )
    print(f"  {len(seqs)} sequences with matched GT + tracker output")

    if len(seqs) == 0:
        print("ERROR: No matching sequences found. Aborting.")
        sys.exit(1)

    # Fix seqlength mismatches: ensure seqinfo.ini seqlength >= max tracker frame
    fixed = 0
    for seq in seqs:
        ini_path     = os.path.join(combined_gt_dir, seq, "seqinfo.ini")
        tracker_path = os.path.join(args.tracker_dir, f"{seq}.txt")
        if not os.path.isfile(ini_path) or not os.path.isfile(tracker_path):
            continue
        max_frame = 0
        with open(tracker_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if parts and parts[0].isdigit():
                    max_frame = max(max_frame, int(parts[0]))
        if max_frame == 0:
            continue
        cp = configparser.ConfigParser()
        cp.read(ini_path)
        try:
            cur_len = int(cp["Sequence"].get("seqlength", "0"))
        except (KeyError, ValueError):
            cur_len = 0
        if cur_len < max_frame:
            with open(ini_path, "r") as f:
                content = f.read()
            content = re.sub(r"(?i)seqlength\s*=\s*\d+", f"seqLength = {max_frame}", content)
            with open(ini_path, "w") as f:
                f.write(content)
            fixed += 1
    if fixed:
        print(f"  Auto-fixed seqlength in {fixed} seqinfo.ini file(s)")

    # Write seqmap
    seqmap_path = os.path.join(args.output_dir, "seqmap.txt")
    write_seqmap(seqmap_path, seqs)
    print(f"  Seqmap written: {seqmap_path}")

    # Create a tracker sub-dir layout that TrackEval expects:
    # TRACKERS_FOLDER/<tracker_name>/<seq>.txt
    tracker_name = args.label
    trackers_root = os.path.join(args.output_dir, "trackeval_trackers")
    tracker_subdir = os.path.join(trackers_root, tracker_name)
    os.makedirs(tracker_subdir, exist_ok=True)
    for seq in seqs:
        src = os.path.join(args.tracker_dir, f"{seq}.txt")
        dst = os.path.join(tracker_subdir, f"{seq}.txt")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Run TrackEval
    te_output_dir = os.path.join(args.output_dir, "trackeval_output")
    os.makedirs(te_output_dir, exist_ok=True)

    # Call TrackEval directly via its Python API instead of subprocess
    # (avoids argparse nargs='+' wrapping SEQMAP_FILE in a list)
    import importlib.util
    te_dir = os.path.join(PROJECT_DIR, "TrackEval")
    sys.path.insert(0, te_dir)
    import trackeval  # noqa: E402

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"]          = False
    eval_config["PRINT_RESULTS"]         = True
    eval_config["PRINT_ONLY_COMBINED"]   = False
    eval_config["PLOT_CURVES"]           = False
    eval_config["OUTPUT_SUMMARY"]        = True
    eval_config["OUTPUT_DETAILED"]       = True
    eval_config["TIME_PROGRESS"]         = True
    eval_config["BREAK_ON_ERROR"]        = False  # Log bad seqs, continue rest
    eval_config["RETURN_ON_ERROR"]       = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"]          = combined_gt_dir
    dataset_config["TRACKERS_FOLDER"]    = trackers_root
    dataset_config["OUTPUT_FOLDER"]      = te_output_dir
    dataset_config["TRACKERS_TO_EVAL"]   = [tracker_name]
    dataset_config["CLASSES_TO_EVAL"]    = ["pedestrian"]
    dataset_config["SPLIT_TO_EVAL"]      = "val"
    dataset_config["SEQMAP_FILE"]        = seqmap_path       # plain string — no argparse wrapping
    dataset_config["SKIP_SPLIT_FOL"]     = True
    dataset_config["TRACKER_SUB_FOLDER"] = ""
    dataset_config["OUTPUT_SUB_FOLDER"]  = ""
    dataset_config["DO_PREPROC"]         = False             # GT already uses class 1 slot
    dataset_config["PRINT_CONFIG"]       = True

    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

    print("\nRunning TrackEval (direct Python API)...\n")
    evaluator    = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric_cls in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric_cls.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric_cls(metrics_config))
    evaluator.evaluate(dataset_list, metrics_list)
    print("\nTrackEval evaluation complete.")

    # Find and copy summary file
    summary_src = os.path.join(te_output_dir, tracker_name, "pedestrian_summary.txt")
    if not os.path.exists(summary_src):
        # TrackEval sometimes puts it one level up
        alt = os.path.join(te_output_dir, "pedestrian_summary.txt")
        if os.path.exists(alt):
            summary_src = alt

    summary_dst = os.path.join(args.output_dir, f"{args.label}_pedestrian_summary.txt")
    if os.path.exists(summary_src):
        shutil.copy2(summary_src, summary_dst)
        print(f"\nRaw TrackEval summary copied to: {summary_dst}")
        print("\n=== SUMMARY CONTENT ===")
        with open(summary_dst) as f:
            print(f.read())
    else:
        print(f"\nWARNING: Could not find pedestrian_summary.txt. Searched:\n  {summary_src}")
        print("  Check output dir:", te_output_dir)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
