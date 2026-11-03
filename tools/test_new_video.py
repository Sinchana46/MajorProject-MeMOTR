"""
tools/test_new_video.py
=======================
All-in-one pipeline: video -> frames -> seqinfo -> inference -> MOT output

Usage (run from project root D:/MajorProject-MeMOTR):
    python -u tools/test_new_video.py

    # Or specify a particular video:
    python -u tools/test_new_video.py --video data/IOTD_manual_test/2_Take-1.mp4

    # Or a whole directory of videos:
    python -u tools/test_new_video.py --video-dir data/IOTD_manual_test

Settings used (from results_after_tuning/final_review/selected_settings.csv):
  - Checkpoint : results/train_iotd_real/checkpoint_3.pth  (Arm1, epoch 3)
  - det_thresh : 0.5
  - track_thresh: 0.5
  - miss_tol   : 30
  - result_score_thresh: 0.5  (output filter, fixed by upstream config)
  - area_thresh: 100 px       (output filter, fixed by upstream code)
  - Config     : configs/train_iotd_real.yaml

Output layout:
  results/manual_test/<seq_name>/
    img1/               <- extracted JPEG frames
    seqinfo.ini         <- sequence metadata
    tracker/
      <seq_name>.txt    <- MOT-format tracker output
    summary.txt         <- quick summary (frame count, track count, etc.)

DO NOT modify this script's settings without also updating
  results_after_tuning/final_review/selected_settings.csv
to maintain provenance.
"""

from __future__ import annotations

import argparse
import configparser
import datetime
import glob
import json
import os
import subprocess
import sys
import time

# ── Project paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MEMOTR_DIR  = os.path.join(PROJECT_DIR, "MeMOTR")
sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))
sys.path.insert(0, PROJECT_DIR)

# ── Final selected settings (do not change without updating provenance) ────────
CHECKPOINT    = os.path.join(PROJECT_DIR, "results", "train_iotd_real", "checkpoint_3.pth")
CONFIG_PATH   = os.path.join(PROJECT_DIR, "configs", "train_iotd_real.yaml")
DET_THRESH    = 0.5
TRACK_THRESH  = 0.5
MISS_TOL      = 30
RESULT_SCORE  = 0.5    # output score filter  (matches upstream submit_engine)
AREA_THRESH   = 100.0  # output area filter   (matches upstream submit_engine)

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_VIDEO_DIR = os.path.join(PROJECT_DIR, "data", "IOTD_manual_test")
OUTPUT_BASE       = os.path.join(PROJECT_DIR, "results", "manual_test")
FFMPEG            = "ffmpeg"   # must be on PATH
PYTHON            = sys.executable


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Find videos
# ══════════════════════════════════════════════════════════════════════════════

def find_videos(video_path: str | None, video_dir: str) -> list[str]:
    if video_path:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        return [video_path]
    exts = ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV",
            "*.mkv", "*.MKV", "*.m4v", "*.M4V")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(video_dir, ext)))
    if not videos:
        raise FileNotFoundError(f"No video files found in: {video_dir}")
    return sorted(videos)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Probe video metadata with ffprobe
# ══════════════════════════════════════════════════════════════════════════════

def probe_video(video_path: str) -> dict:
    """Return duration_s, fps (float), width, height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json",
        video_path
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(r.stdout)
        stream = info["streams"][0]
        width  = int(stream.get("width",  1920))
        height = int(stream.get("height", 1080))

        # Parse FPS  (e.g. "24000/1001" or "25/1")
        fps_str = stream.get("r_frame_rate", "25/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den)

        # Duration / frame count
        if "nb_frames" in stream and stream["nb_frames"] not in ("N/A", ""):
            n_frames = int(stream["nb_frames"])
            duration_s = n_frames / fps
        elif "duration" in stream and stream["duration"] not in ("N/A", ""):
            duration_s = float(stream["duration"])
            n_frames   = int(round(duration_s * fps))
        else:
            duration_s = 0.0
            n_frames   = 0

        return {"fps": fps, "width": width, "height": height,
                "duration_s": duration_s, "n_frames_approx": n_frames}
    except Exception as e:
        print(f"  [WARN] ffprobe failed ({e}); using defaults.")
        return {"fps": 25.0, "width": 1920, "height": 1080,
                "duration_s": 0.0, "n_frames_approx": 0}


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Extract frames with ffmpeg
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, img1_dir: str,
                   fps_out: float | None = None,
                   max_seconds: int = 0,
                   scale_width: int = 0) -> int:
    """
    Extract frames (optionally trimmed, downsampled in fps and/or resolution)
    into img1_dir as %06d.jpg.
    Returns actual frame count written.
    """
    os.makedirs(img1_dir, exist_ok=True)

    # Idempotent re-run
    existing = glob.glob(os.path.join(img1_dir, "*.jpg"))
    if existing:
        print(f"  Frames already extracted ({len(existing)} found). Skipping ffmpeg.")
        return len(existing)

    # Build vf filter: combine fps + optional scale
    filters = []
    if fps_out:
        filters.append(f"fps={fps_out}")
    else:
        filters.append("fps=source_fps")
    if scale_width > 0:
        # scale to scale_width wide; round height to nearest even
        filters.append(f"scale={scale_width}:-2")
    vf_filter = ",".join(filters)

    cmd = [FFMPEG, "-y", "-i", video_path]
    if max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
        print(f"  Trimming to first {max_seconds}s before extraction.")
    cmd += [
        "-vf", vf_filter,
        "-q:v", "2",
        "-start_number", "1",
        os.path.join(img1_dir, "%06d.jpg")
    ]
    print(f"  Running ffmpeg to extract frames...")
    print(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed:\n{result.stderr[-2000:]}")
        sys.exit(1)

    frames = glob.glob(os.path.join(img1_dir, "*.jpg"))
    count = len(frames)
    print(f"  Extracted {count} frames in {elapsed:.1f}s")
    return count


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Write seqinfo.ini
# ══════════════════════════════════════════════════════════════════════════════

def write_seqinfo(seq_dir: str, seq_name: str, fps: float,
                  n_frames: int, width: int, height: int):
    seqinfo_path = os.path.join(seq_dir, "seqinfo.ini")
    if os.path.exists(seqinfo_path):
        print(f"  seqinfo.ini already exists. Skipping.")
        return
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {
        "name":       seq_name,
        "imdir":      "img1",
        "framerate":  str(round(fps)),
        "seqlength":  str(n_frames),
        "imwidth":    str(width),
        "imheight":   str(height),
        "imext":      ".jpg",
    }
    with open(seqinfo_path, "w") as f:
        cfg.write(f)
    print(f"  seqinfo.ini written: fps={round(fps)}, frames={n_frames}, "
          f"size={width}x{height}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Run inference (via run_inference_mot_filtered.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(seq_dir: str, tracker_dir: str, seq_name: str) -> str | None:
    """
    Calls run_inference_mot_filtered.py on the prepared sequence.
    Returns path to the output .txt file, or None on failure.
    """
    os.makedirs(tracker_dir, exist_ok=True)
    out_txt = os.path.join(tracker_dir, f"{seq_name}.txt")

    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        print(f"  Tracker output already exists ({os.path.getsize(out_txt)//1024} KB). Skipping inference.")
        return out_txt

    # Use the parent of seq_dir as the split-dir (run_inference_mot_filtered
    # expects a split directory containing sequence subdirectories)
    split_dir = os.path.dirname(seq_dir)

    infer_script = os.path.join(PROJECT_DIR, "tools", "run_inference_mot_filtered.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        MEMOTR_DIR + os.pathsep +
        os.path.join(MEMOTR_DIR, "models", "ops") + os.pathsep +
        PROJECT_DIR
    )
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        PYTHON, "-u", infer_script,
        "--checkpoint",         CHECKPOINT,
        "--split-dirs",         split_dir,
        "--output-dir",         tracker_dir,
        "--config",             CONFIG_PATH,
        "--det-thresh",         str(DET_THRESH),
        "--track-thresh",       str(TRACK_THRESH),
        "--miss-tol",           str(MISS_TOL),
        "--result-score-thresh",str(RESULT_SCORE),
        "--area-thresh",        str(AREA_THRESH),
        "--label",              f"manual_test_{seq_name}",
    ]
    print(f"  Running inference...")
    print(f"  CMD: {' '.join(cmd)}")
    log_path = os.path.join(tracker_dir, "inference.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        result = subprocess.run(
            cmd, cwd=PROJECT_DIR, env=env,
            stdout=log_f, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] Inference failed (exit {result.returncode}). "
              f"See {log_path}")
        # Print last 40 lines of log for quick diagnosis
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        print("  Last 40 log lines:")
        for l in lines[-40:]:
            print("   ", l.rstrip())
        return None

    print(f"  Inference done in {elapsed:.1f}s")

    # run_inference_mot_filtered.py names output after the sequence sub-folder
    # ("seq"), not the video name. Find the largest .txt that isn't label.txt.
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        return out_txt
    txts = [f for f in glob.glob(os.path.join(tracker_dir, "*.txt"))
            if os.path.basename(f) != "label.txt"]
    if txts:
        txts.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return txts[0]
    print(f"  [WARN] No tracker .txt found in {tracker_dir}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Summarise output
# ══════════════════════════════════════════════════════════════════════════════

def summarise(tracker_txt: str, seq_name: str, meta: dict,
              seq_dir: str, out_dir: str):
    """Parse tracker output and write a human-readable summary."""
    if not tracker_txt or not os.path.exists(tracker_txt):
        print("  [WARN] No tracker output to summarise.")
        return

    frames_with_tracks = set()
    track_ids          = set()
    total_dets         = 0

    with open(tracker_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                frames_with_tracks.add(frame_id)
                track_ids.add(track_id)
                total_dets += 1
            except ValueError:
                continue

    n_frames_tracked = len(frames_with_tracks)
    n_tracks         = len(track_ids)
    avg_dets_per_frame = total_dets / max(n_frames_tracked, 1)

    summary_lines = [
        "=" * 60,
        f"  MeMOTR Manual Test — {seq_name}",
        f"  Run at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "  Input video",
        f"    Path       : {os.path.join(DEFAULT_VIDEO_DIR, seq_name)}",
        f"    FPS        : {meta.get('fps', '?'):.2f}",
        f"    Resolution : {meta.get('width')}x{meta.get('height')}",
        f"    Duration   : {meta.get('duration_s', 0):.1f}s",
        f"    Frames ext : {meta.get('n_frames_actual', '?')}",
        "",
        "  Model settings (from selected_settings.csv)",
        f"    Checkpoint       : results/train_iotd_real/checkpoint_3.pth",
        f"    det_thresh       : {DET_THRESH}",
        f"    track_thresh     : {TRACK_THRESH}",
        f"    miss_tol         : {MISS_TOL}",
        f"    result_score     : {RESULT_SCORE}",
        f"    area_thresh      : {AREA_THRESH} px",
        "",
        "  Tracker output",
        f"    Output file          : {tracker_txt}",
        f"    Frames with tracks   : {n_frames_tracked}",
        f"    Unique track IDs     : {n_tracks}",
        f"    Total detections     : {total_dets}",
        f"    Avg dets/frame       : {avg_dets_per_frame:.2f}",
        "",
        "  Files written",
        f"    Frames    : {os.path.join(seq_dir, 'img1/')}",
        f"    seqinfo   : {os.path.join(seq_dir, 'seqinfo.ini')}",
        f"    Tracker   : {tracker_txt}",
        f"    Infer log : {os.path.join(out_dir, 'tracker', 'inference.log')}",
        f"    Summary   : {os.path.join(out_dir, 'summary.txt')}",
        "",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"  Summary saved to: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="All-in-one MeMOTR test pipeline for new videos."
    )
    p.add_argument("--video",     default=None,
                   help="Path to a single video file (mp4/avi/mov/mkv). "
                        "If omitted, all videos in --video-dir are processed.")
    p.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR,
                   help=f"Directory to scan for videos (default: {DEFAULT_VIDEO_DIR})")
    p.add_argument("--output-dir", default=OUTPUT_BASE,
                   help=f"Root output directory (default: {OUTPUT_BASE})")
    p.add_argument("--fps-out", type=float, default=None,
                   help="Downsample to this FPS before extracting frames. "
                        "Default: use native video FPS.")
    p.add_argument("--scale-width", type=int, default=0,
                   help="Scale frames to this width (px) before inference — "
                        "speeds up the CNN backbone significantly. "
                        "E.g. 640 reduces 2320-wide frames by ~3.6×. "
                        "0 = keep native resolution.")
    p.add_argument("--max-seconds", type=int, default=0,
                   help="Only process the first N seconds of the video (0 = full video). "
                        "Speeds up inference significantly for quick tests.")
    p.add_argument("--seq-name", default=None,
                   help="Override sequence name (default: video stem).")
    return p.parse_args()


def main():
    # Force UTF-8 stdout so Unicode characters don't crash on Windows cp1252 terminals
    import sys, io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    args = parse_args()

    # ── Pre-flight checks ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MeMOTR — New Video Test Pipeline")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
        print("  Make sure results/train_iotd_real/checkpoint_3.pth exists.")
        sys.exit(1)

    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Config not found: {CONFIG_PATH}")
        sys.exit(1)

    ckpt_mb = os.path.getsize(CHECKPOINT) / 1e6
    print(f"\n  Checkpoint : {CHECKPOINT}")
    print(f"             ({ckpt_mb:.0f} MB — Arm1, epoch 3, selected by tuning)")
    print(f"  Config     : {CONFIG_PATH}")
    print(f"  det/track thresh: {DET_THRESH}/{TRACK_THRESH}  miss_tol={MISS_TOL}")

    # ── Find videos ──────────────────────────────────────────────────────────
    videos = find_videos(args.video, args.video_dir)
    print(f"\n  Found {len(videos)} video(s) to process:")
    for v in videos:
        print(f"    {v}")

    # ── Process each video ───────────────────────────────────────────────────
    for video_path in videos:
        stem     = os.path.splitext(os.path.basename(video_path))[0]
        seq_name = args.seq_name or stem
        # Sanitise name (spaces → underscores)
        seq_name = seq_name.replace(" ", "_").replace("-", "_")

        out_dir     = os.path.join(args.output_dir, seq_name)
        seq_dir     = os.path.join(out_dir, "seq")
        img1_dir    = os.path.join(seq_dir, "img1")
        tracker_dir = os.path.join(out_dir, "tracker")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'-'*60}")
        print(f"  Processing: {os.path.basename(video_path)}")
        print(f"  Seq name  : {seq_name}")
        print(f"  Output dir: {out_dir}")
        print(f"{'-'*60}")

        # Step 2 — Probe
        print("\n[Step 1/4] Probing video metadata...")
        meta = probe_video(video_path)
        print(f"  FPS={meta['fps']:.3f}  Size={meta['width']}x{meta['height']}  "
              f"Duration={meta['duration_s']:.1f}s  "
              f"~{meta['n_frames_approx']} frames")

        # Effective duration (respect --max-seconds)
        if args.max_seconds > 0 and meta["duration_s"] > 0:
            eff_duration = min(args.max_seconds, meta["duration_s"])
            eff_frames   = int(round(eff_duration * (args.fps_out or meta["fps"])))
            print(f"  [TRIM] Processing only first {eff_duration:.1f}s "
                  f"(~{eff_frames} frames at {args.fps_out or meta['fps']:.1f} fps)")
        else:
            eff_duration = meta["duration_s"]
            eff_frames   = int(round(eff_duration * (args.fps_out or meta["fps"])))

        fps_extract = args.fps_out if args.fps_out else meta["fps"]

        # Step 3 — Extract frames
        print(f"\n[Step 2/4] Extracting frames (fps={fps_extract:.2f}"
              + (f", first {args.max_seconds}s" if args.max_seconds else "")
              + (f", scale ≤{args.scale_width}px" if args.scale_width else "")
              + ")...")
        n_frames_actual = extract_frames(video_path, img1_dir,
                                         fps_out=args.fps_out,
                                         max_seconds=args.max_seconds,
                                         scale_width=args.scale_width)
        meta["n_frames_actual"] = n_frames_actual
        meta["fps_extracted"]   = fps_extract

        # Step 4 — Write seqinfo
        print("\n[Step 3/4] Writing seqinfo.ini...")
        write_seqinfo(seq_dir, seq_name, fps_extract,
                      n_frames_actual, meta["width"], meta["height"])

        # Step 5 — Run inference
        print("\n[Step 4/4] Running MeMOTR inference...")
        tracker_txt = run_inference(seq_dir, tracker_dir, seq_name)

        # Step 6 — Summarise
        print("\n[Summary]")
        summarise(tracker_txt, seq_name, meta, seq_dir, out_dir)

    print("\n  All videos processed. Done.")


if __name__ == "__main__":
    main()
