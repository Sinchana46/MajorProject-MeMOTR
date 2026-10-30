"""
tools/generate_test_report.py
==============================
Post-processing after test_new_video.py has run.

Given a tracker output directory, this module:
  1. Parses seq.txt  → stats dict
  2. Draws coloured bounding-box overlays on keyframes (every ~200 frames)
  3. Encodes an annotated H.264 mp4 via ffmpeg for browser playback
  4. Exports a CSV of every tracked detection

Called by app.py; can also be run standalone:
    python -u tools/generate_test_report.py --seq-dir results/manual_test/2_Take_1
"""
from __future__ import annotations
import argparse
import collections
import csv
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ── Colour palette (tab20-like, 20 distinguishable colours) ───────────────────
PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44),  (214, 39,  40),
    (148,103, 189), (140,  86,  75), (227,119, 194), (127,127, 127),
    (188,189,  34), (23,  190, 207), (174,199, 232), (255,187, 120),
    (152,223, 138), (255,152, 150), (197,176, 213), (196,156, 148),
    (247,182, 210), (199,199, 199), (219,219, 141), (158,218, 229),
]

def _colour(track_id: int):
    return PALETTE[int(track_id) % len(PALETTE)]


# ── Parse seq.txt ──────────────────────────────────────────────────────────────
def parse_tracker(seq_txt: str) -> dict:
    """
    Returns dict with keys:
      frame_data   : {frame_id: [(tid, x, y, w, h, score), ...]}
      stats        : {total_frames, frames_with_tracks, unique_ids,
                      total_detections, peak_count, peak_frame,
                      avg_dets_per_frame}
      all_rows     : list of raw (frame, tid, x, y, w, h, score) tuples
    """
    frame_data: dict[int, list] = collections.defaultdict(list)
    all_rows = []

    with open(seq_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            fid   = int(parts[0])
            tid   = int(parts[1])
            x, y  = float(parts[2]), float(parts[3])
            w, h  = float(parts[4]), float(parts[5])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            frame_data[fid].append((tid, x, y, w, h, score))
            all_rows.append((fid, tid, x, y, w, h, score))

    if not all_rows:
        return {"frame_data": {}, "stats": {}, "all_rows": []}

    all_frames   = sorted(frame_data.keys())
    total_frames = (max(all_frames) - min(all_frames) + 1) if all_frames else 0
    frames_with  = len(all_frames)
    unique_ids   = len({r[1] for r in all_rows})
    total_dets   = len(all_rows)
    peak_frame   = max(frame_data, key=lambda f: len(frame_data[f]))
    peak_count   = len(frame_data[peak_frame])
    avg_dets     = total_dets / frames_with if frames_with else 0

    stats = {
        "total_frames":       total_frames,
        "frames_with_tracks": frames_with,
        "unique_ids":         unique_ids,
        "total_detections":   total_dets,
        "peak_count":         peak_count,
        "peak_frame":         peak_frame,
        "avg_dets_per_frame": round(avg_dets, 2),
    }
    return {"frame_data": frame_data, "stats": stats, "all_rows": all_rows}


# ── Draw overlay on a single frame ────────────────────────────────────────────
def draw_frame(img_bgr: np.ndarray, tracks: list,
               show_header: bool = True) -> np.ndarray:
    out = img_bgr.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for tid, x, y, w, h, *_ in tracks:
        c = _colour(tid)
        bgr = (c[2], c[1], c[0])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 3)
        label = str(tid)
        fs = 0.7
        thick = 2
        (tw, th), bl = cv2.getTextSize(label, font, fs, thick)
        ly1 = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, ly1), (x1 + tw + 8, ly1 + th + 8), bgr, -1)
        cv2.putText(out, label, (x1 + 4, ly1 + th + 2),
                    font, fs, (255, 255, 255), thick, cv2.LINE_AA)

    if show_header:
        bar_h = 40
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        cv2.putText(out, "MeMOTR | unseen test video | Arm 1 checkpoint_3",
                    (10, 28), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Keyframe gallery ──────────────────────────────────────────────────────────
def make_keyframes(img1_dir: str, frame_data: dict,
                   step: int = 200) -> list[np.ndarray]:
    """Return list of annotated BGR images at every `step` frames."""
    all_frames = sorted(frame_data.keys())
    if not all_frames:
        return []
    chosen = [f for f in all_frames if f % step == 0 or f == all_frames[0]]
    if all_frames[-1] not in chosen:
        chosen.append(all_frames[-1])

    out_imgs = []
    for fid in chosen:
        img_path = os.path.join(img1_dir, f"{fid:06d}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        ann = draw_frame(img, frame_data.get(fid, []))
        out_imgs.append((fid, ann))
    return out_imgs


# ── Annotated video ────────────────────────────────────────────────────────────
MAX_VID_WIDTH = 960   # cap output video width for browser playback & speed

def make_annotated_video(img1_dir: str, frame_data: dict,
                         out_path: str, fps: float = 30.0,
                         progress_cb=None) -> str:
    """
    Draw boxes on every frame and encode as H.264 mp4 via ffmpeg pipe.
    Frames are downscaled to MAX_VID_WIDTH (aspect-preserving, even dims)
    for fast encoding and reliable browser playback.
    Returns out_path on success.

    IMPORTANT: stderr must be DEVNULL (not PIPE) to prevent deadlock:
    if ffmpeg fills its stderr pipe buffer while Python is blocked
    writing to stdin, both sides hang and the moov atom is never written.
    """
    files = sorted(Path(img1_dir).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No frames in {img1_dir}")

    # Read one frame for dimensions
    sample = cv2.imread(str(files[0]))
    if sample is None:
        raise RuntimeError(f"Cannot read sample frame: {files[0]}")
    H, W = sample.shape[:2]

    # Compute output dimensions (scale down if wider than MAX_VID_WIDTH)
    if W > MAX_VID_WIDTH:
        scale  = MAX_VID_WIDTH / W
        out_W  = MAX_VID_WIDTH
        out_H  = int(H * scale)
    else:
        out_W, out_H = W, H
    # libx264 requires even dimensions
    out_W = (out_W // 2) * 2
    out_H = (out_H // 2) * 2
    need_resize = (out_W != W or out_H != H)

    cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-s",       f"{out_W}x{out_H}",
        "-pix_fmt", "bgr24",
        "-r",       str(fps),
        "-i",       "pipe:0",
        "-vcodec",  "libx264",
        "-preset",  "fast",
        "-crf",     "23",
        "-pix_fmt", "yuv420p",
        "-movflags","+faststart",
        out_path,
    ]
    # stderr=DEVNULL is critical — avoids deadlock when ffmpeg writes
    # large amounts to stderr while we are blocked on stdin.write()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    total = len(files)
    blank = np.zeros((out_H, out_W, 3), dtype=np.uint8)
    try:
        for i, fp in enumerate(files):
            fid = int(fp.stem)
            img = cv2.imread(str(fp))
            if img is None:
                img = blank.copy()
            ann = draw_frame(img, frame_data.get(fid, []))
            if need_resize:
                ann = cv2.resize(ann, (out_W, out_H))
            proc.stdin.write(ann.tobytes())
            if progress_cb and i % 30 == 0:
                progress_cb(i / total)
    finally:
        proc.stdin.close()

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret} — video may be corrupt")
    if progress_cb:
        progress_cb(1.0)
    return out_path


# ── CSV export ─────────────────────────────────────────────────────────────────
def make_csv_bytes(all_rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["frame", "track_id", "x", "y", "w", "h", "score"])
    for row in all_rows:
        w.writerow([row[0], row[1],
                    f"{row[2]:.2f}", f"{row[3]:.2f}",
                    f"{row[4]:.2f}", f"{row[5]:.2f}",
                    f"{row[6]:.4f}"])
    return buf.getvalue().encode()


# ── Standalone CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True,
                    help="e.g. results/manual_test/2_Take_1")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    seq_dir  = args.seq_dir
    img1_dir = os.path.join(seq_dir, "seq", "img1")
    seq_txt  = os.path.join(seq_dir, "tracker", "seq.txt")

    result  = parse_tracker(seq_txt)
    stats   = result["stats"]
    fd      = result["frame_data"]
    rows    = result["all_rows"]

    print("Stats:", stats)

    kf = make_keyframes(img1_dir, fd, step=200)
    print(f"Keyframes: {len(kf)}")

    out_vid = os.path.join(seq_dir, "annotated.mp4")
    make_annotated_video(img1_dir, fd, out_vid, fps=args.fps,
                         progress_cb=lambda p: print(f"  video {p*100:.0f}%"))
    print(f"Video: {out_vid}")

    csv_path = os.path.join(seq_dir, "detections.csv")
    open(csv_path, "wb").write(make_csv_bytes(rows))
    print(f"CSV: {csv_path}")
