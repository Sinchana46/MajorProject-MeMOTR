"""
app.py — MeMOTR Vehicle Tracking  |  Streamlit web app
=======================================================
Launch from project root:
    conda activate memotr_project
    streamlit run app.py

Workflow:
  1. User uploads an mp4.
  2. App saves it to  data/IOTD_manual_test/<name>.mp4
  3. Runs  tools/test_new_video.py --video <path>  as a subprocess
     (frame extraction + MeMOTR inference with checkpoint_3.pth).
  4. Runs  tools/generate_test_report.py  (keyframes + annotated video).
  5. Shows cards, keyframe gallery, annotated video player, CSV download.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Project root (app.py lives at the project root) ───────────────────────────
ROOT       = Path(__file__).resolve().parent
TOOLS_DIR  = ROOT / "tools"
UPLOAD_DIR = ROOT / "data" / "IOTD_manual_test"
OUT_BASE   = ROOT / "results" / "manual_test"
PYTHON     = sys.executable

# Add tools to path so we can import generate_test_report
sys.path.insert(0, str(TOOLS_DIR))
from generate_test_report import (
    parse_tracker, make_keyframes, make_annotated_video, make_csv_bytes
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeMOTR Vehicle Tracker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0f111a; color: #e0e4f0; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2238 100%);
    border: 1px solid #2a2d45;
    border-radius: 16px;
    padding: 20px 18px 16px;
    text-align: center;
    margin-bottom: 12px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,179,255,0.15);
}
.metric-val {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-lbl {
    font-size: 0.82rem;
    color: #8890b0;
    margin-top: 6px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.metric-sub { font-size: 0.75rem; color: #555a7a; margin-top: 4px; }

/* Section headers */
.section-hdr {
    font-size: 1.1rem;
    font-weight: 600;
    color: #63b3ff;
    border-left: 3px solid #63b3ff;
    padding-left: 10px;
    margin: 28px 0 14px;
    letter-spacing: 0.02em;
}

/* Gallery frame */
.frame-card {
    background: #1a1d2e;
    border: 1px solid #2a2d45;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 10px;
}
.frame-lbl {
    font-size: 0.75rem;
    color: #8890b0;
    text-align: center;
    padding: 5px 0 7px;
}

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background: #1a1d2e !important;
    border: 2px dashed #2a2d45 !important;
    border-radius: 16px !important;
}

/* Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.85 !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #a78bfa) !important; }

/* Video player */
video { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def seq_name_from_video(video_path: Path) -> str:
    """MUST exactly mirror tools/test_new_video.py main():
       only replace spaces → _ and hyphens → _.
       Parentheses and all other chars are preserved."""
    return video_path.stem.replace(" ", "_").replace("-", "_")


def find_seq_txt(seq_dir: Path) -> Path | None:
    """Find the tracker output txt (seq.txt or largest .txt != label.txt)."""
    tracker_dir = seq_dir / "tracker"
    if not tracker_dir.exists():
        return None
    # prefer seq.txt
    if (tracker_dir / "seq.txt").exists():
        return tracker_dir / "seq.txt"
    # fallback: largest txt
    txts = [f for f in tracker_dir.glob("*.txt") if f.name != "label.txt"]
    if txts:
        return max(txts, key=lambda f: f.stat().st_size)
    return None


def card(col, value: str, label: str, sub: str = ""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{value}</div>
        <div class="metric-lbl">{label}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── State ──────────────────────────────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "upload"          # upload | running | done | error
if "seq_dir" not in st.session_state:
    st.session_state.seq_dir = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "vid_meta" not in st.session_state:
    st.session_state.vid_meta = {}
if "infer_time" not in st.session_state:
    st.session_state.infer_time = None
if "max_seconds" not in st.session_state:
    st.session_state.max_seconds = 0   # 0 = full video
if "fast_mode" not in st.session_state:
    st.session_state.fast_mode = False
if "building_done" not in st.session_state:
    st.session_state.building_done = False


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:24px 0 8px'>
  <span style='font-size:2.2rem;font-weight:700;
    background:linear-gradient(135deg,#63b3ff,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    🚗 MeMOTR Vehicle Tracker
  </span>
  <span style='font-size:0.9rem;color:#555a7a;margin-left:18px'>
    Arm 1 · checkpoint_3 · IOTD fine-tuned
  </span>
</div>
<hr style='border:none;border-top:1px solid #2a2d45;margin-bottom:18px'>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage in ("upload", "error"):

    if st.session_state.stage == "error":
        st.error("⚠️ Something went wrong. Check the log above and try again.")

    st.markdown('<div class="section-hdr">📤 Upload a Video</div>',
                unsafe_allow_html=True)
    st.caption("Supported: mp4, avi, mov, mkv · Single class: **vehicle**  "
               "· No annotations required")

    uploaded = st.file_uploader("Drop your video here",
                                type=["mp4", "avi", "mov", "mkv", "m4v"],
                                label_visibility="collapsed")

    if uploaded:
        save_path = UPLOAD_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Saved: `{save_path.name}`  "
                   f"({uploaded.size / 1024 / 1024:.1f} MB)")

        # Check if results already exist
        _sname   = seq_name_from_video(save_path)
        _seq_dir = OUT_BASE / _sname
        _seq_txt = find_seq_txt(_seq_dir)
        if _seq_txt and _seq_txt.stat().st_size > 0:
            st.info(f"ℹ️ Tracker output already exists for **{_sname}** — "
                    "you can skip inference and go straight to results.")
            if st.button("⚡ Use Existing Results", use_container_width=True):
                st.session_state.stage      = "building"
                st.session_state.video_name = uploaded.name
                st.session_state.seq_dir    = str(_seq_dir)
                st.session_state.infer_time = None
                st.session_state.vid_meta   = {}
                st.rerun()

        # ── Processing options ──────────────────────────────────────────────
        st.markdown('<div class="section-hdr" style="margin-top:18px">⚙️ Processing Options</div>',
                    unsafe_allow_html=True)
        col_opt1, col_opt2 = st.columns([3, 2])
        with col_opt1:
            max_sec = st.slider(
                "Process first N seconds  (0 = full video)",
                min_value=0, max_value=60, value=st.session_state.max_seconds,
                step=5,
                help="Trim the video before running inference. "
                     "10–15 s gives results in ~1–2 min on CPU."
            )
        with col_opt2:
            fast_mode = st.checkbox(
                "⚡ Fast mode (10 fps + 640px width)",
                value=st.session_state.fast_mode,
                help="Samples at 10 fps AND scales frames to 640px wide. "
                     "Cuts inference time by 6–10× on CPU — "
                     "a 10s clip should finish in under 2 minutes."
            )
        if max_sec > 0:
            fps_est = 10 if fast_mode else 30
            extra   = "  ·  640px width" if fast_mode else ""
            st.info(f"ℹ️ Only the first **{max_sec}s** will be processed "
                    f"(~{max_sec * fps_est} frames{extra}).")
        elif fast_mode:
            st.info("ℹ️ Fast mode: 10 fps + 640px wide — expect 6–10× speedup on CPU.")

        run_lbl = "🔄 Re-run Tracker" if (_seq_txt and _seq_txt.stat().st_size > 0) else "🚀 Run Tracker"
        if st.button(run_lbl, type="primary", use_container_width=True):
            st.session_state.stage       = "running"
            st.session_state.video_name  = uploaded.name
            st.session_state.seq_dir     = None
            st.session_state.infer_time  = None
            st.session_state.max_seconds = max_sec
            st.session_state.fast_mode   = fast_mode
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RUNNING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "running":

    video_path = UPLOAD_DIR / st.session_state.video_name
    sname      = seq_name_from_video(video_path)
    seq_dir    = OUT_BASE / sname

    st.markdown(f'<div class="section-hdr">⚙️ Processing  '
                f'<code>{st.session_state.video_name}</code></div>',
                unsafe_allow_html=True)

    status_box  = st.empty()
    prog_bar    = st.progress(0)
    log_expander = st.expander("Show live log", expanded=False)
    log_area    = log_expander.empty()

    def update(msg: str, frac: float, log: str = ""):
        status_box.info(f"**{msg}**")
        prog_bar.progress(frac)
        if log:
            log_area.code(log[-3000:], language="")   # last 3000 chars

    # ── Step A: inference (test_new_video.py) ──────────────────────────────
    max_seconds = st.session_state.get("max_seconds", 0)
    fast_mode   = st.session_state.get("fast_mode", False)

    trim_label = f", first {max_seconds}s" if max_seconds else ""
    fps_label  = ", 10 fps" if fast_mode else ""
    update(f"🎞️ Extracting frames & running tracker{trim_label}{fps_label} …", 0.05)

    cmd_infer = [
        PYTHON, "-u",
        str(TOOLS_DIR / "test_new_video.py"),
        "--video", str(video_path),
    ]
    if max_seconds > 0:
        cmd_infer += ["--max-seconds", str(max_seconds)]
    if fast_mode:
        cmd_infer += ["--fps-out", "10", "--scale-width", "640"]
    t0   = time.time()
    log  = ""
    try:
        import os as _os
        _env = _os.environ.copy()
        _env["PYTHONIOENCODING"] = "utf-8"
        _env["PYTHONUTF8"] = "1"
        proc = subprocess.Popen(
            cmd_infer,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            env=_env,
        )
        for line in proc.stdout:
            log += line
            # crude progress from log keywords
            if "extract" in line.lower():
                update("🎞️ Extracting frames …", 0.15, log)
            elif "inference" in line.lower() or "tracking" in line.lower():
                update("🤖 Running MeMOTR tracker …", 0.45, log)
            elif "frame" in line.lower() and "%" in line:
                update("🤖 Running MeMOTR tracker …", 0.55, log)
            elif "summary" in line.lower() or "detections" in line.lower():
                update("📊 Finalising output …", 0.75, log)
        proc.wait()
        if proc.returncode != 0:
            st.error("Inference subprocess failed. See log above.")
            log_area.code(log, language="")
            st.session_state.stage = "error"
            st.stop()
    except Exception as e:
        st.error(f"Failed to launch subprocess: {e}")
        st.session_state.stage = "error"
        st.stop()

    infer_time = time.time() - t0
    update("📊 Inference done! Building report …", 0.82, log)

    # ── Step B: find tracker output ────────────────────────────────────────
    seq_txt = find_seq_txt(seq_dir)
    if seq_txt is None or seq_txt.stat().st_size == 0:
        st.error(
            f"Tracker output not found or empty in `{seq_dir / 'tracker'}`."
            "\n\nCheck the live log for the actual output path printed by the tracker."
        )
        log_area.code(log, language="")
        st.session_state.stage = "error"
        st.stop()

    update("✅ Inference done!", 0.85, log)
    time.sleep(0.3)

    # ── Move to building stage ─────────────────────────────────────────────
    st.session_state.stage      = "building"
    st.session_state.seq_dir    = str(seq_dir)
    st.session_state.infer_time = infer_time
    st.session_state.vid_meta   = {}
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BUILDING ANNOTATED VIDEO
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "building":

    seq_dir    = Path(st.session_state.seq_dir)
    infer_time = st.session_state.infer_time   # may be None if skipped

    st.markdown('<div class="section-hdr">🎬 Drawing Bounding Boxes &amp; Encoding Video</div>',
                unsafe_allow_html=True)
    bstatus  = st.empty()
    bprog    = st.progress(0)

    def bupdate(msg: str, frac: float):
        bstatus.info(f"**{msg}**")
        bprog.progress(frac)

    bupdate("🔍 Parsing tracker output …", 0.05)

    seq_txt = find_seq_txt(seq_dir)
    if seq_txt is None or seq_txt.stat().st_size == 0:
        st.error(f"Cannot find tracker output in `{seq_dir / 'tracker'}`")
        st.session_state.stage = "error"
        st.stop()

    result     = parse_tracker(str(seq_txt))
    frame_data = result["frame_data"]
    img1_dir   = seq_dir / "seq" / "img1"

    # Read seqinfo
    seqinfo_path = seq_dir / "seq" / "seqinfo.ini"
    fps_out    = 30.0
    duration_s = 0.0
    width, height = 0, 0
    if seqinfo_path.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(str(seqinfo_path))
        try:
            fps_out    = float(cfg["Sequence"]["framerate"])
            n_frames   = int(cfg["Sequence"]["seqlength"])
            duration_s = n_frames / fps_out
            width      = int(cfg["Sequence"].get("imwidth", 0))
            height     = int(cfg["Sequence"].get("imheight", 0))
        except Exception:
            pass

    ann_vid_path = str(seq_dir / "annotated.mp4")
    _av = Path(ann_vid_path)
    if _av.exists() and _av.stat().st_size > 0:
        # Validate it's a real mp4 (has a moov atom)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1",
             ann_vid_path],
            capture_output=True, text=True
        )
        if probe.returncode == 0 and "duration" in probe.stdout:
            bupdate("✅ Valid annotated video already exists — skipping re-encode.", 0.95)
        else:
            # Corrupt file — delete and re-encode
            _av.unlink()
            bupdate("⚠️ Existing video was corrupt — re-encoding …", 0.10)
            _av = Path("")
    if not (_av.exists() and _av.stat().st_size > 0):
        bupdate("🎬 Drawing bounding boxes on every frame …", 0.10)
        try:
            def _vcb(p):
                bupdate(f"🎬 Encoding annotated video … {p*100:.0f}%",
                        0.10 + p * 0.85)
            make_annotated_video(str(img1_dir), frame_data,
                                 ann_vid_path, fps=fps_out,
                                 progress_cb=_vcb)
        except Exception as e:
            st.warning(f"Annotated video encoding failed: {e} — continuing without it.")
            ann_vid_path = None

    bupdate("✅ Done! Loading results …", 1.0)
    time.sleep(0.3)

    st.session_state.stage    = "done"
    st.session_state.vid_meta = {
        "fps": fps_out, "duration_s": duration_s,
        "width": width, "height": height,
    }
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "done":

    seq_dir    = Path(st.session_state.seq_dir)
    infer_time = st.session_state.infer_time or 0.0
    meta       = st.session_state.vid_meta
    fps_out    = meta.get("fps", 30.0)
    duration_s = meta.get("duration_s", 0.0)
    width      = meta.get("width", 0)
    height     = meta.get("height", 0)
    video_name = st.session_state.video_name or ""

    # Re-parse (fast, in-memory)
    seq_txt    = find_seq_txt(seq_dir)
    result     = parse_tracker(str(seq_txt))
    stats      = result["stats"]
    frame_data = result["frame_data"]
    all_rows   = result["all_rows"]
    img1_dir   = seq_dir / "seq" / "img1"
    ann_vid    = seq_dir / "annotated.mp4"

    avg_fps = stats["total_frames"] / infer_time if infer_time > 0 else 0.0

    # ── Header ──────────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([5, 1])
    col_title.markdown(
        f'<div class="section-hdr">📊 Results — <code>{video_name}</code></div>',
        unsafe_allow_html=True)
    if col_btn.button("🔄 New Video"):
        st.session_state.stage = "upload"
        st.rerun()

    # ── Metric cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    card(c1, f"{stats['total_frames']:,}",     "Total Frames",
         f"{duration_s:.1f}s video")
    card(c2, f"{stats['unique_ids']}",          "Unique Vehicles",
         "Persistent track IDs")
    card(c3, f"{stats['peak_count']}",          "Peak Count",
         f"Frame {stats.get('peak_frame', '?')}")
    card(c4, f"{stats['total_detections']:,}",  "Total Detections",
         f"avg {stats['avg_dets_per_frame']:.1f}/frame")
    card(c5, f"{infer_time:.0f}s" if infer_time else "—",
         "Inference Time",
         f"~{avg_fps:.1f} fps" if infer_time else "skipped (cached)")
    card(c6, f"{avg_fps:.1f}" if infer_time else "—",
         "Avg Inference FPS",
         f"{stats['total_frames']:,} frames")
    card(c7, f"{duration_s:.1f}s",             "Video Duration",
         f"{fps_out:.0f} fps · {width}×{height}")

    st.divider()

    # ── Annotated video ──────────────────────────────────────────────────────
    if ann_vid.exists() and ann_vid.stat().st_size > 0:
        st.markdown('<div class="section-hdr">🎬 Annotated Output Video</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Every vehicle is tracked with a **persistent coloured bounding box** "
            "and unique ID label across all frames."
        )
        # Read bytes and pass to st.video — more reliable than a file path
        # for H.264 mp4 files across different OS/browser combos
        vid_bytes = open(str(ann_vid), "rb").read()
        st.video(vid_bytes, format="video/mp4")
        col_dl1, _ = st.columns([2, 5])
        col_dl1.download_button(
            "⬇️ Download Annotated Video",
            data=vid_bytes,
            file_name=f"{seq_dir.name}_annotated.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        st.divider()
    else:
        st.warning("Annotated video not available — encoding may have failed.")

    # ── Keyframe gallery ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🖼️ Keyframe Gallery</div>',
                unsafe_allow_html=True)
    st.caption("Coloured boxes = unique persistent vehicle track IDs · "
               "Sampled every ~200 frames")

    kf_imgs = make_keyframes(str(img1_dir), frame_data, step=200)
    if kf_imgs:
        COLS = 3
        rows = [kf_imgs[i:i+COLS] for i in range(0, len(kf_imgs), COLS)]
        for row in rows:
            cols = st.columns(COLS)
            for ci, (fid, img_bgr) in enumerate(row):
                with cols[ci]:
                    h, w = img_bgr.shape[:2]
                    scale = min(1.0, 600 / max(w, h))
                    disp  = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
                    st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB),
                             caption=f"Frame {fid}  ·  {len(frame_data.get(fid,[]))} tracks",
                             use_container_width=True)
    else:
        st.warning("No keyframes found — check that img1/ directory has JPEG frames.")

    st.divider()

    # ── CSV download ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📥 Download Detections CSV</div>',
                unsafe_allow_html=True)
    st.caption("Columns: frame, track_id, x, y, w, h, score")
    csv_bytes = make_csv_bytes(all_rows)
    col_dl2, col_prev, _ = st.columns([2, 2, 3])
    col_dl2.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"{seq_dir.name}_detections.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if col_prev.button("👁️ Preview first 20 rows", use_container_width=True):
        import pandas as pd
        import io as _io
        df = pd.read_csv(_io.BytesIO(csv_bytes)).head(20)
        st.dataframe(df, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style='border:none;border-top:1px solid #2a2d45;margin-top:32px'>
    <div style='text-align:center;color:#3a3d5c;font-size:0.75rem;padding:8px 0 16px'>
        MeMOTR · Arm 1 · checkpoint_3.pth · IOTD fine-tuned · miss_tol 30 · No ground truth required
    </div>""", unsafe_allow_html=True)
