"""
tools/qualitative_overlay.py
Part B + C: Select best frames and draw tracker overlays on them.

Reads  : results/manual_test/2_Take_1/tracker/seq.txt
         results/manual_test/2_Take_1/seq/img1/
Writes : results/qualitative_paper_figures/overlays/frame_XXXXXX.png
         results/qualitative_paper_figures/overlays/frame_XXXXXX_pair.png
         check/logs/qualitative_partBC.txt
"""
import os, sys, json, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOG_PATH  = "check/logs/qualitative_partBC.txt"
TRACKER   = "results/manual_test/2_Take_1/tracker/seq.txt"
IMG1_DIR  = "results/manual_test/2_Take_1/seq/img1"
OUT_DIR   = "results/qualitative_paper_figures/overlays"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("check/logs", exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# ── Colour palette (tab20 equivalent, 20 distinguishable colours) ─────────────
PALETTE = [
    (31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
    (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207),
    (174,199,232),(255,187,120),(152,223,138),(255,152,150),(197,176,213),
    (196,156,148),(247,182,210),(199,199,199),(219,219,141),(158,218,229),
]
def colour(track_id):
    return PALETTE[int(track_id) % len(PALETTE)]

# ── Load tracker output ───────────────────────────────────────────────────────
log("=== Part A (re-verify) ===")
rows = open(TRACKER).read().strip().split("\n")
log(f"seq.txt rows: {len(rows)}")

# frame_data[frame] = list of (id, x, y, w, h)
frame_data = collections.defaultdict(list)
for r in rows:
    p = r.strip().split(",")
    if len(p) < 6: continue
    fid  = int(p[0])
    tid  = int(p[1])
    x,y,w,h = float(p[2]),float(p[3]),float(p[4]),float(p[5])
    frame_data[fid].append((tid,x,y,w,h))

all_frames = sorted(frame_data.keys())
log(f"Frame range   : {all_frames[0]} – {all_frames[-1]}")
log(f"Track IDs     : {len(set(tid for tracks in frame_data.values() for tid,*_ in tracks))}")

# ── Part B: select frames ─────────────────────────────────────────────────────
log("\n=== Part B: Frame selection ===")

# Histogram of track counts
counts = {f: len(tracks) for f, tracks in frame_data.items()}
bins = collections.Counter((c//5)*5 for c in counts.values())
log("Track-count histogram (binned by 5):")
for b in sorted(bins): log(f"  {b:3d}-{b+4}: {bins[b]} frames")

# Frames with >= 4 active tracks
candidates = sorted([f for f, c in counts.items() if c >= 4])
log(f"\nFrames with >=4 active tracks: {len(candidates)}")

# Pick 6 spread across the video
def pick_spread(frames, n):
    if len(frames) <= n: return frames
    step = (len(frames)-1)/(n-1)
    return [frames[round(i*step)] for i in range(n)]

chosen6 = pick_spread(candidates, 6)
log("Chosen 6 individual frames:")
for f in chosen6: log(f"  frame {f:06d}  tracks={counts[f]}")

# Pick 2 consecutive pairs where both N and N+1 have >=3 tracks
pairs = []
for f in all_frames[:-1]:
    if f+1 in frame_data and len(frame_data[f]) >= 3 and len(frame_data[f+1]) >= 3:
        pairs.append((f, f+1))
        if len(pairs) == 2: break
log(f"\nChosen consecutive pairs:")
for a,b in pairs: log(f"  frames {a} and {b}  (tracks: {len(frame_data[a])}, {len(frame_data[b])})")

# ── Part C: draw overlays ─────────────────────────────────────────────────────
log("\n=== Part C: Drawing overlays ===")

def load_frame(fid):
    fname = os.path.join(IMG1_DIR, f"{fid:06d}.jpg")
    return Image.open(fname).convert("RGB")

def draw_overlay(img, tracks, show_header=True):
    """Draw bounding boxes + IDs on a copy of img. Returns annotated PIL Image."""
    out  = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    W, H = out.size

    # Try to load a font; fall back to default
    try:
        font_lbl = ImageFont.truetype("arial.ttf", 22)
        font_hdr = ImageFont.truetype("arial.ttf", 26)
    except:
        try:
            font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_hdr = font_lbl
        except:
            font_lbl = ImageFont.load_default()
            font_hdr = font_lbl

    for tid, x, y, w, h in tracks:
        c = colour(tid)
        x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(W,x2), min(H,y2)
        # Thick bounding box
        draw.rectangle([x1,y1,x2,y2], outline=c, width=3)
        # Label background
        label = str(tid)
        try:
            bb = font_lbl.getbbox(label)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except:
            tw, th = 14*len(label), 18
        pad = 4
        lx1, ly1 = x1, max(0, y1-th-pad*2)
        lx2, ly2 = x1+tw+pad*2, y1
        draw.rectangle([lx1,ly1,lx2,ly2], fill=c+(220,))
        draw.text((lx1+pad, ly1+pad), label, fill=(255,255,255), font=font_lbl)

    # Header bar
    if show_header:
        hdr = "Test video — unseen footage (no ground truth)"
        bar_h = 44
        draw.rectangle([0,0,W,bar_h], fill=(0,0,0,160))
        draw.text((10, 8), hdr, fill=(255,255,255), font=font_hdr)

    return out

# Individual frames
saved_individual = []
for fid in chosen6:
    img    = load_frame(fid)
    ann    = draw_overlay(img, frame_data[fid])
    out_p  = os.path.join(OUT_DIR, f"frame_{fid:06d}.png")
    ann.save(out_p, format="PNG")
    saved_individual.append((fid, out_p))
    log(f"  Saved: {out_p}  ({os.path.getsize(out_p)//1024} KB)")

# Pair images
saved_pairs = []
for idx, (fa, fb) in enumerate(pairs):
    ia = draw_overlay(load_frame(fa), frame_data[fa], show_header=False)
    ib = draw_overlay(load_frame(fb), frame_data[fb], show_header=False)
    W, H = ia.size
    # side-by-side with separator
    sep = 6
    combined = Image.new("RGB", (W*2+sep, H+40), (0,0,0))

    try:
        font_pair = ImageFont.truetype("arial.ttf", 22)
    except:
        try:
            font_pair = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except:
            font_pair = ImageFont.load_default()

    combined.paste(ia, (0, 40))
    combined.paste(ib, (W+sep, 40))
    draw_c = ImageDraw.Draw(combined)
    draw_c.rectangle([W, 0, W+sep, H+40], fill=(255,255,255))
    draw_c.text((10, 8), f"Frame {fa}", fill=(255,255,255), font=font_pair)
    draw_c.text((W+sep+10, 8), f"Frame {fb}", fill=(255,255,255), font=font_pair)

    out_p = os.path.join(OUT_DIR, f"frame_{fa:06d}_pair.png")
    combined.save(out_p, format="PNG")
    saved_pairs.append((fa, fb, out_p))
    log(f"  Pair saved: {out_p}  ({os.path.getsize(out_p)//1024} KB)")

# Export selection metadata for compose script
meta = {
    "individual": [(fid, p) for fid, p in saved_individual],
    "pairs":      [(fa, fb, p) for fa, fb, p in saved_pairs],
    "chosen6_fids": chosen6,
    "pair_frames":  pairs,
}
meta_path = "results/qualitative_paper_figures/selection_meta.json"
os.makedirs("results/qualitative_paper_figures", exist_ok=True)
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
log(f"\nMeta saved: {meta_path}")

log("\nPart B + C complete.")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"Log written: {LOG_PATH}")
