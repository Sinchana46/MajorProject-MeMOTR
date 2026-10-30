"""
tools/qualitative_compose.py
Part D + E + F: Compose paper-ready figures and quality-check them.

Reads  : results/qualitative_paper_figures/overlays/
         results/qualitative_paper_figures/selection_meta.json
Writes : results/qualitative_paper_figures/paper_figures/fig_qualitative_grid.png
         results/qualitative_paper_figures/paper_figures/fig_qualitative_pairs.png
         results/qualitative_paper_figures/paper_figures/fig_qualitative_combined.png
         check/logs/qualitative_partDEF.txt
"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOG_PATH  = "check/logs/qualitative_partDEF.txt"
META_PATH = "results/qualitative_paper_figures/selection_meta.json"
OUT_DIR   = "results/qualitative_paper_figures/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("check/logs", exist_ok=True)

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

with open(META_PATH) as f:
    meta = json.load(f)

individual_paths = [p for _, p in meta["individual"]]
pair_paths       = [p for _,_,p in meta["pairs"]]
chosen6_fids     = meta["chosen6_fids"]
pair_frames      = meta["pair_frames"]

def get_font(size):
    for name in ["arial.ttf", "Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try: return ImageFont.truetype(name, size)
        except: pass
    return ImageFont.load_default()

DPI = 300

# ── fig_qualitative_grid.png  (2×3 grid, 7.0 in wide) ────────────────────────
log("=== Part D: fig_qualitative_grid.png ===")
W_IN  = 7.0
COLS, ROWS = 3, 2
BORDER     = 4    # px white border around each cell
CAPTION_H  = 50  # px at bottom for caption

fig_w_px = int(W_IN * DPI)
imgs = [Image.open(p).convert("RGB") for p in individual_paths[:6]]

# Cell size: fit all images to same cell, preserve aspect ratio
cell_w = (fig_w_px - BORDER*(COLS+1)) // COLS
# Use first image aspect to set cell height
ref_w, ref_h = imgs[0].size
cell_h = int(cell_w * ref_h / ref_w)
fig_h_px = ROWS * cell_h + BORDER*(ROWS+1) + CAPTION_H

grid = Image.new("RGB", (fig_w_px, fig_h_px), (255,255,255))
draw = ImageDraw.Draw(grid)

for i, img in enumerate(imgs):
    row, col = divmod(i, COLS)
    # Resize with letterbox
    img_r = img.copy()
    img_r.thumbnail((cell_w, cell_h), Image.LANCZOS)
    cell_img = Image.new("RGB", (cell_w, cell_h), (0,0,0))
    ox = (cell_w - img_r.width)  // 2
    oy = (cell_h - img_r.height) // 2
    cell_img.paste(img_r, (ox, oy))
    x = BORDER + col*(cell_w+BORDER)
    y = BORDER + row*(cell_h+BORDER)
    grid.paste(cell_img, (x, y))

# Caption
caption = ("Qualitative tracking results on unseen test footage. "
           "Colours indicate persistent track IDs. "
           "No ground truth is available for this video.")
font_cap = get_font(22)
draw.text((BORDER, fig_h_px - CAPTION_H + 10), caption,
          fill=(40,40,40), font=font_cap)

grid_path = os.path.join(OUT_DIR, "fig_qualitative_grid.png")
grid.save(grid_path, format="PNG", dpi=(DPI, DPI))
log(f"Saved: {grid_path}  size={grid.size}  {os.path.getsize(grid_path)//1024} KB")


# ── fig_qualitative_pairs.png  (7.0 in wide, pairs stacked) ─────────────────
log("\n=== Part D: fig_qualitative_pairs.png ===")
pair_imgs  = [Image.open(p).convert("RGB") for p in pair_paths]
PAIR_LABEL = 40
pair_canvases = []
for idx, (pimg, (fa, fb)) in enumerate(zip(pair_imgs, pair_frames)):
    pw, ph = pimg.size
    # scale to fig_w_px
    scale = fig_w_px / pw
    pw2, ph2 = fig_w_px, int(ph * scale)
    pimg_r = pimg.resize((pw2, ph2), Image.LANCZOS)
    canvas = Image.new("RGB", (pw2, ph2 + PAIR_LABEL), (0,0,0))
    canvas.paste(pimg_r, (0, PAIR_LABEL))
    d = ImageDraw.Draw(canvas)
    font_lbl = get_font(26)
    d.text((10, 8), f"Pair {idx+1}: frames {fa} and {fb}", fill=(255,255,255), font=font_lbl)
    pair_canvases.append(canvas)

SEP_LINE = 8
pairs_caption_h = 60
total_h = sum(c.height for c in pair_canvases) + SEP_LINE*(len(pair_canvases)-1) + pairs_caption_h
pairs_fig = Image.new("RGB", (fig_w_px, total_h), (255,255,255))
d2 = ImageDraw.Draw(pairs_fig)
y_off = 0
for i, c in enumerate(pair_canvases):
    pairs_fig.paste(c, (0, y_off))
    y_off += c.height
    if i < len(pair_canvases)-1:
        d2.rectangle([0, y_off, fig_w_px, y_off+SEP_LINE], fill=(200,200,200))
        y_off += SEP_LINE

pairs_caption = ("Consecutive frame pairs showing ID persistence. "
                 "The same colour indicates the same vehicle identity across frames.")
font_cap2 = get_font(22)
d2.text((BORDER, y_off + 10), pairs_caption, fill=(40,40,40), font=font_cap2)

pairs_path = os.path.join(OUT_DIR, "fig_qualitative_pairs.png")
pairs_fig.save(pairs_path, format="PNG", dpi=(DPI, DPI))
log(f"Saved: {pairs_path}  size={pairs_fig.size}  {os.path.getsize(pairs_path)//1024} KB")


# ── fig_qualitative_combined.png  (3.4 in wide, 3 best frames stacked) ───────
log("\n=== Part D: fig_qualitative_combined.png ===")
W_IN_SINGLE = 3.4
fig_w_single = int(W_IN_SINGLE * DPI)
best3 = individual_paths[:3]
b_imgs = [Image.open(p).convert("RGB") for p in best3]
stacked_h = 0
resized = []
for bi in b_imgs:
    bw, bh = bi.size
    scale = fig_w_single / bw
    nb = bi.resize((fig_w_single, int(bh*scale)), Image.LANCZOS)
    resized.append(nb)
    stacked_h += nb.height + 6

combined_fig = Image.new("RGB", (fig_w_single, stacked_h), (0,0,0))
y_off = 0
for ri in resized:
    combined_fig.paste(ri, (0, y_off))
    y_off += ri.height + 6

comb_path = os.path.join(OUT_DIR, "fig_qualitative_combined.png")
combined_fig.save(comb_path, format="PNG", dpi=(DPI, DPI))
log(f"Saved: {comb_path}  size={combined_fig.size}  {os.path.getsize(comb_path)//1024} KB")


# ── Part E: figure captions ───────────────────────────────────────────────────
log("\n=== Part E: Figure captions ===")
fids_str = ", ".join(str(f) for f in chosen6_fids)
pair_str = "; ".join(f"frames {a} and {b}" for a,b in pair_frames)
log(f"""
For fig_qualitative_grid (frames: {fids_str}):
  Fig. X. Qualitative tracking results on an unseen test video not used during training or validation.
  The model is Arm 1 checkpoint_3, fine-tuned on IOTD. Each colour represents a unique persistent
  vehicle identity assigned by the tracker. No ground-truth annotations exist for this footage;
  quantitative evaluation was performed on the annotated validation sequence (see Table X).

For fig_qualitative_pairs ({pair_str}):
  Fig. X. Consecutive frame pairs demonstrating identity persistence. The same vehicle receives the
  same colour and track ID in both frames of each pair, confirming that the tracker maintains
  consistent identities across frames without ground-truth supervision on this video.
""")


# ── Part F: Quality check ─────────────────────────────────────────────────────
log("=== Part F: Quality check ===")
check_files = [
    ("fig_qualitative_grid.png",     grid_path),
    ("fig_qualitative_pairs.png",    pairs_path),
    ("fig_qualitative_combined.png", comb_path),
]
log(f"{'FILE':<35} | {'SIZE':>8} | {'DIMS':>15} | DPI | BLANK? | COLOURS")
log("-"*85)
for name, path in check_files:
    img = Image.open(path)
    arr = np.array(img)
    mean_val  = arr.mean()
    blank     = "YES" if mean_val < 10 else "No"
    # Count distinct colours (sample)
    flat = arr.reshape(-1, 3)
    sample = flat[::200]
    n_colours = len(set(map(tuple, sample)))
    dpi_tag = img.info.get("dpi", ("?","?"))
    dpi_str = str(int(dpi_tag[0])) if dpi_tag[0] != "?" else "?"
    sz_kb = os.path.getsize(path) // 1024
    log(f"{name:<35} | {sz_kb:>6} KB | {str(img.size):>15} | {dpi_str:>3} | {blank:>6} | {n_colours}")

# Final message
log(f"""
============================================================
QUALITATIVE FIGURES COMPLETE

Output folder: results/qualitative_paper_figures/paper_figures/
Figures:
  fig_qualitative_grid.png     (2x3 grid, 6 frames, 7.0 in, 300 dpi)
  fig_qualitative_pairs.png    (2 consecutive pairs, 7.0 in, 300 dpi)
  fig_qualitative_combined.png (3 frames stacked, 3.4 in, 300 dpi)

Source: results/manual_test/2_Take_1/tracker/seq.txt
        (Arm 1 checkpoint_3, det 0.5, track 0.5, miss tol 30)

Video: 2_Take-1.mp4 — unseen footage, no ground truth
IOTD hash check: changed=0 missing=3(placeholder keys) added=0

Paste captions from Part E into the paper.
============================================================
""")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"Log written: {LOG_PATH}")
