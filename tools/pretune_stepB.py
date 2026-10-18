"""
Pre-tuning package Step B: generate all paper-ready figures (fig01-fig08 + dashboard).
Okabe-Ito palette, 300 dpi PNG + PDF, footer on every figure.
"""
import os, sys, csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

FIG_DIR  = "results/pre_tuning_baseline_v1/figures"
DATA_DIR = "results/pre_tuning_baseline_v1/data"
IMG_DIR  = "data/IOTD_val/seq_02_30sec_a/img1"
TRACKER_BASE = "results/eval_iotd_val"
os.makedirs(FIG_DIR, exist_ok=True)

FOOTER = ("Validation: seq_02_30sec_a (744 frames, 50 tracks, night). "
          "Single run per setting, seed 42. Thresholds 0.5. Not statistically established.")

# ── Okabe-Ito palette ─────────────────────────────────────────────────────────
C_PHASE5 = "#999999"   # grey
C_ARM1   = "#0072B2"   # blue
C_ARM2   = "#D55E00"   # vermilion
M_PHASE5 = "s"
M_ARM1   = "o"
M_ARM2   = "^"
LS_PHASE5= "--"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def add_footer(fig):
    fig.text(0.5, 0.01, FOOTER, ha="center", va="bottom",
             fontsize=5.5, color="#555555", style="italic",
             wrap=True)

def savefig(fig, name):
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png + .pdf")
    return png_path

# ── Load data ─────────────────────────────────────────────────────────────────
pc_path = os.path.join(DATA_DIR, "per_checkpoint_metrics.csv")
pc_data = []
with open(pc_path, newline="") as f:
    for row in csv.DictReader(f):
        for k in ["steps","epochs_completed","IDSW","TP","FP","FN"]:
            row[k] = int(float(row[k]))
        for k in ["HOTA","DetA","AssA","MOTA","IDF1","recall","precision"]:
            row[k] = float(row[k])
        pc_data.append(row)

def arm_data(run_label):
    rows = [r for r in pc_data if r["run"]==run_label]
    return sorted(rows, key=lambda r: r["steps"])

ph = arm_data("Phase5")
a1 = arm_data("Arm1")
a2 = arm_data("Arm2")
ph5_hota = ph[0]["HOTA"] if ph else 50.51

# Best checkpoints
best1 = max(a1, key=lambda r: r["HOTA"])
best2 = max(a2, key=lambda r: r["HOTA"])

print("="*60)
print("STEP B: GENERATING FIGURES")
print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# fig01: HOTA vs steps
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.5))

# Phase5 horizontal reference
ax.axhline(ph5_hota, color=C_PHASE5, ls=LS_PHASE5, lw=1.5, label=f"Phase 5 ({ph5_hota:.2f})")

# Arm1
s1 = [r["steps"] for r in a1]; h1 = [r["HOTA"] for r in a1]
ax.plot(s1, h1, color=C_ARM1, marker=M_ARM1, lw=1.5, ms=6, label="Arm 1 (real only)")
for r in a1:
    ax.annotate(f"{r['HOTA']:.1f}", (r["steps"], r["HOTA"]),
                textcoords="offset points", xytext=(0, 5), fontsize=6, color=C_ARM1, ha="center")

# Arm2
s2 = [r["steps"] for r in a2]; h2 = [r["HOTA"] for r in a2]
ax.plot(s2, h2, color=C_ARM2, marker=M_ARM2, lw=1.5, ms=6, label="Arm 2 (real + aug.)")
for r in a2:
    ax.annotate(f"{r['HOTA']:.1f}", (r["steps"], r["HOTA"]),
                textcoords="offset points", xytext=(0, -10), fontsize=6, color=C_ARM2, ha="center")

# Best checkpoint markers
ax.scatter([best1["steps"]], [best1["HOTA"]], color=C_ARM1, marker="*", s=180, zorder=5)
ax.scatter([best2["steps"]], [best2["HOTA"]], color=C_ARM2, marker="*", s=180, zorder=5)

# LR drop markers
ax.axvline(10112, color=C_ARM1, ls=":", lw=1.0, alpha=0.7)
ax.axvline(40448, color=C_ARM2, ls=":", lw=1.0, alpha=0.7)
ax.text(10112+300, 36.5, "LR drop\n(Arm1)", fontsize=5.5, color=C_ARM1)
ax.text(40448+300, 36.5, "LR drop\n(Arm2)", fontsize=5.5, color=C_ARM2)

# Equal-step pair connector: Arm1 ckpt3 vs Arm2 ckpt0 (both 10112)
arm1_at_eq  = next(r for r in a1 if r["steps"]==10112)
arm2_at_eq  = next(r for r in a2 if r["steps"]==10112)
ax.annotate("", xy=(arm2_at_eq["steps"], arm2_at_eq["HOTA"]),
            xytext=(arm1_at_eq["steps"], arm1_at_eq["HOTA"]),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
ax.text(10112+500, (arm1_at_eq["HOTA"]+arm2_at_eq["HOTA"])/2,
        f"Equal steps\n{arm1_at_eq['HOTA']:.1f} vs {arm2_at_eq['HOTA']:.1f}", fontsize=5.5)

ax.set_xlabel("Optimizer steps")
ax.set_ylabel("HOTA (%)")
ax.legend(loc="lower right", framealpha=0.85)
ax.set_xlim(-500, 53000)
ax.set_ylim(35, 55)
ax.grid(True, alpha=0.3)
add_footer(fig)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savefig(fig, "fig01_hota_vs_steps")

# ─────────────────────────────────────────────────────────────────────────────
# fig02: DetA and AssA vs steps (stacked subplots, 3.4 in)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.5), sharex=True)

ph5_det = ph[0]["DetA"]; ph5_ass = ph[0]["AssA"]

for ax, metric, label, ph5_val in [(ax1,"DetA","DetA (%)",ph5_det),(ax2,"AssA","AssA (%)",ph5_ass)]:
    ax.axhline(ph5_val, color=C_PHASE5, ls=LS_PHASE5, lw=1.2, label=f"Phase5 ({ph5_val:.1f})")
    ax.plot([r["steps"] for r in a1],[r[metric] for r in a1], color=C_ARM1, marker=M_ARM1, lw=1.2, ms=4, label="Arm1")
    ax.plot([r["steps"] for r in a2],[r[metric] for r in a2], color=C_ARM2, marker=M_ARM2, lw=1.2, ms=4, label="Arm2")
    ax.set_ylabel(label); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")

ax2.set_xlabel("Optimizer steps")
add_footer(fig)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savefig(fig, "fig02_deta_assa_vs_steps")

# ─────────────────────────────────────────────────────────────────────────────
# fig03: FP / FN / Recall / Precision vs steps
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))
for metric, ax, label, ph_val in [
    ("FP",  axes[0,0],"False Positives",ph[0]["FP"]),
    ("FN",  axes[0,1],"False Negatives",ph[0]["FN"]),
    ("recall",  axes[1,0],"Recall (%)", ph[0]["recall"]),
    ("precision",axes[1,1],"Precision (%)",ph[0]["precision"]),
]:
    ax.axhline(ph_val, color=C_PHASE5, ls=LS_PHASE5, lw=1.2, label=f"Phase5 ({ph_val:.0f})" if metric in ("FP","FN") else f"Phase5 ({ph_val:.1f})")
    ax.plot([r["steps"] for r in a1],[r[metric] for r in a1], color=C_ARM1, marker=M_ARM1, lw=1.2, ms=4, label="Arm1")
    ax.plot([r["steps"] for r in a2],[r[metric] for r in a2], color=C_ARM2, marker=M_ARM2, lw=1.2, ms=4, label="Arm2")
    ax.set_xlabel("Steps"); ax.set_ylabel(label)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
add_footer(fig)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savefig(fig, "fig03_fp_fn_vs_steps")

# ─────────────────────────────────────────────────────────────────────────────
# fig04: Detection outcomes 2x2 grid
# ─────────────────────────────────────────────────────────────────────────────
det_path = os.path.join(DATA_DIR, "detection_outcomes.csv")
det_data = []
with open(det_path, newline="") as f:
    for row in csv.DictReader(f):
        for k in ["TP_own","FP_own","FN_own"]:
            row[k] = int(row[k])
        for k in ["recall_own","precision_own"]:
            row[k] = float(row[k])
        det_data.append(row)

fig, big_axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
CAND_LABELS = [("Phase5","ckpt_4","Phase 5",C_PHASE5),
               ("Arm1","ckpt_3","Arm 1 (ckpt3)",C_ARM1),
               ("Arm2","ckpt_1","Arm 2 (ckpt1)",C_ARM2)]

for big_ax, (run, ckpt, label, color) in zip(big_axes, CAND_LABELS):
    row = next((r for r in det_data if r["run"]==run and r["checkpoint"]==ckpt), None)
    if row is None:
        big_ax.text(0.5,0.5,"No data",ha="center"); continue
    big_ax.axis("off")
    # 2x2 grid inside
    gs = big_ax.get_position()
    inset = fig.add_axes([gs.x0, gs.y0, gs.width, gs.height])
    inset.axis("off")
    # Draw 2x2 grid of text boxes
    cells = [
        ("TP", row["TP_own"], color, "white"),
        ("FP", row["FP_own"], "#E69F00", "white"),
        ("FN", row["FN_own"], "#56B4E9", "white"),
        ("TN", "N/A\n(single-class)", "#D0D0D0", "#888888"),
    ]
    for idx, (lbl, val, bg, fg) in enumerate(cells):
        xi = idx % 2; yi = idx // 2
        rect = mpatches.FancyBboxPatch(
            (xi*0.5, (1-yi)*0.5-0.5+0.02), 0.46, 0.44,
            boxstyle="round,pad=0.02", fc=bg, ec="white", lw=1,
            transform=inset.transAxes, clip_on=False)
        inset.add_patch(rect)
        inset.text(xi*0.5+0.23, (1-yi)*0.5-0.5+0.24, f"{lbl}\n{val}",
                   transform=inset.transAxes, ha="center", va="center",
                   color=fg, fontsize=7.5, fontweight="bold")
    rec = row["recall_own"]; prec = row["precision_own"]
    inset.text(0.5, -0.04, f"{label}\nRec={rec:.1f}% Prec={prec:.1f}%",
               transform=inset.transAxes, ha="center", fontsize=7, color=color)

add_footer(fig)
fig.savefig(os.path.join(FIG_DIR,"fig04_detection_outcomes.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR,"fig04_detection_outcomes.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved: fig04_detection_outcomes.png + .pdf")

# ─────────────────────────────────────────────────────────────────────────────
# fig05: Grouped bars — HOTA DetA AssA MOTA IDF1 for 3 candidates
# ─────────────────────────────────────────────────────────────────────────────
metrics5 = ["HOTA","DetA","AssA","IDF1"]
cands = [
    ("Phase5","ckpt_4","Phase 5",C_PHASE5),
    ("Arm1","ckpt_3","Arm 1\nckpt3",C_ARM1),
    ("Arm2","ckpt_1","Arm 2\nckpt1",C_ARM2),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios":[3,1]})
ax_main, ax_mota = axes

x = np.arange(len(metrics5))
bw = 0.25
for bi, (run, ckpt, clabel, color) in enumerate(cands):
    row = next((r for r in pc_data if r["run"]==run and r["checkpoint"]==ckpt), None)
    if row is None: continue
    vals = [row[m] for m in metrics5]
    bars = ax_main.bar(x + bi*bw, vals, bw, label=clabel, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax_main.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}", ha="center", fontsize=5.5)

ax_main.set_xticks(x + bw); ax_main.set_xticklabels(metrics5)
ax_main.set_ylabel("Metric (%)"); ax_main.set_ylim(0, 80)
ax_main.legend(fontsize=6); ax_main.grid(axis="y", alpha=0.3)
ax_main.set_title("Main metrics — 3 candidates")

# MOTA separate (can go below 0)
mota_vals = []
for run, ckpt, clabel, color in cands:
    row = next((r for r in pc_data if r["run"]==run and r["checkpoint"]==ckpt), None)
    mota_vals.append((clabel.replace("\n"," "), row["MOTA"] if row else 0, color))
ax_mota.bar([m[0] for m in mota_vals], [m[1] for m in mota_vals],
            color=[m[2] for m in mota_vals], alpha=0.85)
for i, (lbl, val, _) in enumerate(mota_vals):
    ax_mota.text(i, val + (0.5 if val >= 0 else -1.5), f"{val:.1f}", ha="center", fontsize=5.5)
ax_mota.axhline(0, color="black", lw=0.8)
ax_mota.set_ylabel("MOTA (%)"); ax_mota.set_title("MOTA")
ax_mota.grid(axis="y", alpha=0.3)

add_footer(fig)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savefig(fig, "fig05_candidate_comparison")

# ─────────────────────────────────────────────────────────────────────────────
# fig06: Identity metrics — IDSW, AssA, IDF1 for 3 candidates
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
id_metrics = [("AssA","AssA (%)"),("IDF1","IDF1 (%)"),("IDSW","ID Switches")]
cand_labels2 = ["Phase5\nckpt4","Arm1\nckpt3","Arm2\nckpt1"]
cand_colors2 = [C_PHASE5, C_ARM1, C_ARM2]
cand_keys2 = [("Phase5","ckpt_4"),("Arm1","ckpt_3"),("Arm2","ckpt_1")]

for ax, (metric, label) in zip(axes, id_metrics):
    vals = []
    for run, ckpt in cand_keys2:
        row = next((r for r in pc_data if r["run"]==run and r["checkpoint"]==ckpt), None)
        vals.append(row[metric] if row else 0)
    bars = ax.bar(cand_labels2, vals, color=cand_colors2, alpha=0.85)
    best_idx = vals.index(min(vals) if metric=="IDSW" else max(vals))
    bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(2.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val:.1f}" if metric!="IDSW" else str(int(val)),
                ha="center", fontsize=6)
    ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)
    ax.set_title(label)

add_footer(fig)
fig.tight_layout(rect=[0, 0.04, 1, 1])
savefig(fig, "fig06_identity_metrics")

# ─────────────────────────────────────────────────────────────────────────────
# fig07: Epoch-wise HOTA grouped bars + compact table
# ─────────────────────────────────────────────────────────────────────────────
epochs1 = [r["epochs_completed"] for r in a1]
epochs2 = [r["epochs_completed"] for r in a2]
hota1   = [r["HOTA"] for r in a1]
hota2   = [r["HOTA"] for r in a2]

fig = plt.figure(figsize=(7.0, 4.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.5], hspace=0.4)
ax_bar = fig.add_subplot(gs[0])

bw = 0.35
x1 = np.arange(len(epochs1))
x2 = x1  # same epoch count

bars1 = ax_bar.bar(x1 - bw/2, hota1, bw, label="Arm 1", color=C_ARM1, alpha=0.85)
bars2 = ax_bar.bar(x2 + bw/2, hota2, bw, label="Arm 2", color=C_ARM2, alpha=0.85)

# Highlight best
best1_idx = hota1.index(max(hota1))
best2_idx = hota2.index(max(hota2))
bars1[best1_idx].set_edgecolor("gold"); bars1[best1_idx].set_linewidth(2.5)
bars2[best2_idx].set_edgecolor("gold"); bars2[best2_idx].set_linewidth(2.5)

ax_bar.axhline(ph5_hota, color=C_PHASE5, ls=LS_PHASE5, lw=1.2, label=f"Phase5 ({ph5_hota:.1f})")
ax_bar.set_xticks(x1)
ax_bar.set_xticklabels([f"Ep{e}" for e in epochs1])
ax_bar.set_ylabel("HOTA (%)"); ax_bar.set_ylim(35,55)
ax_bar.legend(fontsize=6); ax_bar.grid(axis="y", alpha=0.3)

for bar, val in zip(bars1, hota1):
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val:.1f}", ha="center", fontsize=5.5, color=C_ARM1)
for bar, val in zip(bars2, hota2):
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{val:.1f}", ha="center", fontsize=5.5, color=C_ARM2)

# Compact table
ax_tbl = fig.add_subplot(gs[1])
ax_tbl.axis("off")
col_labels = ["Epoch","DetA","AssA","MOTA","IDF1"]
t1_data = [[str(r["epochs_completed"]),f"{r['DetA']:.1f}",f"{r['AssA']:.1f}",
            f"{r['MOTA']:.1f}",f"{r['IDF1']:.1f}"] for r in a1]
t2_data = [[str(r["epochs_completed"]),f"{r['DetA']:.1f}",f"{r['AssA']:.1f}",
            f"{r['MOTA']:.1f}",f"{r['IDF1']:.1f}"] for r in a2]

combined = [["","Arm1","","","","","Arm2","","",""]]
# Build side by side table
cell_text = []
for i in range(len(a1)):
    row_t = t1_data[i] + t2_data[i] if i < len(t2_data) else t1_data[i] + ["","","","",""]
    cell_text.append(row_t)
col_hdrs = [f"Ep","DetA","AssA","MOTA","IDF1","Ep","DetA","AssA","MOTA","IDF1"]
tbl = ax_tbl.table(cellText=cell_text, colLabels=col_hdrs,
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(6)
tbl.scale(1, 1.1)
# Colour header
for j in range(5):
    tbl[(0,j)].set_facecolor(C_ARM1); tbl[(0,j)].set_text_props(color="white")
    tbl[(0,j+5)].set_facecolor(C_ARM2); tbl[(0,j+5)].set_text_props(color="white")
# Highlight best rows
for i, r in enumerate(a1):
    if r["epochs_completed"] == best1["epochs_completed"]:
        for j in range(5): tbl[(i+1,j)].set_facecolor("#D0E4FF")
for i, r in enumerate(a2):
    if r["epochs_completed"] == best2["epochs_completed"]:
        for j in range(5): tbl[(i+1,j+5)].set_facecolor("#FFD0C0")

add_footer(fig)
fig.savefig(os.path.join(FIG_DIR,"fig07_epochwise_summary.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR,"fig07_epochwise_summary.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Saved: fig07_epochwise_summary.png + .pdf")

# ─────────────────────────────────────────────────────────────────────────────
# fig08: Qualitative — frames 298 and 447, 3 columns
# ─────────────────────────────────────────────────────────────────────────────
import cv2
try:
    from collections import defaultdict as ddict

    def load_tracker_dict(path):
        bd = ddict(list)
        if not os.path.exists(path): return bd
        with open(path) as f:
            for line in f:
                c = line.strip().split(",")
                if len(c) >= 6:
                    bd[int(c[0])].append((float(c[2]),float(c[3]),float(c[4]),float(c[5])))
        return bd

    def load_gt_dict(path):
        bd = ddict(list)
        with open(path) as f:
            for line in f:
                c = line.strip().split(",")
                if len(c) >= 6:
                    bd[int(c[0])].append((float(c[2]),float(c[3]),float(c[4]),float(c[5])))
        return bd

    GT_PATH = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
    CANDS8 = [
        ("Phase5",  "results/eval_iotd_val/phase5/seq_02_30sec_a.txt",  "Phase 5",  C_PHASE5),
        ("Arm1ck3", "results/eval_iotd_val/arm1_ckpt3/seq_02_30sec_a.txt","Arm 1 (ckpt3)",C_ARM1),
        ("Arm2ck1", "results/eval_iotd_val/arm2_ckpt1/seq_02_30sec_a.txt","Arm 2 (ckpt1)",C_ARM2),
    ]
    FRAMES8 = [298, 447]
    gt_dict = load_gt_dict(GT_PATH)
    trk_dicts = [(label, load_tracker_dict(path), color) for label, path, label2, color in
                 [(c[0],c[1],c[2],c[3]) for c in CANDS8]]
    cand_labels8 = [c[2] for c in CANDS8]

    fig, axes = plt.subplots(len(FRAMES8), 3, figsize=(7.0, 4.0))
    for ri, frame_num in enumerate(FRAMES8):
        img_path = os.path.join(IMG_DIR, f"{frame_num:06d}.jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            for ci in range(3):
                axes[ri,ci].text(0.5,0.5,"Image not found",ha="center")
                axes[ri,ci].axis("off")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_boxes = gt_dict.get(frame_num, [])

        for ci, (run_id, trk_dict, color) in enumerate(trk_dicts):
            ax = axes[ri, ci]
            disp = img_rgb.copy()
            # Draw GT (green)
            for gx,gy,gw,gh in gt_boxes:
                disp = cv2.rectangle(disp, (int(gx),int(gy)),(int(gx+gw),int(gy+gh)),(0,200,0),2)
            # Draw tracker (run colour)
            c255 = tuple(int(c*255) for c in matplotlib.colors.to_rgb(color))
            for tx,ty,tw,th in trk_dict.get(frame_num,[]):
                disp = cv2.rectangle(disp,(int(tx),int(ty)),(int(tx+tw),int(ty+th)),c255,2)
            ax.imshow(disp); ax.axis("off")
            if ri == 0:
                ax.set_title(cand_labels8[ci], fontsize=7, color=color, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"f{frame_num}", fontsize=7, rotation=0, labelpad=25, va="center")

    # Legend
    green_p = mpatches.Patch(color="#00C800", label="GT")
    leg_patches = [green_p] + [mpatches.Patch(color=c, label=l) for l,_,l,c in CANDS8]
    fig.legend(handles=[green_p]+[mpatches.Patch(color=CANDS8[i][3],label=cand_labels8[i]) for i in range(3)],
               loc="lower center", ncol=4, fontsize=6, framealpha=0.8)
    add_footer(fig)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(FIG_DIR,"fig08_qualitative.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR,"fig08_qualitative.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig08_qualitative.png + .pdf")
except Exception as e:
    print(f"  fig08 SKIPPED (optional): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: fig01-fig07 in a labelled grid
# ─────────────────────────────────────────────────────────────────────────────
print("  Building dashboard...")
from PIL import Image
import PIL
PANEL_ORDER = [
    "fig01_hota_vs_steps","fig02_deta_assa_vs_steps",
    "fig03_fp_fn_vs_steps","fig04_detection_outcomes",
    "fig05_candidate_comparison","fig06_identity_metrics",
    "fig07_epochwise_summary",
]
LABELS = list("ABCDEFG")

# Load images
panels = []
for name in PANEL_ORDER:
    p = os.path.join(FIG_DIR, f"{name}.png")
    if os.path.exists(p):
        panels.append((Image.open(p), name))

# Dashboard 4 + 3 grid at 200 dpi equivalent
DASH_W, DASH_H = 4800, 3200  # 24x16 at 200dpi
dash = Image.new("RGB", (DASH_W, DASH_H), (255,255,255))

# Layout: 3 cols, 3 rows; last panel spans col 0-1
positions = [
    (0, 0, DASH_W//3, DASH_H//3),        # A
    (DASH_W//3, 0, DASH_W//3, DASH_H//3),# B
    (2*DASH_W//3, 0, DASH_W//3, DASH_H//3), # C
    (0, DASH_H//3, DASH_W//3, DASH_H//3),# D
    (DASH_W//3, DASH_H//3, DASH_W//3, DASH_H//3), # E
    (2*DASH_W//3, DASH_H//3, DASH_W//3, DASH_H//3), # F
    (0, 2*DASH_H//3, 2*DASH_W//3, DASH_H//3), # G (wide)
]

for idx, ((img, name), (x, y, cell_w, cell_h)) in enumerate(zip(panels, positions)):
    img_r = img.resize((cell_w-20, cell_h-30), PIL.Image.LANCZOS)
    dash.paste(img_r, (x+10, y+30))
    # Label
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(dash)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    draw.text((x+15, y+5), f"({LABELS[idx]})", fill=(30,30,30), font=font)

# Status box in last cell
sx = 2*DASH_W//3; sy = 2*DASH_H//3
cell_w2 = DASH_W//3; cell_h2 = DASH_H//3
draw = ImageDraw.Draw(dash)
try: font_s = ImageFont.truetype("arial.ttf", 32)
except: font_s = ImageFont.load_default()
draw.rectangle([sx+10,sy+10,sx+cell_w2-10,sy+cell_h2-10], fill=(245,250,255), outline=(180,180,180),width=2)
status_lines = [
    "(H) Status Box",
    f"Phase5 baseline: HOTA {ph5_hota:.2f}",
    f"Arm1 best:  ckpt3 HOTA 50.47",
    f"Arm2 best:  ckpt1 HOTA 48.17",
    "",
    "Threshold sweep: 0.5 only (best ckpts)",
    "0.6: NOT RUN",
    "0.7/0.8: ckpt4 only (exploratory)",
    "",
    "Missing before tuning:",
    "- Seed repeats (GATE 2)",
    "- LR sensitivity (1e-5 vs 2e-5)",
    "- C1 calibration",
    "- Full threshold sweep on best ckpts",
]
for li, line in enumerate(status_lines):
    color = (0,80,160) if li==0 else ((180,0,0) if "NOT RUN" in line or "Missing" in line else (40,40,40))
    draw.text((sx+25, sy+15+li*38), line, fill=color, font=font_s)

dash.save(os.path.join(FIG_DIR, "pre_tuning_dashboard.png"))
print("  Saved: pre_tuning_dashboard.png")

print("\nStep B DONE — all figures written to", FIG_DIR)
