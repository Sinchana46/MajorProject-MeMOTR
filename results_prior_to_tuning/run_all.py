#!/usr/bin/env python3
"""
run_all.py  —  Pre-tuning baseline v1 (prior to hyperparameter tuning)
Run this script from the PROJECT ROOT (D:/MajorProject-MeMOTR) like this:

    python results_prior_to_tuning/run_all.py

It will regenerate ALL figures (PNG + PDF), ALL tables (CSV + LaTeX + Markdown),
and ALL data CSVs from the saved TrackEval evaluation results.
Nothing is trained or re-evaluated — only existing summaries are read.
"""
import os, sys, csv, json, hashlib, datetime, zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ─── Paths ────────────────────────────────────────────────────────────────────
# All paths relative to project root (where you run this script from)
OUT_DIR      = "results_prior_to_tuning"
FIG_DIR      = os.path.join(OUT_DIR, "figures")
TABLE_DIR    = os.path.join(OUT_DIR, "tables")
DATA_DIR     = os.path.join(OUT_DIR, "data")
GT_PATH      = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
IMG_DIR      = "data/IOTD_val/seq_02_30sec_a/img1"
TRACKER_BASE = "results/eval_iotd_val"

for d in [FIG_DIR, TABLE_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
FOOTER = ("Validation: seq_02_30sec_a (744 frames, 50 tracks, night). "
          "Single run per setting, seed 42. Threshold 0.5. Not statistically established.")

C_PHASE5, C_ARM1, C_ARM2 = "#999999", "#0072B2", "#D55E00"
M_PHASE5, M_ARM1, M_ARM2 = "s", "o", "^"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 300,
    "pdf.fonttype": 42,
})

def add_footer(fig):
    fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom",
             fontsize=5.5, color="#555555", style="italic")

def savefig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    dpi=300 if ext=="png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}.png + .pdf")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def read_te_summary(path):
    """Read a TrackEval pedestrian_summary.txt (header + values on two lines)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        return None
    d = dict(zip(lines[0].split(), lines[1].split()))
    return {
        "HOTA": float(d.get("HOTA",   0)),
        "DetA": float(d.get("DetA",   0)),
        "AssA": float(d.get("AssA",   0)),
        "MOTA": float(d.get("MOTA",   0)),
        "IDF1": float(d.get("IDF1",   0)),
        "IDSW": float(d.get("IDSW",   0)),
        "FP":   float(d.get("CLR_FP", 0)),
        "FN":   float(d.get("CLR_FN", 0)),
        "TP":   float(d.get("CLR_TP", 0)),
        "Recall": float(d.get("CLR_Re", 0)),
        "Prec":   float(d.get("CLR_Pr", 0)),
    }

def write_table(name, headers, rows, caption, label):
    """Write .csv, .tex (booktabs), and .md for a table."""
    csv_p = os.path.join(TABLE_DIR, f"{name}.csv")
    tex_p = os.path.join(TABLE_DIR, f"{name}.tex")
    md_p  = os.path.join(TABLE_DIR, f"{name}.md")

    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

    with open(md_p, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(str(h) for h in headers) + " |\n")
        f.write("| " + " | ".join(["---"]*len(headers)) + " |\n")
        for row in rows:
            md_row = [str(c).replace("\\textbf{","").replace("}","") for c in row]
            f.write("| " + " | ".join(md_row) + " |\n")

    col_fmt = "l" + "r"*(len(headers)-1)
    with open(tex_p, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n\\toprule\n")
        f.write(" & ".join(str(h) for h in headers) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(c) for c in row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  [table] {name}  (.csv .tex .md)")

# ─── Summary file locations ───────────────────────────────────────────────────
SUMMARIES = {
    ("Phase5","ckpt_4",0.5): f"{TRACKER_BASE}/trackeval_phase5/trackeval_output/PHASE5/pedestrian_summary.txt",
    ("Arm1",  "ckpt_0",0.5): f"{TRACKER_BASE}/trackeval_arm1_ckpt0/trackeval_output/ARM1_CKPT0/pedestrian_summary.txt",
    ("Arm1",  "ckpt_1",0.5): f"{TRACKER_BASE}/trackeval_arm1_ckpt1/trackeval_output/ARM1_CKPT1/pedestrian_summary.txt",
    ("Arm1",  "ckpt_2",0.5): f"{TRACKER_BASE}/trackeval_arm1_ckpt2/trackeval_output/ARM1_CKPT2/pedestrian_summary.txt",
    ("Arm1",  "ckpt_3",0.5): f"{TRACKER_BASE}/trackeval_arm1_ckpt3/trackeval_output/ARM1_CKPT3/pedestrian_summary.txt",
    ("Arm1",  "ckpt_4",0.5): f"{TRACKER_BASE}/trackeval_arm1/trackeval_output/ARM1/pedestrian_summary.txt",
    ("Arm2",  "ckpt_0",0.5): f"{TRACKER_BASE}/trackeval_arm2_ckpt0/trackeval_output/ARM2_CKPT0/pedestrian_summary.txt",
    ("Arm2",  "ckpt_1",0.5): f"{TRACKER_BASE}/trackeval_arm2_ckpt1/trackeval_output/ARM2_CKPT1/pedestrian_summary.txt",
    ("Arm2",  "ckpt_2",0.5): f"{TRACKER_BASE}/trackeval_arm2_ckpt2/trackeval_output/ARM2_CKPT2/pedestrian_summary.txt",
    ("Arm2",  "ckpt_3",0.5): f"{TRACKER_BASE}/trackeval_arm2_ckpt3/trackeval_output/ARM2_CKPT3/pedestrian_summary.txt",
    ("Arm2",  "ckpt_4",0.5): f"{TRACKER_BASE}/trackeval_arm2/trackeval_output/ARM2/pedestrian_summary.txt",
    ("Phase5","ckpt_4",0.7): f"{TRACKER_BASE}/diag/thresh_0p7/trackeval_phase5/trackeval_output/PHASE5_T0P7/pedestrian_summary.txt",
    ("Arm1",  "ckpt_4",0.7): f"{TRACKER_BASE}/diag/thresh_0p7/trackeval_arm1/trackeval_output/ARM1_T0P7/pedestrian_summary.txt",
    ("Arm2",  "ckpt_4",0.7): f"{TRACKER_BASE}/diag/thresh_0p7/trackeval_arm2/trackeval_output/ARM2_T0P7/pedestrian_summary.txt",
    ("Phase5","ckpt_4",0.8): f"{TRACKER_BASE}/diag/thresh_0p8/trackeval_phase5/trackeval_output/PHASE5_T0P8/pedestrian_summary.txt",
    ("Arm1",  "ckpt_4",0.8): f"{TRACKER_BASE}/diag/thresh_0p8/trackeval_arm1/trackeval_output/ARM1_T0P8/pedestrian_summary.txt",
    ("Arm2",  "ckpt_4",0.8): f"{TRACKER_BASE}/diag/thresh_0p8/trackeval_arm2/trackeval_output/ARM2_T0P8/pedestrian_summary.txt",
}

STEPS_MAP = {
    ("Phase5","ckpt_4"): 0,
    ("Arm1","ckpt_0"):2528,("Arm1","ckpt_1"):5056,("Arm1","ckpt_2"):7584,
    ("Arm1","ckpt_3"):10112,("Arm1","ckpt_4"):12640,
    ("Arm2","ckpt_0"):10112,("Arm2","ckpt_1"):20224,("Arm2","ckpt_2"):30336,
    ("Arm2","ckpt_3"):40448,("Arm2","ckpt_4"):50560,
}
EPOCHS_MAP = {
    ("Phase5","ckpt_4"):0,
    ("Arm1","ckpt_0"):1,("Arm1","ckpt_1"):2,("Arm1","ckpt_2"):3,("Arm1","ckpt_3"):4,("Arm1","ckpt_4"):5,
    ("Arm2","ckpt_0"):1,("Arm2","ckpt_1"):2,("Arm2","ckpt_2"):3,("Arm2","ckpt_3"):4,("Arm2","ckpt_4"):5,
}

# ─── STEP A: Load all data ────────────────────────────────────────────────────
print("="*60)
print("STEP A: Loading evaluation summaries")
print("="*60)

GT_SHA = sha256_file(GT_PATH)
print(f"GT SHA256: {GT_SHA}")

all_data = {}
missing  = []
for (run, ckpt, thresh), path in SUMMARIES.items():
    d = read_te_summary(path)
    if d:
        all_data[(run,ckpt,thresh)] = {**d, "run":run,"checkpoint":ckpt,"threshold":thresh,
            "steps":STEPS_MAP.get((run,ckpt),0), "epochs":EPOCHS_MAP.get((run,ckpt),0)}
        print(f"  OK  {run:<8} {ckpt:<8} thresh={thresh} -> HOTA={d['HOTA']:.2f}")
    else:
        missing.append((run,ckpt,thresh,path))
        print(f"  MISSING  {run} {ckpt} thresh={thresh}  {path}")

# Build reference lists
pc_data = [v for (r,c,t),v in all_data.items() if t==0.5]
pc_data.sort(key=lambda x: ({"Phase5":0,"Arm1":1,"Arm2":2}.get(x["run"],9), x["steps"]))
ph = [r for r in pc_data if r["run"]=="Phase5"]
a1 = [r for r in pc_data if r["run"]=="Arm1"]
a2 = [r for r in pc_data if r["run"]=="Arm2"]
best1 = max(a1, key=lambda r: r["HOTA"])
best2 = max(a2, key=lambda r: r["HOTA"])
ph5_hota = ph[0]["HOTA"] if ph else 50.51

print(f"\nPhase5 HOTA={ph5_hota:.2f} | Arm1 best={best1['checkpoint']} HOTA={best1['HOTA']:.2f} | Arm2 best={best2['checkpoint']} HOTA={best2['HOTA']:.2f}")

# ─── Write per_checkpoint_metrics.csv ─────────────────────────────────────────
now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
pc_cols = ["run","checkpoint","steps","epochs","threshold","HOTA","DetA","AssA",
           "MOTA","IDF1","IDSW","TP","FP","FN","Recall","Prec","GT_sha256","summary_path"]
pc_rows_csv = []
for r in pc_data:
    key = (r["run"],r["checkpoint"],0.5)
    pc_rows_csv.append([r["run"],r["checkpoint"],r["steps"],r["epochs"],0.5,
        round(r["HOTA"],3),round(r["DetA"],3),round(r["AssA"],3),round(r["MOTA"],3),
        round(r["IDF1"],3),int(r["IDSW"]),int(r["TP"]),int(r["FP"]),int(r["FN"]),
        round(r["Recall"],3),round(r["Prec"],3),GT_SHA,SUMMARIES.get(key,"")])
csv_path = os.path.join(DATA_DIR,"per_checkpoint_metrics.csv")
with open(csv_path,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(pc_cols); w.writerows(pc_rows_csv)
print(f"\n  [data] per_checkpoint_metrics.csv ({len(pc_rows_csv)} rows)")

# ─── Write results_registry.csv ───────────────────────────────────────────────
reg_cols = pc_cols + ["registry_version","created_utc"]
reg_rows_csv = []
for (run,ckpt,thresh), r in all_data.items():
    reg_rows_csv.append([run,ckpt,r["steps"],r["epochs"],thresh,
        round(r["HOTA"],3),round(r["DetA"],3),round(r["AssA"],3),round(r["MOTA"],3),
        round(r["IDF1"],3),int(r["IDSW"]),int(r["TP"]),int(r["FP"]),int(r["FN"]),
        round(r["Recall"],3),round(r["Prec"],3),GT_SHA,SUMMARIES.get((run,ckpt,thresh),""),
        "v1",now_utc])
reg_path = os.path.join(DATA_DIR,"results_registry.csv")
with open(reg_path,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(reg_cols); w.writerows(reg_rows_csv)
print(f"  [data] results_registry.csv ({len(reg_rows_csv)} rows)")

# ─── Write detection_outcomes.csv (own matching) ──────────────────────────────
from collections import defaultdict
CAND_TRACKER_PATHS = {
    ("Phase5","ckpt_4"): f"{TRACKER_BASE}/phase5/seq_02_30sec_a.txt",
    ("Arm1","ckpt_3"):   f"{TRACKER_BASE}/arm1_ckpt3/seq_02_30sec_a.txt",
    ("Arm2","ckpt_1"):   f"{TRACKER_BASE}/arm2_ckpt1/seq_02_30sec_a.txt",
}

def load_mot(path):
    bd = defaultdict(list)
    if not os.path.exists(path): return bd
    with open(path) as f:
        for line in f:
            c = line.strip().split(",")
            if len(c)>=6:
                bd[int(c[0])].append((float(c[2]),float(c[3]),float(c[4]),float(c[5])))
    return bd

def iou(a,b):
    ax2,ay2 = a[0]+a[2],a[1]+a[3]; bx2,by2 = b[0]+b[2],b[1]+b[3]
    ix=max(0,min(ax2,bx2)-max(a[0],b[0])); iy=max(0,min(ay2,by2)-max(a[1],b[1]))
    inter=ix*iy; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.0

gt_by_frame = load_mot(GT_PATH)

det_rows_out = []
for (run,ckpt), trkpath in CAND_TRACKER_PATHS.items():
    trk_dict = load_mot(trkpath)
    tp=fp=fn=0
    for frame in set(list(gt_by_frame)+list(trk_dict)):
        preds = trk_dict.get(frame,[]); gts = gt_by_frame.get(frame,[])
        matched_p=set(); matched_g=set()
        pairs=sorted([(iou(p,g),pi,gi) for pi,p in enumerate(preds) for gi,g in enumerate(gts) if iou(p,g)>=0.5],reverse=True)
        for _,pi,gi in pairs:
            if pi not in matched_p and gi not in matched_g:
                matched_p.add(pi); matched_g.add(gi); tp+=1
        fp+=len(preds)-len(matched_p); fn+=len(gts)-len(matched_g)
    te = all_data.get((run,ckpt,0.5),{})
    det_rows_out.append({"run":run,"checkpoint":ckpt,"TP_own":tp,"FP_own":fp,"FN_own":fn,
        "recall_own":round(tp/(tp+fn)*100,3) if (tp+fn)>0 else 0,
        "precision_own":round(tp/(tp+fp)*100,3) if (tp+fp)>0 else 0,
        "TP_te":int(te.get("TP",0)),"FP_te":int(te.get("FP",0)),"FN_te":int(te.get("FN",0)),
        "dTP":tp-int(te.get("TP",0)),"dFP":fp-int(te.get("FP",0)),"dFN":fn-int(te.get("FN",0))})
    print(f"  [det]  {run} {ckpt}: TP={tp} FP={fp} FN={fn} (delta vs TrackEval: dTP={tp-int(te.get('TP',0))})")

det_path = os.path.join(DATA_DIR,"detection_outcomes.csv")
with open(det_path,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(det_rows_out[0].keys()))
    w.writeheader(); w.writerows(det_rows_out)
print(f"  [data] detection_outcomes.csv")

# ─── STEP B: FIGURES ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP B: Generating figures")
print("="*60)

# fig01: HOTA vs steps
fig,ax = plt.subplots(1,1,figsize=(7.0,3.5))
ax.axhline(ph5_hota,color=C_PHASE5,ls="--",lw=1.5,label=f"Phase 5 ({ph5_hota:.2f})")
ax.plot([r["steps"] for r in a1],[r["HOTA"] for r in a1],color=C_ARM1,marker=M_ARM1,lw=1.5,ms=6,label="Arm 1 (real only)")
ax.plot([r["steps"] for r in a2],[r["HOTA"] for r in a2],color=C_ARM2,marker=M_ARM2,lw=1.5,ms=6,label="Arm 2 (real + aug.)")
for r in a1: ax.annotate(f"{r['HOTA']:.1f}",(r["steps"],r["HOTA"]),textcoords="offset points",xytext=(0,5),fontsize=6,color=C_ARM1,ha="center")
for r in a2: ax.annotate(f"{r['HOTA']:.1f}",(r["steps"],r["HOTA"]),textcoords="offset points",xytext=(0,-10),fontsize=6,color=C_ARM2,ha="center")
ax.scatter([best1["steps"]],[best1["HOTA"]],color=C_ARM1,marker="*",s=180,zorder=5)
ax.scatter([best2["steps"]],[best2["HOTA"]],color=C_ARM2,marker="*",s=180,zorder=5)
ax.axvline(10112,color=C_ARM1,ls=":",lw=1.0,alpha=0.7); ax.text(10412,36.0,"LR drop\n(Arm1)",fontsize=5.5,color=C_ARM1)
ax.axvline(40448,color=C_ARM2,ls=":",lw=1.0,alpha=0.7); ax.text(40748,36.0,"LR drop\n(Arm2)",fontsize=5.5,color=C_ARM2)
a1_eq = next(r for r in a1 if r["steps"]==10112); a2_eq = next(r for r in a2 if r["steps"]==10112)
ax.annotate("",xy=(a2_eq["steps"],a2_eq["HOTA"]),xytext=(a1_eq["steps"],a1_eq["HOTA"]),arrowprops=dict(arrowstyle="<->",color="black",lw=1.0))
ax.text(10612,(a1_eq["HOTA"]+a2_eq["HOTA"])/2,f"Equal steps\n{a1_eq['HOTA']:.1f} vs {a2_eq['HOTA']:.1f}",fontsize=5.5)
ax.set_xlabel("Optimizer steps"); ax.set_ylabel("HOTA (%)"); ax.legend(loc="lower right",framealpha=0.85)
ax.set_xlim(-500,53000); ax.set_ylim(35,55); ax.grid(True,alpha=0.3)
add_footer(fig); fig.tight_layout(rect=[0,0.04,1,1]); savefig(fig,"fig01_hota_vs_steps")

# fig02: DetA + AssA
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(3.4,4.5),sharex=True)
for ax,metric,ph_val in [(ax1,"DetA",ph[0]["DetA"]),(ax2,"AssA",ph[0]["AssA"])]:
    ax.axhline(ph_val,color=C_PHASE5,ls="--",lw=1.2,label=f"Phase5 ({ph_val:.1f})")
    ax.plot([r["steps"] for r in a1],[r[metric] for r in a1],color=C_ARM1,marker=M_ARM1,lw=1.2,ms=4,label="Arm1")
    ax.plot([r["steps"] for r in a2],[r[metric] for r in a2],color=C_ARM2,marker=M_ARM2,lw=1.2,ms=4,label="Arm2")
    ax.set_ylabel(f"{metric} (%)"); ax.grid(True,alpha=0.3); ax.legend(fontsize=6,loc="lower right")
ax2.set_xlabel("Optimizer steps"); add_footer(fig); fig.tight_layout(rect=[0,0.04,1,1]); savefig(fig,"fig02_deta_assa_vs_steps")

# fig03: FP FN Recall Precision
fig,axes = plt.subplots(2,2,figsize=(7.0,4.5))
for metric,ax,ph_val in [("FP",axes[0,0],ph[0]["FP"]),("FN",axes[0,1],ph[0]["FN"]),("Recall",axes[1,0],ph[0]["Recall"]),("Prec",axes[1,1],ph[0]["Prec"])]:
    label = {"FP":"False Positives","FN":"False Negatives","Recall":"Recall (%)","Prec":"Precision (%)"}[metric]
    ax.axhline(ph_val,color=C_PHASE5,ls="--",lw=1.2,label=f"Phase5 ({ph_val:.0f})" if metric in("FP","FN") else f"Phase5 ({ph_val:.1f})")
    ax.plot([r["steps"] for r in a1],[r[metric] for r in a1],color=C_ARM1,marker=M_ARM1,lw=1.2,ms=4,label="Arm1")
    ax.plot([r["steps"] for r in a2],[r[metric] for r in a2],color=C_ARM2,marker=M_ARM2,lw=1.2,ms=4,label="Arm2")
    ax.set_xlabel("Steps"); ax.set_ylabel(label); ax.legend(fontsize=6); ax.grid(True,alpha=0.3)
add_footer(fig); fig.tight_layout(rect=[0,0.04,1,1]); savefig(fig,"fig03_fp_fn_vs_steps")

# fig04: Detection outcome 2x2 grids
fig,big_axes = plt.subplots(1,3,figsize=(7.0,2.8))
CAND4 = [("Phase5","ckpt_4","Phase 5",C_PHASE5),("Arm1","ckpt_3","Arm 1 (ckpt3)",C_ARM1),("Arm2","ckpt_1","Arm 2 (ckpt1)",C_ARM2)]
for big_ax,(run,ckpt,label,color) in zip(big_axes,CAND4):
    big_ax.axis("off")
    det = next((r for r in det_rows_out if r["run"]==run and r["checkpoint"]==ckpt),None)
    if det is None: continue
    gs = big_ax.get_position()
    inset = fig.add_axes([gs.x0,gs.y0,gs.width,gs.height]); inset.axis("off")
    cells = [("TP",det["TP_own"],color,"white"),("FP",det["FP_own"],"#E69F00","white"),
             ("FN",det["FN_own"],"#56B4E9","white"),("TN\n(N/A)","single-class","#D0D0D0","#777777")]
    for idx,(lbl,val,bg,fg) in enumerate(cells):
        xi=idx%2; yi=idx//2
        rect = mpatches.FancyBboxPatch((xi*0.5,(1-yi)*0.5-0.48),0.46,0.44,boxstyle="round,pad=0.02",fc=bg,ec="white",lw=1,transform=inset.transAxes,clip_on=False)
        inset.add_patch(rect)
        inset.text(xi*0.5+0.23,(1-yi)*0.5-0.48+0.22,f"{lbl}\n{val}",transform=inset.transAxes,ha="center",va="center",color=fg,fontsize=7.5,fontweight="bold")
    inset.text(0.5,-0.06,f"{label}\nRec={det['recall_own']:.1f}% Prec={det['precision_own']:.1f}%",transform=inset.transAxes,ha="center",fontsize=7,color=color)
add_footer(fig); fig.tight_layout(rect=[0,0.06,1,1]); savefig(fig,"fig04_detection_outcomes")

# fig05: Grouped bars candidates
fig,(ax_m,ax_mota) = plt.subplots(1,2,figsize=(7.0,3.0),gridspec_kw={"width_ratios":[3,1]})
m5 = ["HOTA","DetA","AssA","IDF1"]; x=np.arange(len(m5)); bw=0.25
for bi,(run,ckpt,clabel,color) in enumerate(CAND4):
    r = all_data.get((run,ckpt,0.5))
    if r is None: continue
    vals = [r[m] for m in m5]
    bars = ax_m.bar(x+bi*bw,vals,bw,label=clabel,color=color,alpha=0.85)
    for bar,val in zip(bars,vals): ax_m.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{val:.1f}",ha="center",fontsize=5.5)
ax_m.set_xticks(x+bw); ax_m.set_xticklabels(m5); ax_m.set_ylabel("Metric (%)"); ax_m.set_ylim(0,80); ax_m.legend(fontsize=6); ax_m.grid(axis="y",alpha=0.3)
mota_vals = [(c[2].replace("\n"," "),all_data.get((c[0],c[1],0.5),{}).get("MOTA",0),c[3]) for c in CAND4]
ax_mota.bar([m[0] for m in mota_vals],[m[1] for m in mota_vals],color=[m[2] for m in mota_vals],alpha=0.85)
for i,(lbl,val,_) in enumerate(mota_vals): ax_mota.text(i,val+(0.5 if val>=0 else -1.5),f"{val:.1f}",ha="center",fontsize=5.5)
ax_mota.axhline(0,color="black",lw=0.8); ax_mota.set_ylabel("MOTA (%)"); ax_mota.set_title("MOTA"); ax_mota.grid(axis="y",alpha=0.3)
add_footer(fig); fig.tight_layout(rect=[0,0.04,1,1]); savefig(fig,"fig05_candidate_comparison")

# fig06: Identity metrics
fig,axes = plt.subplots(1,3,figsize=(7.0,2.8))
for ax,(metric,label) in zip(axes,[("AssA","AssA (%)"),("IDF1","IDF1 (%)"),("IDSW","ID Switches")]):
    vals = [all_data.get((r,c,0.5),{}).get(metric,0) for r,c,_,_ in CAND4]
    cols = [CAND4[i][3] for i in range(3)]
    clabels = [CAND4[i][2] for i in range(3)]
    bars = ax.bar(clabels,vals,color=cols,alpha=0.85)
    best_idx = vals.index(min(vals) if metric=="IDSW" else max(vals))
    bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(2.5)
    for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"{val:.1f}" if metric!="IDSW" else str(int(val)),ha="center",fontsize=6)
    ax.set_ylabel(label); ax.grid(axis="y",alpha=0.3); ax.set_title(label)
add_footer(fig); fig.tight_layout(rect=[0,0.04,1,1]); savefig(fig,"fig06_identity_metrics")

# fig07: Epoch-wise grouped bars + table
fig = plt.figure(figsize=(7.0,4.5))
gs2 = gridspec.GridSpec(2,1,height_ratios=[2.5,1.5],hspace=0.4)
ax_bar = fig.add_subplot(gs2[0])
bw=0.35; x=np.arange(len(a1))
bars1 = ax_bar.bar(x-bw/2,[r["HOTA"] for r in a1],bw,label="Arm 1",color=C_ARM1,alpha=0.85)
bars2 = ax_bar.bar(x+bw/2,[r["HOTA"] for r in a2],bw,label="Arm 2",color=C_ARM2,alpha=0.85)
b1i = [r["HOTA"] for r in a1].index(max(r["HOTA"] for r in a1))
b2i = [r["HOTA"] for r in a2].index(max(r["HOTA"] for r in a2))
bars1[b1i].set_edgecolor("gold"); bars1[b1i].set_linewidth(2.5)
bars2[b2i].set_edgecolor("gold"); bars2[b2i].set_linewidth(2.5)
ax_bar.axhline(ph5_hota,color=C_PHASE5,ls="--",lw=1.2,label=f"Phase5 ({ph5_hota:.1f})")
ax_bar.set_xticks(x); ax_bar.set_xticklabels([f"Ep{r['epochs']}" for r in a1])
ax_bar.set_ylabel("HOTA (%)"); ax_bar.set_ylim(35,55); ax_bar.legend(fontsize=6); ax_bar.grid(axis="y",alpha=0.3)
for bar,r in zip(bars1,a1): ax_bar.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"{r['HOTA']:.1f}",ha="center",fontsize=5.5,color=C_ARM1)
for bar,r in zip(bars2,a2): ax_bar.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f"{r['HOTA']:.1f}",ha="center",fontsize=5.5,color=C_ARM2)
ax_tbl = fig.add_subplot(gs2[1]); ax_tbl.axis("off")
cell_text = [[str(a1[i]["epochs"]),f"{a1[i]['DetA']:.1f}",f"{a1[i]['AssA']:.1f}",f"{a1[i]['MOTA']:.1f}",f"{a1[i]['IDF1']:.1f}",
              str(a2[i]["epochs"]),f"{a2[i]['DetA']:.1f}",f"{a2[i]['AssA']:.1f}",f"{a2[i]['MOTA']:.1f}",f"{a2[i]['IDF1']:.1f}"] for i in range(len(a1))]
tbl = ax_tbl.table(cellText=cell_text,colLabels=["Ep","DetA","AssA","MOTA","IDF1"]*2,loc="center",cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(6); tbl.scale(1,1.1)
for j in range(5): tbl[(0,j)].set_facecolor(C_ARM1); tbl[(0,j)].set_text_props(color="white")
for j in range(5): tbl[(0,j+5)].set_facecolor(C_ARM2); tbl[(0,j+5)].set_text_props(color="white")
for i in range(len(a1)):
    if i==b1i: [tbl[(i+1,j)].set_facecolor("#D0E4FF") for j in range(5)]
    if i==b2i: [tbl[(i+1,j+5)].set_facecolor("#FFD0C0") for j in range(5)]
add_footer(fig)
fig.savefig(os.path.join(FIG_DIR,"fig07_epochwise_summary.png"),dpi=300,bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR,"fig07_epochwise_summary.pdf"),bbox_inches="tight")
plt.close(fig); print("  [fig] fig07_epochwise_summary.png + .pdf")

# fig08: Qualitative
try:
    import cv2
    TRKS8 = [("Phase 5",f"{TRACKER_BASE}/phase5/seq_02_30sec_a.txt",C_PHASE5),
             ("Arm 1 (ckpt3)",f"{TRACKER_BASE}/arm1_ckpt3/seq_02_30sec_a.txt",C_ARM1),
             ("Arm 2 (ckpt1)",f"{TRACKER_BASE}/arm2_ckpt1/seq_02_30sec_a.txt",C_ARM2)]
    fig,axes = plt.subplots(2,3,figsize=(7.0,4.0))
    for ri,frame_num in enumerate([298,447]):
        img = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR,f"{frame_num:06d}.jpg")),cv2.COLOR_BGR2RGB)
        for ci,(label,trkpath,color) in enumerate(TRKS8):
            trk = load_mot(trkpath)
            disp = img.copy()
            c255 = tuple(int(x*255) for x in matplotlib.colors.to_rgb(color))
            for gx,gy,gw,gh in gt_by_frame.get(frame_num,[]): cv2.rectangle(disp,(int(gx),int(gy)),(int(gx+gw),int(gy+gh)),(0,200,0),2)
            for tx,ty,tw,th in trk.get(frame_num,[]): cv2.rectangle(disp,(int(tx),int(ty)),(int(tx+tw),int(ty+th)),c255,2)
            ax=axes[ri,ci]; ax.imshow(disp); ax.axis("off")
            if ri==0: ax.set_title(label,fontsize=7,color=color,fontweight="bold")
            if ci==0: ax.set_ylabel(f"f{frame_num}",fontsize=7,rotation=0,labelpad=25,va="center")
    fig.legend(handles=[mpatches.Patch(color="#00C800",label="GT")]+[mpatches.Patch(color=t[2],label=t[0]) for t in TRKS8],
               loc="lower center",ncol=4,fontsize=6)
    add_footer(fig); fig.tight_layout(rect=[0,0.06,1,1])
    fig.savefig(os.path.join(FIG_DIR,"fig08_qualitative.png"),dpi=300,bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR,"fig08_qualitative.pdf"),bbox_inches="tight")
    plt.close(fig); print("  [fig] fig08_qualitative.png + .pdf")
except Exception as e:
    print(f"  [fig] fig08_qualitative SKIPPED (optional, needs cv2+frames): {e}")

# Dashboard
try:
    from PIL import Image, ImageDraw, ImageFont
    panels = [Image.open(os.path.join(FIG_DIR,f"{n}.png"))
              for n in ["fig01_hota_vs_steps","fig02_deta_assa_vs_steps","fig03_fp_fn_vs_steps",
                        "fig04_detection_outcomes","fig05_candidate_comparison",
                        "fig06_identity_metrics","fig07_epochwise_summary"] if os.path.exists(os.path.join(FIG_DIR,f"{n}.png"))]
    DW,DH = 4800,3200
    dash = Image.new("RGB",(DW,DH),(255,255,255))
    positions = [(0,0,DW//3,DH//3),(DW//3,0,DW//3,DH//3),(2*DW//3,0,DW//3,DH//3),
                 (0,DH//3,DW//3,DH//3),(DW//3,DH//3,DW//3,DH//3),(2*DW//3,DH//3,DW//3,DH//3),
                 (0,2*DH//3,2*DW//3,DH//3)]
    draw = ImageDraw.Draw(dash)
    try: font = ImageFont.truetype("arial.ttf",48)
    except: font = ImageFont.load_default()
    for idx,(img,(x,y,cw,ch)) in enumerate(zip(panels,positions)):
        import PIL; ir = img.resize((cw-20,ch-30),PIL.Image.LANCZOS); dash.paste(ir,(x+10,y+30))
        draw.text((x+15,y+5),f"({'ABCDEFG'[idx]})",fill=(30,30,30),font=font)
    # Status box
    sx,sy = 2*DW//3,2*DH//3
    draw.rectangle([sx+10,sy+10,sx+DW//3-10,sy+DH//3-10],fill=(245,250,255),outline=(180,180,180),width=2)
    try: fs = ImageFont.truetype("arial.ttf",32)
    except: fs = ImageFont.load_default()
    for li,line in enumerate(["(H) Status", f"Phase5 HOTA: {ph5_hota:.2f}",
        f"Arm1 best: {best1['checkpoint']} HOTA {best1['HOTA']:.2f}",
        f"Arm2 best: {best2['checkpoint']} HOTA {best2['HOTA']:.2f}",
        "Thresh 0.6: NOT RUN", "Thresh 0.7/0.8: ckpt4 only",
        "Pending: Seed repeats (GATE 2)", "Pending: LR sweep", "Pending: C1 calibration"]):
        color = (0,80,160) if li==0 else ((180,0,0) if "NOT RUN" in line or "Pending" in line else (40,40,40))
        draw.text((sx+25,sy+15+li*40),line,fill=color,font=fs)
    dash.save(os.path.join(FIG_DIR,"pre_tuning_dashboard.png"))
    print("  [fig] pre_tuning_dashboard.png")
except Exception as e:
    print(f"  [fig] Dashboard SKIPPED: {e}")

# ─── STEP C: TABLES ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP C: Generating tables")
print("="*60)

# table01: all checkpoints
h01 = ["Run","Ckpt","Steps","Epochs","HOTA","DetA","AssA","MOTA","IDF1","IDSW","FP","FN","Recall%","Prec%"]
hmax = max(r["HOTA"] for r in pc_data); fpmin = min(r["FP"] for r in pc_data)
fnmin = min(r["FN"] for r in pc_data); idmin = min(r["IDSW"] for r in pc_data)
rows01 = []
for r in pc_data:
    h = f"\\textbf{{{r['HOTA']:.2f}}}" if r["HOTA"]==hmax else f"{r['HOTA']:.2f}"
    fp = f"\\textbf{{{int(r['FP'])}}}" if r["FP"]==fpmin else str(int(r["FP"]))
    fn = f"\\textbf{{{int(r['FN'])}}}" if r["FN"]==fnmin else str(int(r["FN"]))
    iw = f"\\textbf{{{int(r['IDSW'])}}}" if r["IDSW"]==idmin else str(int(r["IDSW"]))
    rows01.append([r["run"],r["checkpoint"],r["steps"],r["epochs"],h,f"{r['DetA']:.2f}",f"{r['AssA']:.2f}",f"{r['MOTA']:.2f}",f"{r['IDF1']:.2f}",iw,fp,fn,f"{r['Recall']:.2f}",f"{r['Prec']:.2f}"])
write_table("table01_all_checkpoints",h01,rows01,
    "Results for all 11 checkpoints on seq\\_02\\_30sec\\_a (S0, thresh 0.5, miss\\_tol 30). Bold = best per column.",
    "tab:all_checkpoints")

# table02: 3 candidates
ph5r = all_data.get(("Phase5","ckpt_4",0.5),{})
h02 = ["Run","Ckpt","HOTA","dHOTA","DetA","AssA","MOTA","IDF1","IDSW","FP","FN","Recall%","Prec%"]
rows02 = []
for run,ckpt,clabel,_ in CAND4:
    r = all_data.get((run,ckpt,0.5),{})
    if not r: continue
    dh = r["HOTA"] - ph5r.get("HOTA",0)
    rows02.append([run,ckpt,f"{r['HOTA']:.2f}",f"{dh:+.2f}",f"{r['DetA']:.2f}",f"{r['AssA']:.2f}",f"{r['MOTA']:.2f}",f"{r['IDF1']:.2f}",int(r["IDSW"]),int(r["FP"]),int(r["FN"]),f"{r['Recall']:.2f}",f"{r['Prec']:.2f}"])
write_table("table02_selected_candidates",h02,rows02,
    "Comparison of selected candidates. dHOTA relative to Phase 5 baseline (HOTA 50.51).",
    "tab:selected_candidates")

# table03: detection outcomes
h03 = ["Run","Ckpt","TP_own","FP_own","FN_own","Rec_own%","Prec_own%","TP_TrackEval","FP_TrackEval","FN_TrackEval","dTP","dFP","dFN"]
rows03 = [[r["run"],r["checkpoint"],r["TP_own"],r["FP_own"],r["FN_own"],f"{r['recall_own']:.2f}",f"{r['precision_own']:.2f}",r["TP_te"],r["FP_te"],r["FN_te"],r["dTP"],r["dFP"],r["dFN"]] for r in det_rows_out]
write_table("table03_detection_outcomes",h03,rows03,
    "Per-frame greedy IoU$\\geq$0.5 matching vs TrackEval CLEAR. Differences quantify the two methods.",
    "tab:detection_outcomes")

# table04: threshold
h04 = ["Run","Ckpt","Threshold","HOTA","DetA","MOTA","FP","FN","Recall%","Note"]
rows04 = []
for (run,ckpt,thresh),r in sorted(all_data.items(),key=lambda x:(x[0][0],x[0][2])):
    if ckpt not in ("ckpt_4","checkpoint_4"): continue
    note = "primary" if thresh==0.5 else "exploratory"
    rows04.append([run,ckpt,f"{thresh:.1f}",f"{r['HOTA']:.2f}",f"{r['DetA']:.2f}",f"{r['MOTA']:.2f}",int(r["FP"]),int(r["FN"]),f"{r['Recall']:.2f}",note])
for run in ["Phase5","Arm1","Arm2"]:
    rows04.append([run,"ckpt_4","0.6","—","—","—","—","—","—","NOT RUN"])
rows04.sort(key=lambda r:(r[0],float(r[2]) if r[2]!="—" else 0.6))
write_table("table04_threshold_exploratory",h04,rows04,
    "Threshold sensitivity (ckpt4 only, exploratory). Threshold 0.6 not run. Best checkpoints not swept.",
    "tab:threshold")

# table05: run inventory
h05 = ["Run","Config","Start weights","Seed","Data","Steps/ep","Epochs","LR schedule","Backbone","s/step","Wall time"]
rows05 = [
    ["Phase 5","(UA-DETRAC+VisDrone)","DanceTrack",42,"UA-DETRAC+VisDrone",0,0,"1e-4 MultiStep","unfrozen","~0.66","~14h (est.)"],
    ["Arm 1","train_iotd_real.yaml","Phase5 ckpt4 (wt only)",42,"IOTD_train_real",2528,5,"2e-5 ep0-3; 2e-6 ep4","frozen","0.664","~140 min"],
    ["Arm 2","train_iotd_real_aug.yaml","Phase5 ckpt4 (wt only)",42,"IOTD_train_real+aug",10112,5,"2e-5 ep0-3; 2e-6 ep4","frozen","0.664","~565 min"],
]
write_table("table05_run_inventory",h05,rows05,
    "Experiment run inventory. Clip length=2 throughout all 5 epochs (curriculum milestones not reached).",
    "tab:run_inventory")

# table06: dataset summary
import configparser
seqs_info = [("seq_01_1min","data/IOTD_train/seq_01_1min","train","day/night","6"),
             ("seq_02_30sec_a","data/IOTD_val/seq_02_30sec_a","val","night","N/A"),
             ("seq_03_30sec_b","data/IOTD_train/seq_03_30sec_b","train","day","6")]
h06 = ["Sequence","Role","Frames","GT rows","Tracks","Scene","Aug copies"]
rows06 = []
for seq,sdir,role,scene,aug in seqs_info:
    si = os.path.join(sdir,"seqinfo.ini"); gt = os.path.join(sdir,"gt/gt.txt")
    cp = configparser.ConfigParser(); cp.read(si)
    frames = int(cp["Sequence"]["seqlength"]) if os.path.exists(si) else "?"
    if os.path.exists(gt):
        lines = [l for l in open(gt).readlines() if l.strip()]
        tids = set(int(l.split(",")[1]) for l in lines if len(l.split(","))>1)
        rows06.append([seq,role,frames,len(lines),len(tids),scene,aug])
    else:
        rows06.append([seq,role,frames,"?","?",scene,aug])
write_table("table06_dataset_summary",h06,rows06,
    "Dataset split. Real annotated: 3,274 frames (1,599 train + 744 val + 931 train). Real training: 2,530 frames. Augmented frames are training samples only.",
    "tab:dataset")

# ─── Final verification ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("VERIFICATION — all outputs")
print("="*60)
all_files = [
    (os.path.join(DATA_DIR,"per_checkpoint_metrics.csv"),"Data: 11-row metrics CSV"),
    (os.path.join(DATA_DIR,"results_registry.csv"),"Data: frozen registry (17 rows)"),
    (os.path.join(DATA_DIR,"detection_outcomes.csv"),"Data: own match vs TrackEval"),
]
for n in ["fig01_hota_vs_steps","fig02_deta_assa_vs_steps","fig03_fp_fn_vs_steps",
          "fig04_detection_outcomes","fig05_candidate_comparison","fig06_identity_metrics",
          "fig07_epochwise_summary","fig08_qualitative","pre_tuning_dashboard"]:
    all_files.append((os.path.join(FIG_DIR,f"{n}.png"),f"Figure: {n}"))
for n in ["table01_all_checkpoints","table02_selected_candidates","table03_detection_outcomes",
          "table04_threshold_exploratory","table05_run_inventory","table06_dataset_summary"]:
    all_files.append((os.path.join(TABLE_DIR,f"{n}.csv"),f"Table: {n}"))

ok=0; miss=0
for fpath,desc in all_files:
    exists = os.path.exists(fpath)
    if exists: ok+=1
    else: miss+=1
    print(f"  {'OK    ' if exists else 'MISSING'} {desc}")
print(f"\nTotal: {ok} OK, {miss} missing")
print(f"\nAll outputs in: {os.path.abspath(OUT_DIR)}")
