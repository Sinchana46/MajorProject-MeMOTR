"""
Pre-tuning package Step C: generate tables (CSV, LaTeX booktabs, Markdown).
"""
import os, sys, csv, json, datetime

OUT_BASE  = "results/pre_tuning_baseline_v1"
DATA_DIR  = os.path.join(OUT_BASE, "data")
TABLE_DIR = os.path.join(OUT_BASE, "tables")
GT_PATH   = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
os.makedirs(TABLE_DIR, exist_ok=True)

def bold_max(vals, fmt="{:.2f}", skip_idx=None, lower_is_better=False):
    """Return formatted strings with LaTeX bold on best value."""
    nums = [float(v) for v in vals]
    best = min(nums) if lower_is_better else max(nums)
    out = []
    for i, v in enumerate(nums):
        s = fmt.format(v)
        if abs(v - best) < 0.005 and (skip_idx is None or i != skip_idx):
            out.append(f"\\textbf{{{s}}}")
        else:
            out.append(s)
    return out

def write_table(name, headers, rows, caption, label):
    """Write CSV, LaTeX booktabs, and Markdown."""
    csv_path = os.path.join(TABLE_DIR, f"{name}.csv")
    tex_path = os.path.join(TABLE_DIR, f"{name}.tex")
    md_path  = os.path.join(TABLE_DIR, f"{name}.md")

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

    # Markdown
    col_sep = " | "
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + col_sep.join(str(h) for h in headers) + " |\n")
        f.write("| " + " | ".join(["---"]*len(headers)) + " |\n")
        for row in rows:
            # Strip LaTeX bold for markdown
            md_row = [str(c).replace("\\textbf{","").replace("}","") for c in row]
            f.write("| " + col_sep.join(md_row) + " |\n")

    # LaTeX booktabs
    col_fmt = "l" + "r"*(len(headers)-1)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(str(h) for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(c) for c in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  Wrote: {name}.csv, .tex, .md")
    return csv_path, tex_path, md_path

# Load per_checkpoint_metrics
pc_data = []
pc_path = os.path.join(DATA_DIR, "per_checkpoint_metrics.csv")
with open(pc_path, newline="") as f:
    for row in csv.DictReader(f):
        for k in ["steps","epochs_completed","IDSW","TP","FP","FN"]:
            row[k] = int(float(row[k]))
        for k in ["HOTA","DetA","AssA","MOTA","IDF1","recall","precision"]:
            row[k] = float(row[k])
        pc_data.append(row)

det_data = []
det_path = os.path.join(DATA_DIR, "detection_outcomes.csv")
with open(det_path, newline="") as f:
    for row in csv.DictReader(f):
        for k in ["TP_own","FP_own","FN_own","TP_trackeval","FP_trackeval","FN_trackeval","delta_TP","delta_FP","delta_FN"]:
            row[k] = int(row[k])
        for k in ["recall_own","precision_own"]:
            row[k] = float(row[k])
        det_data.append(row)

# Threshold data from registry
reg_path = os.path.join(DATA_DIR, "results_registry.csv")
reg_data = []
with open(reg_path, newline="") as f:
    for row in csv.DictReader(f):
        for k in ["steps","epochs_completed","IDSW","TP","FP","FN"]:
            row[k] = int(float(row[k]))
        for k in ["HOTA","DetA","AssA","MOTA","IDF1","recall","precision","threshold"]:
            row[k] = float(row[k])
        reg_data.append(row)

print("=" * 60)
print("STEP C: BUILDING TABLES")
print("=" * 60)

# ─── table01: all checkpoints ─────────────────────────────────────────────────
h01 = ["Run","Ckpt","Steps","Epochs","HOTA","DetA","AssA","MOTA","IDF1","IDSW","FP","FN","Recall%","Prec%"]
arm_order = {"Phase5":0,"Arm1":1,"Arm2":2}
sorted_pc = sorted(pc_data, key=lambda r: (arm_order.get(r["run"],9), r["steps"]))

# Bold per-column bests
hota_vals = [r["HOTA"] for r in sorted_pc]
fp_vals   = [r["FP"]   for r in sorted_pc]
fn_vals   = [r["FN"]   for r in sorted_pc]
idsw_vals = [r["IDSW"] for r in sorted_pc]
rows01 = []
for r in sorted_pc:
    hota_s = f"{r['HOTA']:.2f}"
    if r["HOTA"] == max(hota_vals): hota_s = f"\\textbf{{{hota_s}}}"
    fp_s = str(r["FP"])
    if r["FP"] == min(fp_vals): fp_s = f"\\textbf{{{fp_s}}}"
    fn_s = str(r["FN"])
    if r["FN"] == min(fn_vals): fn_s = f"\\textbf{{{fn_s}}}"
    idsw_s = str(r["IDSW"])
    if r["IDSW"] == min(idsw_vals): idsw_s = f"\\textbf{{{idsw_s}}}"
    rows01.append([
        r["run"], r["checkpoint"], r["steps"], r["epochs_completed"],
        hota_s,
        f"{r['DetA']:.2f}", f"{r['AssA']:.2f}", f"{r['MOTA']:.2f}",
        f"{r['IDF1']:.2f}", idsw_s, fp_s, fn_s,
        f"{r['recall']:.2f}", f"{r['precision']:.2f}"
    ])
write_table("table01_all_checkpoints", h01, rows01,
    "Results for all 11 checkpoints on seq\\_02\\_30sec\\_a (S0 original GT, det/track 0.5, miss tolerance 30).",
    "tab:all_checkpoints")

# ─── table02: selected candidates ────────────────────────────────────────────
CANDS = [("Phase5","ckpt_4"),("Arm1","ckpt_3"),("Arm2","ckpt_1")]
cand_data = {}
for run, ckpt in CANDS:
    row = next((r for r in pc_data if r["run"]==run and r["checkpoint"]==ckpt), None)
    cand_data[(run,ckpt)] = row

ph5 = cand_data[("Phase5","ckpt_4")]
h02 = ["Run","Ckpt","HOTA","dHOTA","DetA","AssA","MOTA","IDF1","IDSW","FP","FN","Recall%","Prec%"]
rows02 = []
for run, ckpt in CANDS:
    r = cand_data[(run,ckpt)]
    if r is None: continue
    delta = r["HOTA"] - ph5["HOTA"] if ph5 else 0
    delta_s = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
    rows02.append([
        run, r["checkpoint"],
        f"{r['HOTA']:.2f}", delta_s,
        f"{r['DetA']:.2f}", f"{r['AssA']:.2f}", f"{r['MOTA']:.2f}",
        f"{r['IDF1']:.2f}", r["IDSW"], r["FP"], r["FN"],
        f"{r['recall']:.2f}", f"{r['precision']:.2f}"
    ])
write_table("table02_selected_candidates", h02, rows02,
    "Comparison of selected candidates vs Phase 5 baseline. Phase 5 HOTA = 50.51 (reference). ΔHOTA relative to Phase 5.",
    "tab:selected_candidates")

# ─── table03: detection outcomes ─────────────────────────────────────────────
h03 = ["Run","Ckpt","TP_own","FP_own","FN_own","Rec_own%","Prec_own%",
       "TP_TrackEval","FP_TrackEval","FN_TrackEval","dTP","dFP","dFN"]
rows03 = []
for r in det_data:
    rows03.append([
        r["run"], r["checkpoint"],
        r["TP_own"], r["FP_own"], r["FN_own"],
        f"{r['recall_own']:.2f}", f"{r['precision_own']:.2f}",
        r["TP_trackeval"], r["FP_trackeval"], r["FN_trackeval"],
        r["delta_TP"], r["delta_FP"], r["delta_FN"]
    ])
write_table("table03_detection_outcomes", h03, rows03,
    "Per-frame greedy IoU$\\geq$0.5 match vs TrackEval CLEAR counts. Differences quantify the two matching methods.",
    "tab:detection_outcomes")

# ─── table04: threshold exploratory ──────────────────────────────────────────
h04 = ["Run","Ckpt","Threshold","HOTA","DetA","MOTA","FP","FN","Recall%","Prec%","Note"]
thresh_rows = [r for r in reg_data if r["threshold"] in (0.5, 0.7, 0.8) and
               r["checkpoint"] in ("ckpt_4","checkpoint_4")]
rows04 = []
for r in sorted(thresh_rows, key=lambda x: (x["run"], x["threshold"])):
    note = "exploratory" if r["threshold"] != 0.5 else "primary"
    rows04.append([
        r["run"], r["checkpoint"], f"{r['threshold']:.1f}",
        f"{r['HOTA']:.2f}", f"{r['DetA']:.2f}", f"{r['MOTA']:.2f}",
        r["FP"], r["FN"], f"{r['recall']:.2f}", f"{r['precision']:.2f}", note
    ])
# Add 0.6 as missing
for run in ["Phase5","Arm1","Arm2"]:
    rows04.append([run, "ckpt_4", "0.6", "—","—","—","—","—","—","—","NOT RUN"])
rows04.sort(key=lambda r: (r[0], float(r[2]) if r[2] != "0.6" else 0.6))
write_table("table04_threshold_exploratory", h04, rows04,
    "Threshold sensitivity (exploratory). Best checkpoints per arm not yet swept. 0.6 not run for any run.",
    "tab:threshold_exploratory")

# ─── table05: run inventory ───────────────────────────────────────────────────
h05 = ["Run","Config","Start weights","Seed","Data","Steps/epoch","Epochs","LR","Backbone","s/step","Wall time"]
rows05 = [
    ["Phase 5","(UA-DETRAC+VisDrone fine-tune)","DanceTrack pretrain",42,"UA-DETRAC+VisDrone",0,0,"1e-4 (MultiStep)","unfrozen","~0.66","~14h (est.)"],
    ["Arm 1","configs/train_iotd_real.yaml","phase5 ckpt4 (wt only)",42,"IOTD_train_real",2528,5,"2e-5 (0-3), 2e-6 (4)","frozen","0.664","~140 min"],
    ["Arm 2","configs/train_iotd_real_aug.yaml","phase5 ckpt4 (wt only)",42,"IOTD_train_real+aug",10112,5,"2e-5 (0-3), 2e-6 (4)","frozen","0.664","~565 min"],
]
write_table("table05_run_inventory", h05, rows05,
    "Experiment run inventory. Both IOTD arms start from Phase 5 checkpoint 4 (weights only). Backbone frozen throughout. Clip length 2 in all 5 epochs.",
    "tab:run_inventory")

# ─── table06: dataset summary ─────────────────────────────────────────────────
# Get frame counts and GT stats from files
import configparser
seqs = [
    ("seq_01_1min",    "data/IOTD_train/seq_01_1min",    "train",  "day-night"),
    ("seq_02_30sec_a", "data/IOTD_val/seq_02_30sec_a",   "val",    "night"),
    ("seq_03_30sec_b", "data/IOTD_train/seq_03_30sec_b", "train",  "day"),
]

h06 = ["Sequence","Role","Frames","GT rows","Tracks","Mean luma","Class dist.","Augmented copies"]
rows06 = []
for seq_name, seq_dir, role, scene in seqs:
    si = os.path.join(seq_dir, "seqinfo.ini")
    gt = os.path.join(seq_dir, "gt/gt.txt")
    if os.path.exists(si):
        cp = configparser.ConfigParser(); cp.read(si)
        frames = int(cp["Sequence"]["seqlength"])
    else:
        frames = "?"
    if os.path.exists(gt):
        gt_rows_list = open(gt).readlines()
        gt_n = len([l for l in gt_rows_list if l.strip()])
        tids = set(int(l.split(",")[1]) for l in gt_rows_list if l.strip() and len(l.split(","))>1)
        n_tracks = len(tids)
        # Class distribution
        from collections import Counter
        cls_map = {1:"car",2:"2w",3:"auto",4:"bus",5:"truck"}
        cls_c = Counter(int(l.split(",")[7]) for l in gt_rows_list if l.strip() and len(l.split(","))>8)
        cls_str = " ".join(f"{cls_map.get(k,k)}:{v}" for k,v in sorted(cls_c.items()))
    else:
        gt_n = "?"; n_tracks = "?"; cls_str = "?"
    aug_copies = "6" if role=="train" else "N/A"
    rows06.append([seq_name, role, frames, gt_n, n_tracks, scene, cls_str, aug_copies])
write_table("table06_dataset_summary", h06, rows06,
    "Dataset split. Real annotated frames: 3,274 total (1,599 train + 744 val + 931 train). Real training frames: 2,530. Augmented frames are training samples, not additional annotated data.",
    "tab:dataset_summary")

print("\nStep C DONE — all tables written to", TABLE_DIR)
