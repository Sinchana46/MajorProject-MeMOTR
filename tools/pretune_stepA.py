"""
Pre-tuning package Step A: load all TrackEval summaries, verify vs claims,
build per_checkpoint_metrics.csv, results_registry.csv, detection_outcomes.csv.
"""
import os, sys, csv, json, hashlib, datetime
from collections import defaultdict
import numpy as np

OUT_BASE  = "results/pre_tuning_baseline_v1"
DATA_DIR  = os.path.join(OUT_BASE, "data")
GT_ORIG   = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
TRACKER_BASE = "results/eval_iotd_val"

os.makedirs(DATA_DIR, exist_ok=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def read_trackeval_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        return None
    headers = lines[0].split()
    vals    = lines[1].split()
    d = dict(zip(headers, vals))
    return d

# Claimed values from antigravity_pretuning_report.md §2
CLAIMS = [
    {"run":"Phase5","ckpt":"checkpoint_4","steps":0,"epochs":0,
     "HOTA":50.51,"DetA":43.88,"AssA":58.49,"MOTA":29.10,"IDF1":68.84,"IDSW":1,"FP":2117,"FN":847},
    {"run":"Arm1","ckpt":"ckpt_0","steps":2528,"epochs":1,
     "HOTA":45.72,"DetA":42.49,"AssA":49.58,"MOTA":17.03,"IDF1":63.08,"IDSW":21,"FP":2847,"FN":602},
    {"run":"Arm1","ckpt":"ckpt_1","steps":5056,"epochs":2,
     "HOTA":49.30,"DetA":46.42,"AssA":52.76,"MOTA":30.03,"IDF1":68.38,"IDSW":14,"FP":2499,"FN":413},
    {"run":"Arm1","ckpt":"ckpt_2","steps":7584,"epochs":3,
     "HOTA":48.87,"DetA":45.93,"AssA":52.27,"MOTA":27.83,"IDF1":68.43,"IDSW":12,"FP":2643,"FN":363},
    {"run":"Arm1","ckpt":"ckpt_3","steps":10112,"epochs":4,
     "HOTA":50.47,"DetA":46.96,"AssA":54.51,"MOTA":29.63,"IDF1":70.23,"IDSW":9,"FP":2629,"FN":305},
    {"run":"Arm1","ckpt":"ckpt_4","steps":12640,"epochs":5,
     "HOTA":49.58,"DetA":45.70,"AssA":54.04,"MOTA":24.94,"IDF1":69.66,"IDSW":7,"FP":2855,"FN":277},
    {"run":"Arm2","ckpt":"ckpt_0","steps":10112,"epochs":1,
     "HOTA":40.51,"DetA":36.49,"AssA":45.15,"MOTA":16.28,"IDF1":56.63,"IDSW":10,"FP":1964,"FN":1527},
    {"run":"Arm2","ckpt":"ckpt_1","steps":20224,"epochs":2,
     "HOTA":48.17,"DetA":45.54,"AssA":51.14,"MOTA":29.84,"IDF1":68.64,"IDSW":12,"FP":2432,"FN":490},
    {"run":"Arm2","ckpt":"ckpt_2","steps":30336,"epochs":3,
     "HOTA":44.77,"DetA":41.22,"AssA":48.77,"MOTA":8.42,"IDF1":67.13,"IDSW":3,"FP":3671,"FN":156},
    {"run":"Arm2","ckpt":"ckpt_3","steps":40448,"epochs":4,
     "HOTA":43.41,"DetA":39.18,"AssA":48.21,"MOTA":-3.97,"IDF1":64.56,"IDSW":3,"FP":4225,"FN":120},
    {"run":"Arm2","ckpt":"ckpt_4","steps":50560,"epochs":5,
     "HOTA":42.84,"DetA":38.32,"AssA":48.01,"MOTA":-6.58,"IDF1":63.72,"IDSW":6,"FP":4266,"FN":185},
]

SUMMARY_PATHS = {
    ("Phase5","ckpt_4"): f"{TRACKER_BASE}/trackeval_phase5/trackeval_output/PHASE5/pedestrian_summary.txt",
    ("Arm1","ckpt_0"):   f"{TRACKER_BASE}/trackeval_arm1_ckpt0/trackeval_output/ARM1_CKPT0/pedestrian_summary.txt",
    ("Arm1","ckpt_1"):   f"{TRACKER_BASE}/trackeval_arm1_ckpt1/trackeval_output/ARM1_CKPT1/pedestrian_summary.txt",
    ("Arm1","ckpt_2"):   f"{TRACKER_BASE}/trackeval_arm1_ckpt2/trackeval_output/ARM1_CKPT2/pedestrian_summary.txt",
    ("Arm1","ckpt_3"):   f"{TRACKER_BASE}/trackeval_arm1_ckpt3/trackeval_output/ARM1_CKPT3/pedestrian_summary.txt",
    ("Arm1","ckpt_4"):   f"{TRACKER_BASE}/trackeval_arm1/trackeval_output/ARM1/pedestrian_summary.txt",
    ("Arm2","ckpt_0"):   f"{TRACKER_BASE}/trackeval_arm2_ckpt0/trackeval_output/ARM2_CKPT0/pedestrian_summary.txt",
    ("Arm2","ckpt_1"):   f"{TRACKER_BASE}/trackeval_arm2_ckpt1/trackeval_output/ARM2_CKPT1/pedestrian_summary.txt",
    ("Arm2","ckpt_2"):   f"{TRACKER_BASE}/trackeval_arm2_ckpt2/trackeval_output/ARM2_CKPT2/pedestrian_summary.txt",
    ("Arm2","ckpt_3"):   f"{TRACKER_BASE}/trackeval_arm2_ckpt3/trackeval_output/ARM2_CKPT3/pedestrian_summary.txt",
    ("Arm2","ckpt_4"):   f"{TRACKER_BASE}/trackeval_arm2/trackeval_output/ARM2/pedestrian_summary.txt",
}

TRACKER_PATHS = {
    ("Phase5","ckpt_4"): f"{TRACKER_BASE}/phase5/seq_02_30sec_a.txt",
    ("Arm1","ckpt_0"):   f"{TRACKER_BASE}/arm1_ckpt0/seq_02_30sec_a.txt",
    ("Arm1","ckpt_1"):   f"{TRACKER_BASE}/arm1_ckpt1/seq_02_30sec_a.txt",
    ("Arm1","ckpt_2"):   f"{TRACKER_BASE}/arm1_ckpt2/seq_02_30sec_a.txt",
    ("Arm1","ckpt_3"):   f"{TRACKER_BASE}/arm1_ckpt3/seq_02_30sec_a.txt",
    ("Arm1","ckpt_4"):   f"{TRACKER_BASE}/arm1/seq_02_30sec_a.txt",
    ("Arm2","ckpt_0"):   f"{TRACKER_BASE}/arm2_ckpt0/seq_02_30sec_a.txt",
    ("Arm2","ckpt_1"):   f"{TRACKER_BASE}/arm2_ckpt1/seq_02_30sec_a.txt",
    ("Arm2","ckpt_2"):   f"{TRACKER_BASE}/arm2_ckpt2/seq_02_30sec_a.txt",
    ("Arm2","ckpt_3"):   f"{TRACKER_BASE}/arm2_ckpt3/seq_02_30sec_a.txt",
    ("Arm2","ckpt_4"):   f"{TRACKER_BASE}/arm2/seq_02_30sec_a.txt",
}

CONFIG_MAP = {
    "Arm1": "configs/train_iotd_real.yaml",
    "Arm2": "configs/train_iotd_real_aug.yaml",
    "Phase5": "results/phase5_interim_baseline/checkpoint_4.pth",
}

GT_SHA = sha256_file(GT_ORIG)
print(f"GT SHA256: {GT_SHA}")
print()

# ─── Step A1: Locate and verify ──────────────────────────────────────────────
print("="*80)
print("STEP A1: Summary file locations and claim verification")
print("="*80)

METRIC_MAP = {
    "HOTA":"HOTA","DetA":"DetA","AssA":"AssA",
    "MOTA":"MOTA","IDF1":"IDF1","IDSW":"IDSW",
    "FP":"CLR_FP","FN":"CLR_FN"
}

all_metrics = {}
mismatches = []
print(f"{'Run':<8}{'Ckpt':<8}{'HOTA':>6}{'DetA':>6}{'AssA':>6}{'MOTA':>7}{'IDF1':>6}{'IDSW':>5}{'FP':>6}{'FN':>6}  Status")
print("-"*80)

for claim in CLAIMS:
    run  = claim["run"]
    ckpt_raw = claim["ckpt"]
    ckpt = ckpt_raw.replace("checkpoint_","ckpt_")
    key  = (run, ckpt)
    sumpath = SUMMARY_PATHS.get(key)
    trkpath = TRACKER_PATHS.get(key)
    d = read_trackeval_summary(sumpath) if sumpath else None

    if d is None:
        print(f"{run:<8}{ckpt:<8}  SUMMARY NOT FOUND: {sumpath}")
        all_metrics[key] = None
        continue

    row = {}
    status_parts = []
    for metric, col in METRIC_MAP.items():
        val = float(d.get(col, 0))
        row[metric] = val
        claimed = float(claim[metric])
        diff = abs(val - claimed)
        if diff > 0.01:
            status_parts.append(f"{metric}:claimed={claimed},got={val:.2f}")
    row["TP"]       = float(d.get("CLR_TP", 0))
    row["Recall"]   = float(d.get("CLR_Re", 0))
    row["Prec"]     = float(d.get("CLR_Pr", 0))
    row["summary_path"]  = sumpath
    row["summary_sha256"]= sha256_file(sumpath)
    row["tracker_path"]  = trkpath or ""
    row["tracker_rows"]  = sum(1 for _ in open(trkpath)) if trkpath and os.path.exists(trkpath) else -1
    all_metrics[key] = row

    status = "OK" if not status_parts else "MISMATCH: " + "; ".join(status_parts)
    if status_parts:
        mismatches.append({"run":run,"ckpt":ckpt,"details":"; ".join(status_parts)})
    print(f"{run:<8}{ckpt:<8}{row['HOTA']:>6.2f}{row['DetA']:>6.2f}{row['AssA']:>6.2f}"
          f"{row['MOTA']:>7.2f}{row['IDF1']:>6.2f}{row['IDSW']:>5.0f}"
          f"{row['FP']:>6.0f}{row['FN']:>6.0f}  {status}")

print(f"\nTotal mismatches (>0.01): {len(mismatches)}")
for m in mismatches:
    print(f"  {m['run']} {m['ckpt']}: {m['details']}")

# ─── Step A2: per_checkpoint_metrics.csv ────────────────────────────────────
print("\n" + "="*80)
print("STEP A2: Building per_checkpoint_metrics.csv")
print("="*80)

now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
pc_cols = ["run","checkpoint","steps","epochs_completed","seed","config","scoring",
           "threshold","GT_sha256","HOTA","DetA","AssA","MOTA","IDF1","IDSW",
           "TP","FP","FN","recall","precision","summary_path","summary_sha256"]
pc_rows = []

SEED = 42
THRESH = 0.5
STEPS_PER_EPOCH = {"Phase5":0,"Arm1":2528,"Arm2":10112}
CONFIG_PATHS = {
    "Phase5":"results/phase5_interim_baseline/checkpoint_4.pth (weights-only load)",
    "Arm1":"configs/train_iotd_real.yaml",
    "Arm2":"configs/train_iotd_real_aug.yaml",
}

for claim in CLAIMS:
    run  = claim["run"]
    ckpt_raw = claim["ckpt"]
    ckpt = ckpt_raw.replace("checkpoint_","ckpt_")
    key  = (run, ckpt)
    r    = all_metrics.get(key)
    if r is None:
        continue
    pc_rows.append({
        "run": run, "checkpoint": ckpt,
        "steps": claim["steps"], "epochs_completed": claim["epochs"],
        "seed": SEED, "config": CONFIG_PATHS.get(run,""),
        "scoring": "S0", "threshold": THRESH,
        "GT_sha256": GT_SHA,
        "HOTA": round(r["HOTA"],3), "DetA": round(r["DetA"],3),
        "AssA": round(r["AssA"],3), "MOTA": round(r["MOTA"],3),
        "IDF1": round(r["IDF1"],3), "IDSW": int(r["IDSW"]),
        "TP": int(r["TP"]), "FP": int(r["FP"]), "FN": int(r["FN"]),
        "recall": round(r["Recall"],3), "precision": round(r["Prec"],3),
        "summary_path": r["summary_path"],
        "summary_sha256": r["summary_sha256"],
    })

pc_path = os.path.join(DATA_DIR, "per_checkpoint_metrics.csv")
with open(pc_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pc_cols)
    w.writeheader(); w.writerows(pc_rows)
print(f"Wrote {pc_path} ({len(pc_rows)} rows)")

# ─── Step A2: results_registry.csv ───────────────────────────────────────────
print("\nBuilding results_registry.csv...")
reg_cols = pc_cols + ["registry_version","created_utc"]
reg_rows = [{**r, "registry_version":"v1","created_utc":now_utc} for r in pc_rows]

# Also add threshold 0.7 / 0.8 rows for ckpt_4 runs
THRESH_SUMMARIES = {
    ("Phase5","ckpt_4","0.7"): "results/eval_iotd_val/diag/thresh_0p7/trackeval_phase5/trackeval_output/PHASE5_T0P7/pedestrian_summary.txt",
    ("Arm1","ckpt_4","0.7"):   "results/eval_iotd_val/diag/thresh_0p7/trackeval_arm1/trackeval_output/ARM1_T0P7/pedestrian_summary.txt",
    ("Arm2","ckpt_4","0.7"):   "results/eval_iotd_val/diag/thresh_0p7/trackeval_arm2/trackeval_output/ARM2_T0P7/pedestrian_summary.txt",
    ("Phase5","ckpt_4","0.8"): "results/eval_iotd_val/diag/thresh_0p8/trackeval_phase5/trackeval_output/PHASE5_T0P8/pedestrian_summary.txt",
    ("Arm1","ckpt_4","0.8"):   "results/eval_iotd_val/diag/thresh_0p8/trackeval_arm1/trackeval_output/ARM1_T0P8/pedestrian_summary.txt",
    ("Arm2","ckpt_4","0.8"):   "results/eval_iotd_val/diag/thresh_0p8/trackeval_arm2/trackeval_output/ARM2_T0P8/pedestrian_summary.txt",
}

for (run, ckpt, thr), sumpath in THRESH_SUMMARIES.items():
    d = read_trackeval_summary(sumpath)
    if d is None:
        print(f"  MISSING threshold summary: {run} {ckpt} thresh={thr}")
        continue
    reg_rows.append({
        "run":run,"checkpoint":ckpt,
        "steps":{"Phase5":12640,"Arm1":12640,"Arm2":50560}.get(run,0),
        "epochs_completed":5,"seed":SEED,"config":CONFIG_PATHS.get(run,""),
        "scoring":"S0","threshold":float(thr),"GT_sha256":GT_SHA,
        "HOTA":round(float(d.get("HOTA",0)),3),"DetA":round(float(d.get("DetA",0)),3),
        "AssA":round(float(d.get("AssA",0)),3),"MOTA":round(float(d.get("MOTA",0)),3),
        "IDF1":round(float(d.get("IDF1",0)),3),"IDSW":int(float(d.get("IDSW",0))),
        "TP":int(float(d.get("CLR_TP",0))),"FP":int(float(d.get("CLR_FP",0))),
        "FN":int(float(d.get("CLR_FN",0))),
        "recall":round(float(d.get("CLR_Re",0)),3),"precision":round(float(d.get("CLR_Pr",0)),3),
        "summary_path":sumpath,"summary_sha256":sha256_file(sumpath),
        "registry_version":"v1","created_utc":now_utc,
    })

reg_path = os.path.join(DATA_DIR, "results_registry.csv")
with open(reg_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=reg_cols)
    w.writeheader(); w.writerows(reg_rows)
print(f"Wrote {reg_path} ({len(reg_rows)} rows)")

# ─── Step A3: detection_outcomes.csv (own Hungarian matching) ────────────────
print("\n" + "="*80)
print("STEP A3: Detection outcomes (own IoU matching, IoU>=0.5)")
print("="*80)

def load_tracker(path):
    rows = []
    if not path or not os.path.exists(path): return rows
    with open(path) as f:
        for line in f:
            c = line.strip().split(",")
            if len(c) >= 6:
                rows.append((int(c[0]), int(c[1]), float(c[2]), float(c[3]),
                             float(c[4]), float(c[5])))
    return rows

def load_gt(path):
    rows = []
    with open(path) as f:
        for line in f:
            c = line.strip().split(",")
            if len(c) >= 6:
                rows.append((int(c[0]), int(c[1]), float(c[2]), float(c[3]),
                             float(c[4]), float(c[5])))
    return rows

def box_iou(a, b):
    ax1,ay1,ax2,ay2 = a[0],a[1],a[0]+a[2],a[1]+a[3]
    bx1,by1,bx2,by2 = b[0],b[1],b[0]+b[2],b[1]+b[3]
    ix = max(0, min(ax2,bx2)-max(ax1,bx1))
    iy = max(0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.0

def match_frame(preds, gts, iou_thr=0.5):
    """Simple greedy match (descending IoU) — not Hungarian but close for sparse cases."""
    matched_p = set(); matched_g = set()
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = box_iou((p[2],p[3],p[4],p[5]), (g[2],g[3],g[4],g[5]))
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    tp = 0
    for iou, pi, gi in pairs:
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi); matched_g.add(gi); tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn

gt_rows = load_gt(GT_ORIG)
from itertools import groupby
from operator import itemgetter

gt_by_frame = defaultdict(list)
for r in gt_rows:
    gt_by_frame[r[0]].append(r)

CANDIDATES = [
    ("Phase5",  "ckpt_4", TRACKER_PATHS[("Phase5","ckpt_4")]),
    ("Arm1",    "ckpt_3", TRACKER_PATHS[("Arm1","ckpt_3")]),
    ("Arm2",    "ckpt_1", TRACKER_PATHS[("Arm2","ckpt_1")]),
]

det_rows = []
print(f"{'Run':<8}{'Ckpt':<8}{'TP_own':>8}{'FP_own':>8}{'FN_own':>8}{'Rec_own':>8}{'Pr_own':>8}"
      f"  vs  {'TP_te':>8}{'FP_te':>8}{'FN_te':>8}  dTP  dFP  dFN")
print("-"*95)
for run, ckpt, trkpath in CANDIDATES:
    trk_rows = load_tracker(trkpath)
    trk_by_frame = defaultdict(list)
    for r in trk_rows:
        trk_by_frame[r[0]].append(r)

    total_tp = total_fp = total_fn = 0
    for frame in set(list(gt_by_frame.keys()) + list(trk_by_frame.keys())):
        tp, fp, fn = match_frame(trk_by_frame.get(frame,[]), gt_by_frame.get(frame,[]))
        total_tp += tp; total_fp += fp; total_fn += fn

    rec_own = total_tp / (total_tp + total_fn) * 100 if (total_tp+total_fn) > 0 else 0
    prec_own = total_tp / (total_tp + total_fp) * 100 if (total_tp+total_fp) > 0 else 0

    # TrackEval CLEAR values
    te = all_metrics.get((run, ckpt), {}) or {}
    te_tp = int(te.get("TP", 0)); te_fp = int(te.get("FP", 0)); te_fn = int(te.get("FN", 0))
    dtp = total_tp - te_tp; dfp = total_fp - te_fp; dfn = total_fn - te_fn

    print(f"{run:<8}{ckpt:<8}{total_tp:>8}{total_fp:>8}{total_fn:>8}{rec_own:>8.2f}{prec_own:>8.2f}"
          f"  vs  {te_tp:>8}{te_fp:>8}{te_fn:>8}{dtp:>5}{dfp:>5}{dfn:>5}")

    det_rows.append({
        "run":run,"checkpoint":ckpt,
        "TP_own":total_tp,"FP_own":total_fp,"FN_own":total_fn,
        "recall_own":round(rec_own,3),"precision_own":round(prec_own,3),
        "TP_trackeval":te_tp,"FP_trackeval":te_fp,"FN_trackeval":te_fn,
        "delta_TP":dtp,"delta_FP":dfp,"delta_FN":dfn,
        "note":"Greedy per-frame IoU>=0.5 match; TrackEval uses temporal CLEAR matching across frames"
    })

det_path = os.path.join(DATA_DIR, "detection_outcomes.csv")
with open(det_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
    w.writeheader(); w.writerows(det_rows)
print(f"\nWrote {det_path}")
print("Note: Own match uses greedy per-frame IoU>=0.5; TrackEval CLEAR matching is temporal.")
print("      Differences represent the mismatch between the two methods.")

# ─── Checkpoint selection ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("CHECKPOINT SELECTION (S0, det/track=0.5, written rule: highest HOTA)")
print("="*80)

arm1_rows = [r for r in pc_rows if r["run"]=="Arm1"]
arm2_rows = [r for r in pc_rows if r["run"]=="Arm2"]
all_rows  = pc_rows

if arm1_rows:
    best_hota1 = max(arm1_rows, key=lambda r: r["HOTA"])
    low_fp1    = min(arm1_rows, key=lambda r: r["FP"])
    low_fn1    = min(arm1_rows, key=lambda r: r["FN"])
    low_idsw1  = min(arm1_rows, key=lambda r: r["IDSW"])
    print(f"Arm1 best HOTA : {best_hota1['checkpoint']} HOTA={best_hota1['HOTA']}")
    print(f"Arm1 lowest FP : {low_fp1['checkpoint']} FP={low_fp1['FP']}")
    print(f"Arm1 lowest FN : {low_fn1['checkpoint']} FN={low_fn1['FN']}")
    print(f"Arm1 lowest IDSW: {low_idsw1['checkpoint']} IDSW={low_idsw1['IDSW']}")

if arm2_rows:
    best_hota2 = max(arm2_rows, key=lambda r: r["HOTA"])
    low_fp2    = min(arm2_rows, key=lambda r: r["FP"])
    low_fn2    = min(arm2_rows, key=lambda r: r["FN"])
    low_idsw2  = min(arm2_rows, key=lambda r: r["IDSW"])
    print(f"\nArm2 best HOTA : {best_hota2['checkpoint']} HOTA={best_hota2['HOTA']}")
    print(f"Arm2 lowest FP : {low_fp2['checkpoint']} FP={low_fp2['FP']}")
    print(f"Arm2 lowest FN : {low_fn2['checkpoint']} FN={low_fn2['FN']}")
    print(f"Arm2 lowest IDSW: {low_idsw2['checkpoint']} IDSW={low_idsw2['IDSW']}")

print("\nStep A DONE")
print(f"  per_checkpoint_metrics.csv: {pc_path}")
print(f"  results_registry.csv:       {reg_path}")
print(f"  detection_outcomes.csv:     {det_path}")
