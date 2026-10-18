"""
Part C: Threshold sweep. Confidence column is constant=1 (internal tracker logic),
so rerun inference at det_thresh 0.7 and 0.8 (track_thresh same) for all 3 final checkpoints.
Then run TrackEval for each. Print table for full and ROI-filtered outputs.
"""
import subprocess, sys, os

DATA_DIR = "data/IOTD_val"
EVAL_GT  = "results/eval_iotd_val/gt_eval"
PYTHON   = sys.executable

FINAL_CKPTS = [
    ("phase5", "results/phase5_interim_baseline/checkpoint_4.pth",
                "configs/train_iotd_real.yaml"),
    ("arm1",   "results/train_iotd_real/checkpoint_4.pth",
                "configs/train_iotd_real.yaml"),
    ("arm2",   "results/train_iotd_real_aug/checkpoint_4.pth",
                "configs/train_iotd_real_aug.yaml"),
]
THRESHOLDS = [0.7, 0.8]
GT_HULL_FILE = "results/eval_iotd_val/diag/roi/hull_poly.npy"

# Load hull for ROI filtering
import numpy as np
import cv2

GT_TXT   = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
SEQINFO  = "data/IOTD_val/seq_02_30sec_a/seqinfo.ini"
import configparser
cp = configparser.ConfigParser(); cp.read(SEQINFO)
W, H = int(cp["Sequence"]["imwidth"]), int(cp["Sequence"]["imheight"])

# Rebuild hull (same as Part B)
from scipy.spatial import ConvexHull
pts = []
with open(GT_TXT) as f:
    for line in f:
        cols = line.strip().split(",")
        if len(cols) < 6: continue
        x,y,w,h_ = float(cols[2]),float(cols[3]),float(cols[4]),float(cols[5])
        pts.append([x+w/2, y+h_])
pts = np.array(pts, dtype=np.float64)
hull = ConvexHull(pts)
hull_pts = pts[hull.vertices]
dilation = 0.05 * W
centroid  = hull_pts.mean(axis=0)
dirs = hull_pts - centroid
norms = np.linalg.norm(dirs, axis=1, keepdims=True)
dilated_hull = (hull_pts + (dirs/(norms+1e-9)) * dilation).astype(np.int32)
hull_poly = dilated_hull

def pt_in_hull(px, py):
    return cv2.pointPolygonTest(hull_poly.reshape(-1,1,2), (float(px),float(py)), False) >= 0

results_table = {}  # (run, thresh) -> {HOTA, DetA, AssA, MOTA, IDF1, CLR_FP, CLR_FN, CLR_Re, CLR_Pr}

for thresh in THRESHOLDS:
    thresh_str = f"{thresh:.1f}".replace(".","p")
    for run_name, ckpt, cfg in FINAL_CKPTS:
        label = f"{run_name}_t{thresh_str}"
        out_dir  = f"results/eval_iotd_val/diag/thresh_{thresh_str}/{run_name}"
        out_file = os.path.join(out_dir, "seq_02_30sec_a.txt")

        if not os.path.exists(out_file):
            os.makedirs(out_dir, exist_ok=True)
            cmd = [PYTHON, "-u", "tools/run_inference_mot.py",
                   "--checkpoint", ckpt, "--split-dirs", DATA_DIR,
                   "--output-dir", out_dir, "--config", cfg,
                   "--label", label.upper(),
                   "--det-thresh", str(thresh),
                   "--track-thresh", str(thresh), "--miss-tol", "30"]
            print(f"\n[{label}] Running inference (thresh={thresh})...")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            lines = [l for l in (r.stdout+r.stderr).split("\n")
                     if "detection" in l or "Done" in l or "Error" in l]
            print(f"  {lines[-2:]}")
        else:
            with open(out_file) as f:
                rows = sum(1 for l in f if l.strip())
            print(f"[{label}] Already exists ({rows} rows) - SKIP")

        if not os.path.exists(out_file):
            print(f"  ERROR: {out_file} not created"); continue

        # Run TrackEval - full
        te_out = f"results/eval_iotd_val/diag/thresh_{thresh_str}/trackeval_{run_name}"
        cmd = [PYTHON, "-u", "tools/run_trackeval.py",
               "--tracker-dir", out_dir,
               "--gt-dirs", EVAL_GT,
               "--output-dir", te_out,
               "--label", label.upper(), "--num-cores", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        summary = None
        for root, dirs, files in os.walk(te_out):
            for fn in files:
                if "pedestrian_summary" in fn.lower():
                    with open(os.path.join(root,fn)) as f:
                        lns = f.readlines()
                    summary = dict(zip(lns[0].split(), lns[1].split()))
        if summary:
            results_table[(run_name, thresh, "full")] = summary
            print(f"  [{label} full] HOTA={summary.get('HOTA','?')} MOTA={summary.get('MOTA','?')} "
                  f"FP={summary.get('CLR_FP','?')} FN={summary.get('CLR_FN','?')} "
                  f"Prec={summary.get('CLR_Pr','?')}")

        # ROI filter + TrackEval
        roi_dir  = f"results/eval_iotd_val/diag/thresh_{thresh_str}/roi_{run_name}"
        roi_file = os.path.join(roi_dir, "seq_02_30sec_a.txt")
        os.makedirs(roi_dir, exist_ok=True)
        kept = 0
        with open(out_file) as fin, open(roi_file, "w") as fout:
            for line in fin:
                cols = line.strip().split(",")
                if len(cols) < 6:
                    fout.write(line); kept+=1; continue
                x,y,w,h_ = float(cols[2]),float(cols[3]),float(cols[4]),float(cols[5])
                if pt_in_hull(x+w/2, y+h_):
                    fout.write(line); kept+=1

        roi_te_out = f"results/eval_iotd_val/diag/thresh_{thresh_str}/trackeval_roi_{run_name}"
        cmd = [PYTHON, "-u", "tools/run_trackeval.py",
               "--tracker-dir", roi_dir,
               "--gt-dirs", EVAL_GT,
               "--output-dir", roi_te_out,
               "--label", label.upper()+"_ROI", "--num-cores", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        roi_summary = None
        for root, dirs, files in os.walk(roi_te_out):
            for fn in files:
                if "pedestrian_summary" in fn.lower():
                    with open(os.path.join(root,fn)) as f:
                        lns = f.readlines()
                    roi_summary = dict(zip(lns[0].split(), lns[1].split()))
        if roi_summary:
            results_table[(run_name, thresh, "roi")] = roi_summary
            print(f"  [{label} ROI]  HOTA={roi_summary.get('HOTA','?')} MOTA={roi_summary.get('MOTA','?')} "
                  f"FP={roi_summary.get('CLR_FP','?')} FN={roi_summary.get('CLR_FN','?')} "
                  f"Prec={roi_summary.get('CLR_Pr','?')}")

# Print consolidated table
print("\n" + "="*110)
print("PART C: THRESHOLD SWEEP TABLE")
print("="*110)
print(f"{'Run':<10}{'Thresh':>7}{'Type':<6}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'MOTA':>8}{'IDF1':>7}{'FP':>7}{'FN':>6}{'Recall':>8}{'Prec':>8}")
print("-"*110)
# Baseline at 0.5
baseline = [
    ("phase5",0.5,"full","50.51","43.88","58.49","29.10","68.84","2117","847","79.75","61.17"),
    ("arm1",  0.5,"full","49.58","45.70","54.04","24.94","69.66","2855","277","93.38","57.77"),
    ("arm2",  0.5,"full","42.84","38.32","48.01","-6.58","63.72","4266","185","95.58","48.37"),
    ("phase5",0.5,"roi", "52.25","46.89","58.60","39.00","71.93","1703","847","79.75","66.20"),
    ("arm1",  0.5,"roi", "49.69","45.77","54.19","25.20","69.73","2844","277","93.38","57.86"),
    ("arm2",  0.5,"roi", "43.14","38.68","48.25","-4.66","64.13","4186","185","95.58","48.84"),
]
for run,th,typ,h,da,aa,mo,i1,fp,fn,re,pr in sorted(baseline,key=lambda x:(x[0],x[1])):
    print(f"  {run:<8}{th:>7.1f}{typ:<6}{h:>7}{da:>7}{aa:>7}{mo:>8}{i1:>7}{fp:>7}{fn:>6}{float(re):>8.2f}{float(pr):>8.2f}")
for (run,thresh,typ), vals in sorted(results_table.items(), key=lambda x:(x[0][0],x[0][1])):
    print(f"  {run:<8}{thresh:>7.1f}{typ:<6}"
          f"{vals.get('HOTA','?'):>7}{vals.get('DetA','?'):>7}{vals.get('AssA','?'):>7}"
          f"{vals.get('MOTA','?'):>8}{vals.get('IDF1','?'):>7}"
          f"{vals.get('CLR_FP','?'):>7}{vals.get('CLR_FN','?'):>6}"
          f"{float(vals.get('CLR_Re','0')):>8.2f}{float(vals.get('CLR_Pr','0')):>8.2f}")

print("\nPart C DONE")
