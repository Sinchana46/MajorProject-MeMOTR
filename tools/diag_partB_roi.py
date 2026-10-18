"""
Part B: ROI hull from annotations, FP classification, ROI-filtered TrackEval, renders.
"""
import os, sys, json
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import subprocess

GT_TXT        = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
IMG_DIR       = "data/IOTD_val/seq_02_30sec_a/img1"
SEQINFO       = "data/IOTD_val/seq_02_30sec_a/seqinfo.ini"
EVAL_GT_DIR   = "results/eval_iotd_val/gt_eval"
TRACKER_DIRS  = {
    "phase5": "results/eval_iotd_val/phase5",
    "arm1":   "results/eval_iotd_val/arm1",
    "arm2":   "results/eval_iotd_val/arm2",
}
ROI_OUT_ROOT  = "results/eval_iotd_val/diag/roi"
RENDER_OUT    = "check/eval_diag"
os.makedirs(ROI_OUT_ROOT, exist_ok=True)
os.makedirs(RENDER_OUT, exist_ok=True)

# Read seqinfo
import configparser
cp = configparser.ConfigParser()
cp.read(SEQINFO)
W = int(cp["Sequence"]["imwidth"])
H = int(cp["Sequence"]["imheight"])
N_FRAMES = int(cp["Sequence"]["seqlength"])
print(f"Sequence: {N_FRAMES} frames, {W}x{H}")

# ─── B.1: Build convex hull from GT bottom-centre points ────────────────────
print("\n=== PART B.1: ROI CONVEX HULL ===")
pts = []
gt_by_frame = {}
with open(GT_TXT) as f:
    for line in f:
        cols = line.strip().split(",")
        if len(cols) < 6: continue
        frame = int(cols[0])
        x, y, w, h = float(cols[2]), float(cols[3]), float(cols[4]), float(cols[5])
        cx = x + w / 2.0
        cy = y + h
        pts.append([cx, cy])
        if frame not in gt_by_frame:
            gt_by_frame[frame] = []
        gt_by_frame[frame].append([x, y, w, h])

pts = np.array(pts, dtype=np.float64)
hull = ConvexHull(pts)
hull_pts = pts[hull.vertices]  # shape (K, 2)

# Dilate: expand each point away from centroid by 5% of frame width
dilation = 0.05 * W
centroid = hull_pts.mean(axis=0)
directions = hull_pts - centroid
norms = np.linalg.norm(directions, axis=1, keepdims=True)
dilated_hull = hull_pts + (directions / (norms + 1e-9)) * dilation
# Close hull
hull_poly = dilated_hull.astype(np.int32)

hull_area = cv2.contourArea(hull_poly)
frame_area = W * H
print(f"Hull vertices: {len(hull_poly)}")
print(f"Hull polygon (x, y): {hull_poly.tolist()}")
print(f"Hull area: {hull_area:.0f} px ({hull_area/frame_area*100:.1f}% of frame)")

def pt_in_hull(px, py, poly):
    """Return True if (px,py) is inside (or on border of) the polygon."""
    pt = (float(px), float(py))
    result = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), pt, False)
    return result >= 0

# ─── B.2: Classify FP per run ────────────────────────────────────────────────
print("\n=== PART B.2: FP CLASSIFICATION ===")

def iou(a, b):
    """IoU between [x,y,w,h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0]+b[2], b[1]+b[3]
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0

run_stats = {}
for run_name, tracker_dir in TRACKER_DIRS.items():
    tracker_file = os.path.join(tracker_dir, "seq_02_30sec_a.txt")
    tracker_by_frame = {}
    with open(tracker_file) as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 6: continue
            frame = int(cols[0])
            x, y, w, h = float(cols[2]), float(cols[3]), float(cols[4]), float(cols[5])
            if frame not in tracker_by_frame:
                tracker_by_frame[frame] = []
            tracker_by_frame[frame].append([x, y, w, h])

    tp, fp_in, fp_out = 0, 0, 0
    for frame in sorted(set(list(gt_by_frame.keys()) + list(tracker_by_frame.keys()))):
        gts = gt_by_frame.get(frame, [])
        preds = tracker_by_frame.get(frame, [])
        if not preds:
            continue
        if not gts:
            for p in preds:
                cx = p[0] + p[2]/2
                cy = p[1] + p[3]
                if pt_in_hull(cx, cy, hull_poly): fp_in += 1
                else: fp_out += 1
            continue
        # Hungarian via greedy IoU (acceptable for diagnostic)
        matched_gt = set()
        matched_pred = set()
        iou_matrix = np.zeros((len(preds), len(gts)))
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                iou_matrix[pi, gi] = iou(p, g)
        # Greedy match
        for _ in range(min(len(preds), len(gts))):
            if iou_matrix.max() < 0.5: break
            pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_pred.add(pi); matched_gt.add(gi)
            tp += 1
            iou_matrix[pi, :] = -1
            iou_matrix[:, gi] = -1
        for pi, p in enumerate(preds):
            if pi not in matched_pred:
                cx = p[0] + p[2]/2
                cy = p[1] + p[3]
                if pt_in_hull(cx, cy, hull_poly): fp_in += 1
                else: fp_out += 1

    total_fp = fp_in + fp_out
    run_stats[run_name] = {"tp": tp, "fp_in": fp_in, "fp_out": fp_out, "total_fp": total_fp}
    pct = fp_out/total_fp*100 if total_fp > 0 else 0
    print(f"  {run_name}: TP={tp}  FP_inside={fp_in}  FP_outside={fp_out}  "
          f"({pct:.1f}% of FPs outside hull)")

# ─── B.3: ROI-filtered outputs ──────────────────────────────────────────────
print("\n=== PART B.3: ROI-FILTERED TRACKER OUTPUTS ===")
roi_tracker_dirs = {}
for run_name, tracker_dir in TRACKER_DIRS.items():
    roi_dir = os.path.join(ROI_OUT_ROOT, run_name)
    os.makedirs(roi_dir, exist_ok=True)
    tracker_file = os.path.join(tracker_dir, "seq_02_30sec_a.txt")
    out_file = os.path.join(roi_dir, "seq_02_30sec_a.txt")
    kept, removed = 0, 0
    with open(tracker_file) as fin, open(out_file, "w") as fout:
        for line in fin:
            cols = line.strip().split(",")
            if len(cols) < 6:
                fout.write(line); kept += 1; continue
            x, y, w, h = float(cols[2]), float(cols[3]), float(cols[4]), float(cols[5])
            cx = x + w/2
            cy = y + h
            if pt_in_hull(cx, cy, hull_poly):
                fout.write(line); kept += 1
            else:
                removed += 1
    print(f"  {run_name}: kept={kept}  removed={removed}  -> {out_file}")
    roi_tracker_dirs[run_name] = roi_dir

# Run TrackEval on ROI-filtered outputs
print("\n=== PART B.3: TRACKEVAL ON ROI-FILTERED ===")
roi_te_results = {}
for run_name, roi_dir in roi_tracker_dirs.items():
    te_out = os.path.join("results/eval_iotd_val/diag/roi", f"trackeval_{run_name}")
    label = f"ROI_{run_name.upper()}"
    cmd = [sys.executable, "-u", "tools/run_trackeval.py",
           "--tracker-dir", roi_dir,
           "--gt-dirs", EVAL_GT_DIR,
           "--output-dir", te_out,
           "--label", label,
           "--num-cores", "4"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # Find summary
    summary_path = None
    for root, dirs, files in os.walk(te_out):
        for fn in files:
            if "pedestrian_summary" in fn.lower():
                summary_path = os.path.join(root, fn)
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            lines = f.readlines()
        vals = dict(zip(lines[0].split(), lines[1].split()))
        roi_te_results[run_name] = vals
        print(f"  {run_name}: HOTA={vals.get('HOTA','?')} MOTA={vals.get('MOTA','?')} "
              f"FP={vals.get('CLR_FP','?')} FN={vals.get('CLR_FN','?')} "
              f"Recall={vals.get('CLR_Re','?')} Prec={vals.get('CLR_Pr','?')}")
    else:
        print(f"  {run_name}: TRACKEVAL FAILED (check stderr)")
        print(result.stderr[-500:] if result.stderr else "")
        roi_te_results[run_name] = {}

# ─── B.4: Render 6 evenly spaced frames per run ─────────────────────────────
print("\n=== PART B.4: RENDERING ===")
render_frames = [int(N_FRAMES * i / 5) + 1 for i in range(6)]
render_frames = [min(max(f, 1), N_FRAMES) for f in render_frames]
print(f"Render frames: {render_frames}")

for run_name, tracker_dir in TRACKER_DIRS.items():
    tracker_file = os.path.join(tracker_dir, "seq_02_30sec_a.txt")
    tracker_by_frame = {}
    with open(tracker_file) as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 6: continue
            frame = int(cols[0])
            if frame not in tracker_by_frame:
                tracker_by_frame[frame] = []
            tracker_by_frame[frame].append(list(map(float, cols[2:6])))

    for frame_idx in render_frames:
        fname = f"{frame_idx:06d}.jpg"
        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(img_path):
            print(f"  {run_name} frame {fname}: image not found")
            continue
        img = cv2.imread(img_path)

        # Draw hull (yellow)
        cv2.polylines(img, [hull_poly.reshape(-1,1,2)], isClosed=True,
                      color=(0,255,255), thickness=2)

        # Draw GT boxes (green)
        for g in gt_by_frame.get(frame_idx, []):
            x,y,w,h = int(g[0]),int(g[1]),int(g[2]),int(g[3])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Draw tracker boxes (red)
        for t in tracker_by_frame.get(frame_idx, []):
            x,y,w,h = int(t[0]),int(t[1]),int(t[2]),int(t[3])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

        # Label
        cv2.putText(img, f"{run_name} f{frame_idx:06d}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
        cv2.putText(img, "GT=green  Tracker=red  Hull=yellow", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

        out_path = os.path.join(RENDER_OUT, f"render_{run_name}_f{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"  Saved: {out_path}")

# ─── Final print ───────────────────────────────────────────────────────────
print("\n=== PART B: FP SUMMARY ===")
for run_name in ["phase5","arm1","arm2"]:
    s = run_stats[run_name]
    pct = s["fp_out"]/s["total_fp"]*100 if s["total_fp"]>0 else 0
    print(f"  {run_name}: TP={s['tp']}  FP_inside={s['fp_in']}  FP_outside={s['fp_out']}  "
          f"({pct:.1f}% outside hull)")

print("\n=== PART B: ROI-FILTERED METRICS ===")
full_results = {
    "phase5": {"HOTA":"50.51","DetA":"43.88","AssA":"58.49","MOTA":"29.10","IDF1":"68.84",
               "IDSW":"1","CLR_FP":"2117","CLR_FN":"847","CLR_Re":"79.747","CLR_Pr":"61.17"},
    "arm1":   {"HOTA":"49.58","DetA":"45.70","AssA":"54.04","MOTA":"24.94","IDF1":"69.66",
               "IDSW":"7","CLR_FP":"2855","CLR_FN":"277","CLR_Re":"93.376","CLR_Pr":"57.766"},
    "arm2":   {"HOTA":"42.84","DetA":"38.32","AssA":"48.01","MOTA":"-6.58","IDF1":"63.72",
               "IDSW":"6","CLR_FP":"4266","CLR_FN":"185","CLR_Re":"95.576","CLR_Pr":"48.372"},
}
print(f"{'Run':<10}{'Type':<10}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'MOTA':>8}{'IDF1':>7}{'FP':>7}{'FN':>7}{'Recall':>8}{'Prec':>8}")
print("-"*90)
for run_name in ["phase5","arm1","arm2"]:
    fr = full_results[run_name]
    print(f"  {run_name:<8}{'full':<10}{fr['HOTA']:>7}{fr['DetA']:>7}{fr['AssA']:>7}"
          f"{fr['MOTA']:>8}{fr['IDF1']:>7}{fr['CLR_FP']:>7}{fr['CLR_FN']:>7}"
          f"{float(fr['CLR_Re']):>8.2f}{float(fr['CLR_Pr']):>8.2f}")
    rv = roi_te_results.get(run_name, {})
    if rv:
        print(f"  {run_name:<8}{'ROI':<10}"
              f"{rv.get('HOTA','?'):>7}{rv.get('DetA','?'):>7}{rv.get('AssA','?'):>7}"
              f"{rv.get('MOTA','?'):>8}{rv.get('IDF1','?'):>7}"
              f"{rv.get('CLR_FP','?'):>7}{rv.get('CLR_FN','?'):>7}"
              f"{float(rv.get('CLR_Re','0')):>8.2f}{float(rv.get('CLR_Pr','0')):>8.2f}")
    else:
        print(f"  {run_name:<8}{'ROI':<10} FAILED")
    print()

print("\nDONE Part B")
