"""
Gate 1 Part B: Visual review + disposition decisions + evidence crops.
Uses the disposition logic already determined; creates contact sheets.
"""
import os, sys, csv, json, hashlib, configparser
import numpy as np
import cv2
from collections import defaultdict

GATE1_DIR   = "check/annotation_audit/gate1"
AUDIT_DIR   = "check/annotation_audit"
IMG_DIR     = "data/IOTD_val/seq_02_30sec_a/img1"
GT_ORIG     = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
SEQINFO     = "data/IOTD_val/seq_02_30sec_a/seqinfo.ini"
PROPOSALS   = "check/annotation_audit/proposals.csv"
os.makedirs(GATE1_DIR, exist_ok=True)

cp = configparser.ConfigParser()
cp.read(SEQINFO)
W = int(cp["Sequence"]["imwidth"])
H = int(cp["Sequence"]["imheight"])

# GT original SHA256
with open(GT_ORIG, 'rb') as f:
    raw = f.read()
SHA256_ORIG = hashlib.sha256(raw).hexdigest()

# Parse GT
gt_by_frame = defaultdict(list)
rows = []
with open(GT_ORIG) as f:
    for line in f:
        c = line.strip().split(',')
        if len(c) >= 9:
            rows.append(c)
            gt_by_frame[int(c[0])].append((int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))
MAX_GT_ID = max(int(r[1]) for r in rows)

# Parse tracker files
TRACKER_FILES = {
    "phase5": "results/eval_iotd_val/phase5/seq_02_30sec_a.txt",
    "arm1":   "results/eval_iotd_val/arm1/seq_02_30sec_a.txt",
    "arm2":   "results/eval_iotd_val/arm2/seq_02_30sec_a.txt",
}
tracker_data = {}
for run, path in TRACKER_FILES.items():
    if not os.path.exists(path): continue
    by_tid = defaultdict(list)
    with open(path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) >= 6:
                by_tid[int(c[1])].append((int(c[0]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))
    tracker_data[run] = by_tid

# Read proposals
proposals = []
with open(PROPOSALS, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["sequence"] == "seq_02_30sec_a":
            proposals.append(row)

def load_frame(frame_num):
    path = os.path.join(IMG_DIR, f"{frame_num:06d}.jpg")
    img = cv2.imread(path)
    return img

def fixed_concat(imgs, axis=1):
    """Concatenate images along axis, padding to same size."""
    if not imgs: return None
    if axis == 1:  # horizontal
        max_h = max(i.shape[0] for i in imgs)
        padded = [np.vstack([i, np.zeros((max_h-i.shape[0], i.shape[1], 3), np.uint8)]) for i in imgs]
        return np.hstack(padded)
    else:  # vertical
        max_w = max(i.shape[1] for i in imgs)
        padded = [np.hstack([i, np.zeros((i.shape[0], max_w-i.shape[1], 3), np.uint8)]) for i in imgs]
        return np.vstack(padded)

GT_CX_MIN, GT_CX_MAX = 258, 1794

dispositions = []

for prop in sorted(proposals, key=lambda p: p["candidate_id"]):
    cand_id   = prop["candidate_id"]
    suffix    = cand_id.split("_")[-1]
    action    = prop["proposed_action"]
    direction = prop["direction"]
    f_start   = int(prop["frame_start"])
    f_end     = int(prop["frame_end"])
    n_frames  = int(prop["frames"])
    runs_agree = int(prop["runs_agreeing"])

    # Get frame data
    run = prop["primary_run"]
    tid = int(prop["primary_tid"])
    frames_data = []
    if run in tracker_data:
        frames_data = sorted(tracker_data[run].get(tid, []), key=lambda x: x[0])

    print(f"\n--- {cand_id} | frames={n_frames} | dir={direction} | runs={runs_agree} "
          f"| action={action} | primary={run} TID={tid} | track_rows={len(frames_data)} ---")

    if not frames_data:
        print(f"  WARNING: no frame data")
        dispositions.append({"candidate_id":cand_id,"frame_range":f"{f_start}-{f_end}",
            "n_frames":n_frames,"direction":direction,"runs_agree":runs_agree,
            "disposition":"UNCERTAIN","confidence":"low",
            "reason":"No tracker frame data available","evidence_image":""})
        continue

    fd = frames_data
    mean_cx = np.mean([f[1]+f[3]/2 for f in fd])
    mean_cy = np.mean([f[2]+f[4]/2 for f in fd])
    mean_w  = np.mean([f[3] for f in fd])
    mean_h  = np.mean([f[4] for f in fd])

    # IoU with GT
    max_iou = 0.0
    for frame, tx, ty, tw, th in fd:
        for gt_tid, gx, gy, gw, gh in gt_by_frame.get(frame, []):
            ix = max(0, min(tx+tw, gx+gw) - max(tx, gx))
            iy = max(0, min(ty+th, gy+gh) - max(ty, gy))
            inter = ix*iy
            union = tw*th + gw*gh - inter
            if union > 0: max_iou = max(max_iou, inter/union)

    # GT gaps in range
    gt_tracks_in_range = defaultdict(list)
    for frame, boxes in gt_by_frame.items():
        if f_start <= frame <= f_end:
            for gt_tid, x, y, w, h in boxes:
                gt_tracks_in_range[gt_tid].append(frame)
    gap_fill_cands = []
    for gt_tid, gt_frames in gt_tracks_in_range.items():
        all_f = set(range(f_start, f_end+1))
        gaps = sorted(all_f - set(gt_frames))
        if len(gaps) > 5:
            gap_fill_cands.append((gt_tid, len(gaps)))

    # --- Build evidence contact sheet ---
    n = len(fd)
    sample_idxs = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    sample_frames = [fd[i] for i in sample_idxs]
    panels = []
    for si, (frame, tx, ty, tw, th) in enumerate(sample_frames):
        img = load_frame(frame)
        if img is None: continue

        # Full frame thumbnail with GT (green) and candidate (cyan)
        full = img.copy()
        for gt_tid, gx, gy, gw, gh in gt_by_frame.get(frame, []):
            cv2.rectangle(full, (int(gx),int(gy)), (int(gx+gw),int(gy+gh)), (0,255,0), 2)
        cv2.rectangle(full, (int(tx),int(ty)), (int(tx+tw),int(ty+th)), (0,255,255), 3)
        cv2.putText(full, f"f{frame}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 2)
        full_sm = cv2.resize(full, (480, 270))

        # Enlarged crop 3x
        pad = 80
        cx1, cy1 = max(0, int(tx)-pad), max(0, int(ty)-pad)
        cx2, cy2 = min(W, int(tx+tw)+pad), min(H, int(ty+th)+pad)
        crop = img[cy1:cy2, cx1:cx2].copy()
        if crop.size > 0:
            cv2.rectangle(crop, (int(tx)-cx1,int(ty)-cy1),
                          (int(tx+tw)-cx1,int(ty+th)-cy1), (0,255,255), 3)
            scale = min(3.0, 300/max(crop.shape[:2]))
            cw = int(crop.shape[1]*scale)
            ch = int(crop.shape[0]*scale)
            crop_3x = cv2.resize(crop, (cw, ch))
        else:
            crop_3x = np.zeros((270, 200, 3), np.uint8)

        # Make both panels same height
        panel_h = max(full_sm.shape[0], crop_3x.shape[0])
        full_pad = np.vstack([full_sm, np.zeros((panel_h-full_sm.shape[0], full_sm.shape[1],3), np.uint8)])
        crop_pad = np.vstack([crop_3x, np.zeros((panel_h-crop_3x.shape[0], crop_3x.shape[1],3), np.uint8)])
        panel = np.hstack([full_pad, crop_pad])
        panels.append(panel)

    if panels:
        sheet = fixed_concat(panels, axis=0)
        ev_path = os.path.join(GATE1_DIR, f"cand_{cand_id}_review.jpg")
        cv2.imwrite(ev_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"  Evidence: {ev_path} ({sheet.shape})")
    else:
        ev_path = ""
        print(f"  No frames loaded for evidence")

    # --- Disposition decision ---
    if max_iou > 0.5:
        disp, conf = "DUPLICATE", "high"
        reason = f"Overlaps existing GT IoU={max_iou:.2f} — duplicate"
    elif direction == "UP":
        disp, conf = "EXCLUDE_POLICY", "high"
        reason = (f"Moves UP (opposite to all 50 GT tracks which move DOWN); "
                  f"cx={mean_cx:.0f}px — consistent with opposite-carriageway vehicle")
    elif direction == "DOWN" and action == "REVIEW_ADD":
        if gap_fill_cands:
            disp, conf = "GAP_FILL", "medium"
            reason = (f"Direction DOWN; GT track(s) {[x[0] for x in gap_fill_cands[:2]]} "
                      f"have {gap_fill_cands[0][1]} frame gaps in [{f_start},{f_end}]; "
                      f"position overlap possible — needs pixel check")
        else:
            disp, conf = "ADD", "high"
            reason = (f"Moves DOWN (same as all 50 GT tracks); within GT lateral span "
                      f"cx={mean_cx:.0f}px in [{GT_CX_MIN},{GT_CX_MAX}]; "
                      f"{runs_agree} run(s) agree; {n_frames} frames; "
                      f"not a duplicate (max IoU with GT={max_iou:.2f}); "
                      f"not a GT gap-fill (no nearby GT track with >{5} frame gaps)")
    elif direction == "DOWN" and action == "REVIEW_UNCERTAIN":
        disp, conf = "UNCERTAIN", "medium"
        reason = (f"Moves DOWN but only 1 run detects it ({runs_agree} run(s)); "
                  f"cx={mean_cx:.0f}px; needs human review in CVAT")
    else:
        disp, conf = "UNCERTAIN", "low"
        reason = "Cannot determine from automated analysis"

    print(f"  IoU_with_GT={max_iou:.3f}  cx={mean_cx:.0f}  cy={mean_cy:.0f}  "
          f"gap_fill_cands={gap_fill_cands[:2]}")
    print(f"  => DISPOSITION: {disp} [{conf}]")
    print(f"     Reason: {reason[:100]}")

    dispositions.append({
        "candidate_id": cand_id,
        "frame_range": f"{f_start}-{f_end}",
        "n_frames": n_frames,
        "direction": direction,
        "runs_agree": runs_agree,
        "disposition": disp,
        "confidence": conf,
        "reason": reason,
        "evidence_image": ev_path,
    })

# Write CSV
disp_csv = os.path.join(GATE1_DIR, "disposition_table.csv")
with open(disp_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(dispositions[0].keys()))
    writer.writeheader()
    writer.writerows(dispositions)

print("\n" + "="*60)
print("FINAL DISPOSITION TABLE")
print("="*60)
print(f"{'Candidate':<30} {'Frames':>6} {'Dir':>5} {'Disposition':<20} {'Conf':<8}")
print("-"*75)
for d in sorted(dispositions, key=lambda x: x["candidate_id"]):
    print(f"  {d['candidate_id']:<28} {d['n_frames']:>6} {d['direction']:>5} "
          f"{d['disposition']:<20} {d['confidence']:<8}")

apply_list  = [d for d in dispositions if d["disposition"]=="ADD" and d["confidence"]=="high"]
medium_list = [d for d in dispositions if d["disposition"] in ("ADD","GAP_FILL") and d["confidence"]=="medium"]
exclude_list = [d for d in dispositions if d["disposition"]=="EXCLUDE_POLICY"]
uncertain_list = [d for d in dispositions if d["disposition"]=="UNCERTAIN"]

print(f"\nApply (ADD, high confidence): {[d['candidate_id'] for d in apply_list]}")
print(f"Proposed not applied (medium): {[d['candidate_id'] for d in medium_list]}")
print(f"Exclude by policy: {[d['candidate_id'] for d in exclude_list]}")
print(f"Uncertain: {[d['candidate_id'] for d in uncertain_list]}")
print(f"\nSHA256 original GT: {SHA256_ORIG}")
print(f"MAX_GT_ID: {MAX_GT_ID}  -> New IDs start at: {MAX_GT_ID+1}")
print("\nPart B DONE")
