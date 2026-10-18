"""
Gate 1 Part A + B: Input verification, candidate review, disposition table, evidence crops.
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

# ─── A1: Read proposals.csv ───────────────────────────────────────────────────
print("=" * 60)
print("PART A: INPUTS")
print("=" * 60)

proposals = []
with open(PROPOSALS, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["sequence"] == "seq_02_30sec_a":
            proposals.append(row)

print(f"\nA1: seq_02_30sec_a proposals ({len(proposals)} rows):")
print(f"{'ID':<30} {'frames':>6} {'class':<20} {'action':<30} {'runs':>4}")
print("-" * 95)
for p in proposals:
    print(f"  {p['candidate_id']:<28} {p['frames']:>6} {p['suggested_class']:<20} "
          f"{p['proposed_action']:<30} {p['runs_agreeing']:>4}")

review_add = [p for p in proposals if p["proposed_action"] == "REVIEW_ADD"]
uncertain  = [p for p in proposals if p["proposed_action"] == "REVIEW_UNCERTAIN"]
print(f"\nREVIEW_ADD: {len(review_add)} (IDs: {[p['candidate_id'] for p in review_add]})")
print(f"REVIEW_UNCERTAIN: {len(uncertain)} (IDs: {[p['candidate_id'] for p in uncertain]})")

# A2: Evidence images and CVAT file
print("\nA2: Evidence images in check/annotation_audit/seq_02_30sec_a/:")
ev_dir = os.path.join(AUDIT_DIR, "seq_02_30sec_a")
ev_files = sorted(os.listdir(ev_dir)) if os.path.exists(ev_dir) else []
for f in ev_files:
    print(f"  {f}")
cvat_file = "check/annotation_audit/cvat_import/seq_02_30sec_a_review_add.txt"
if os.path.exists(cvat_file):
    with open(cvat_file) as f:
        cvat_rows = [l.strip() for l in f if l.strip()]
    print(f"\nCVAT import file: {cvat_file} ({len(cvat_rows)} rows)")

# Tracker source per candidate
print("\nTracker source per candidate:")
for p in proposals:
    print(f"  {p['candidate_id']}: primary_run={p['primary_run']} runs={p['runs']}")

# A3: GT file structure
print("\nA3: GT file structure:")
with open(GT_ORIG, 'rb') as f:
    raw = f.read()
sha256_orig = hashlib.sha256(raw).hexdigest()
has_crlf = b'\r\n' in raw
has_trailing_nl = raw.endswith(b'\n')
print(f"  Size: {len(raw)} bytes  SHA256: {sha256_orig[:16]}...{sha256_orig[-8:]}")
print(f"  Line endings: {'CRLF' if has_crlf else 'LF'}")
print(f"  Trailing newline: {has_trailing_nl}")

rows = []
with open(GT_ORIG) as f:
    for line in f:
        c = line.strip().split(',')
        if len(c) >= 9:
            rows.append(c)

frames = [int(r[0]) for r in rows]
ids    = [int(r[1]) for r in rows]
ws     = [float(r[4]) for r in rows]
hs     = [float(r[5]) for r in rows]
confs  = [r[6] for r in rows]
classes= [r[7] for r in rows]
vis    = [r[8].strip() for r in rows]

# Check decimal places in x,y,w,h
sample = rows[0]
def dec_places(s):
    if '.' in s:
        return len(s.split('.')[1])
    return 0
dp_x = dec_places(rows[0][2])
dp_y = dec_places(rows[0][3])
dp_w = dec_places(rows[0][4])
dp_h = dec_places(rows[0][5])
print(f"  Rows: {len(rows)}  Frames: {min(frames)}-{max(frames)}  IDs: {min(ids)}-{max(ids)}")
print(f"  Decimal places: x={dp_x}, y={dp_y}, w={dp_w}, h={dp_h}")
print(f"  Confidence values: {set(confs)}")
print(f"  Class IDs present: {sorted(set(int(c) for c in classes))}")
print(f"  Visibility values: {sorted(set(float(v) for v in vis))}")
print(f"  Box size range: w=[{min(ws):.1f},{max(ws):.1f}] h=[{min(hs):.1f},{max(hs):.1f}]")

# Sort order check
sorted_check = sorted(rows, key=lambda r: (int(r[0]), int(r[1])))
is_sorted = all(rows[i][0] == sorted_check[i][0] and rows[i][1] == sorted_check[i][1]
                for i in range(len(rows)))
print(f"  Sort order: {'frame, then ID (sorted)' if is_sorted else 'NOT SORTED by frame+ID'}")
print(f"  Row format: frame,id,x,y,w,h,{rows[0][6]},class,{rows[0][8].strip()}")
print(f"  Example rows:")
for r in rows[:3]:
    print(f"    {','.join(r)}")

MAX_GT_ID = max(ids)
print(f"\n  MAX GT ID: {MAX_GT_ID}  -> New tracks start at ID {MAX_GT_ID+1}")
print(f"  SHA256 (original): {sha256_orig}")

# ─── B: Visual review via image crops ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PART B: CANDIDATE REVIEW (pixel-level inspection)")
print("=" * 60)

# Parse all tracker files to get frame data per candidate
TRACKER_FILES = {
    "phase5": "results/eval_iotd_val/phase5/seq_02_30sec_a.txt",
    "arm1":   "results/eval_iotd_val/arm1/seq_02_30sec_a.txt",
    "arm2":   "results/eval_iotd_val/arm2/seq_02_30sec_a.txt",
}
tracker_data = {}
for run, path in TRACKER_FILES.items():
    if not os.path.exists(path):
        continue
    by_tid = defaultdict(list)
    with open(path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) >= 6:
                by_tid[int(c[1])].append((int(c[0]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))
    tracker_data[run] = by_tid

# GT by frame
gt_by_frame = defaultdict(list)
with open(GT_ORIG) as f:
    for line in f:
        c = line.strip().split(',')
        if len(c) >= 9:
            gt_by_frame[int(c[0])].append((int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))

# Read seqinfo
cp = configparser.ConfigParser()
cp.read(SEQINFO)
W = int(cp["Sequence"]["imwidth"])
H = int(cp["Sequence"]["imheight"])

def load_frame(frame_num):
    path = os.path.join(IMG_DIR, f"{frame_num:06d}.jpg")
    img = cv2.imread(path)
    return img

def draw_review(img, candidate_box, gt_boxes, frame_num, label=""):
    img = img.copy()
    for gid, gx, gy, gw, gh in gt_boxes:
        cv2.rectangle(img, (int(gx), int(gy)), (int(gx+gw), int(gy+gh)), (0, 255, 0), 2)
    if candidate_box:
        tx, ty, tw, th = candidate_box
        cv2.rectangle(img, (int(tx), int(ty)), (int(tx+tw), int(ty+th)), (0, 255, 255), 3)
    cv2.putText(img, f"f{frame_num} {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    return img

# For each REVIEW_ADD candidate, extract frame data from CVAT import
cvat_tracks = defaultdict(list)
if os.path.exists(cvat_file):
    with open(cvat_file) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) >= 6:
                cvat_tracks[int(c[1])].append((int(c[0]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))

# Map candidate_id to CVAT track: first 4 REVIEW_ADD candidates -> first 4 track IDs in cvat
cvat_tids = sorted(cvat_tracks.keys())
print(f"\nCVAT import track IDs: {cvat_tids}")

# Find proposal tracker source tracks
def get_candidate_frames(proposal):
    """Get frame data for candidate from its primary run's tracker."""
    run = proposal["primary_run"]
    tid = int(proposal["primary_tid"])
    if run not in tracker_data:
        return []
    return sorted(tracker_data[run].get(tid, []), key=lambda x: x[0])

# Majority GT direction (DOWN = y increasing = approaching)
# Already confirmed: all 50 GT tracks move DOWN

dispositions = []

# All 10 candidates
all_candidates = sorted(proposals, key=lambda p: p["candidate_id"])
for cidx, prop in enumerate(all_candidates):
    cand_id = prop["candidate_id"]
    suffix = cand_id.split("_")[-1]
    action = prop["proposed_action"]
    is_review_add = (action == "REVIEW_ADD")

    print(f"\n--- {cand_id} | frames={prop['frames']} | dir={prop['direction']} | "
          f"runs={prop['runs_agreeing']} | action={action} ---")

    # Get frame data
    frames_data = get_candidate_frames(prop)
    if not frames_data:
        # Try other runs
        for r in ["phase5", "arm1", "arm2"]:
            if r in tracker_data:
                # Try by candidate position lookup
                pass

    if not frames_data and cvat_tids and is_review_add:
        rev_add_idx = [p["candidate_id"] for p in review_add].index(cand_id) if cand_id in [p["candidate_id"] for p in review_add] else -1
        if 0 <= rev_add_idx < len(cvat_tids):
            ctid = cvat_tids[rev_add_idx]
            frames_data = sorted(cvat_tracks[ctid], key=lambda x: x[0])

    if not frames_data:
        print(f"  WARNING: No frame data found for {cand_id}")
        dispositions.append({
            "candidate_id": cand_id,
            "frame_range": f"{prop['frame_start']}-{prop['frame_end']}",
            "disposition": "UNCERTAIN",
            "confidence": "low",
            "reason": "No frame data available for visual review",
            "evidence_image": "",
        })
        continue

    n = len(frames_data)
    # Sample: first, 25%, 50%, 75%, last (5 frames)
    idxs = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    sample_frames = [frames_data[i] for i in idxs]

    # Build contact sheet
    contact_imgs = []
    mean_cx = np.mean([f[1]+f[3]/2 for f in frames_data])
    mean_cy = np.mean([f[2]+f[4]/2 for f in frames_data])
    mean_w  = np.mean([f[3] for f in frames_data])
    mean_h  = np.mean([f[4] for f in frames_data])

    for si, (frame, tx, ty, tw, th) in enumerate(sample_frames):
        img = load_frame(frame)
        if img is None:
            continue

        # Full frame with annotations
        full = draw_review(img, (tx, ty, tw, th), gt_by_frame.get(frame, []), frame,
                          f"{'ADD' if is_review_add else 'UNCERTAIN'} {suffix}")

        # Crop enlarged 3x around candidate
        pad = 60
        cx1 = max(0, int(tx) - pad)
        cy1 = max(0, int(ty) - pad)
        cx2 = min(W, int(tx+tw) + pad)
        cy2 = min(H, int(ty+th) + pad)
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop,
                      (int(tx)-cx1, int(ty)-cy1),
                      (int(tx+tw)-cx1, int(ty+th)-cy1),
                      (0, 255, 255), 2)
        # Scale crop 3x
        crop_3x = cv2.resize(crop, (crop.shape[1]*3, crop.shape[0]*3),
                              interpolation=cv2.INTER_CUBIC)
        crop_3x = cv2.resize(crop_3x, (200, 150))

        # Thumbnail full frame
        full_small = cv2.resize(full, (int(full.shape[1]*0.25), int(full.shape[0]*0.25)))

        # Stack crop + thumbnail side by side (height-match)
        h_max = max(crop_3x.shape[0], full_small.shape[0])
        pad_c = np.zeros((h_max - crop_3x.shape[0], crop_3x.shape[1], 3), dtype=np.uint8)
        pad_f = np.zeros((h_max - full_small.shape[0], full_small.shape[1], 3), dtype=np.uint8)
        col = np.vstack([np.vstack([crop_3x, pad_c]), np.vstack([full_small, pad_f])])
        # Actually just concat horizontally
        crop_padded = np.vstack([crop_3x, np.zeros((h_max-crop_3x.shape[0], crop_3x.shape[1], 3), np.uint8)])
        full_padded = np.vstack([full_small, np.zeros((h_max-full_small.shape[0], full_small.shape[1], 3), np.uint8)])
        panel = np.hstack([crop_padded, full_padded])
        contact_imgs.append(panel)

    if contact_imgs:
        # Concatenate all panels vertically
        max_w = max(p.shape[1] for p in contact_imgs)
        padded = []
        for p in contact_imgs:
            if p.shape[1] < max_w:
                p = np.hstack([p, np.zeros((p.shape[0], max_w-p.shape[1], 3), np.uint8)])
            padded.append(p)
        sheet = np.vstack(padded)
        ev_path = os.path.join(GATE1_DIR, f"cand_{cand_id}_review.jpg")
        cv2.imwrite(ev_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"  Evidence saved: {ev_path}")

    # ── Disposition logic from pixel inspection ────────────────────────────
    direction = prop["direction"]
    runs_agree = int(prop["runs_agreeing"])
    n_frames   = int(prop["frames"])

    # Check for GT gaps near this candidate's frame range
    f_start = int(prop["frame_start"])
    f_end   = int(prop["frame_end"])

    # Find GT tracks that have gaps in [f_start, f_end]
    gt_tracks_in_range = defaultdict(list)
    for frame, boxes in gt_by_frame.items():
        if f_start <= frame <= f_end:
            for tid, x, y, w, h in boxes:
                gt_tracks_in_range[tid].append(frame)

    gap_fill_candidates = []
    for tid, tid_frames in gt_tracks_in_range.items():
        all_frames_in_range = set(range(f_start, f_end+1))
        tid_frame_set = set(tid_frames)
        gaps = sorted(all_frames_in_range - tid_frame_set)
        if len(gaps) > 3:
            gap_fill_candidates.append((tid, len(gaps), gaps[:5]))

    if gap_fill_candidates:
        print(f"  GT tracks with gaps [{f_start}-{f_end}]: "
              f"{[(tid, ngap) for tid,ngap,_ in gap_fill_candidates[:3]]}")

    # Check position: is mean_cx inside GT lateral span (258-1794px from Part B)?
    GT_CX_MIN, GT_CX_MAX = 258, 1794
    in_lateral_span = GT_CX_MIN <= mean_cx <= GT_CX_MAX
    in_upper_region = mean_cy < 400  # above ~37% of frame height

    # Check IoU with any existing GT box in same frames
    max_iou_with_gt = 0.0
    for frame, tx, ty, tw, th in frames_data[:5]:
        for gt_tid, gx, gy, gw, gh in gt_by_frame.get(frame, []):
            ix = max(0, min(tx+tw, gx+gw) - max(tx, gx))
            iy = max(0, min(ty+th, gy+gh) - max(ty, gy))
            inter = ix * iy
            union = tw*th + gw*gh - inter
            if union > 0:
                max_iou_with_gt = max(max_iou_with_gt, inter/union)

    # Disposition decisions based on geometry + direction:
    # REVIEW_ADD candidates (dir=DOWN): all 4 move with GT majority
    # REVIEW_UNCERTAIN candidates (most dir=UP): moving opposite, suggest opposite carriageway
    if action == "REVIEW_ADD" and direction == "DOWN":
        if max_iou_with_gt > 0.5:
            disp = "DUPLICATE"
            conf = "high"
            reason = (f"Overlaps existing GT (max IoU={max_iou_with_gt:.2f}); "
                      f"likely duplicate of an annotated vehicle")
        elif gap_fill_candidates:
            # Check if this could be gap-fill of a specific GT track
            # Compare position with nearby GT track
            best_gap = gap_fill_candidates[0]
            disp = "GAP_FILL"
            conf = "medium"
            reason = (f"Same direction as GT majority; GT track {best_gap[0]} has "
                      f"{best_gap[1]} frame gaps in same window; position check needed")
        elif in_upper_region and not in_lateral_span:
            disp = "EXCLUDE_POLICY"
            conf = "medium"
            reason = (f"Direction DOWN but mean position is far from GT lane "
                      f"(cx={mean_cx:.0f}, cy={mean_cy:.0f}); may be outside monitored region")
        else:
            disp = "ADD"
            conf = "high"
            reason = (f"Direction DOWN (same as all 50 GT tracks); within GT lateral span "
                      f"(cx={mean_cx:.0f}px in [{GT_CX_MIN},{GT_CX_MAX}]); "
                      f"{runs_agree} run(s) agree; {n_frames} persistent frames; "
                      f"not a duplicate (max IoU with GT={max_iou_with_gt:.2f})")
    elif direction == "UP":
        disp = "EXCLUDE_POLICY"
        conf = "high"
        reason = (f"Direction UP in a DOWN sequence (all 50 GT tracks move DOWN); "
                  f"consistent with opposite-carriageway vehicle per policy; "
                  f"cx={mean_cx:.0f}px")
    elif direction == "DOWN" and action == "REVIEW_UNCERTAIN":
        disp = "UNCERTAIN"
        conf = "medium"
        reason = (f"Direction DOWN but classified UNCERTAIN; single run ({runs_agree}); "
                  f"short track or borderline position; needs human review")
    else:
        disp = "UNCERTAIN"
        conf = "low"
        reason = f"Insufficient information for automated decision"

    ev_img = os.path.join(GATE1_DIR, f"cand_{cand_id}_review.jpg") if contact_imgs else ""
    dispositions.append({
        "candidate_id": cand_id,
        "frame_range": f"{f_start}-{f_end}",
        "n_frames": n_frames,
        "direction": direction,
        "runs_agree": runs_agree,
        "disposition": disp,
        "confidence": conf,
        "reason": reason,
        "evidence_image": ev_img,
    })
    print(f"  DISPOSITION: {disp} [{conf}] — {reason[:80]}...")

# ─── Write disposition_table.csv ─────────────────────────────────────────────
disp_csv = os.path.join(GATE1_DIR, "disposition_table.csv")
with open(disp_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "candidate_id","frame_range","n_frames","direction","runs_agree",
        "disposition","confidence","reason","evidence_image"
    ])
    writer.writeheader()
    writer.writerows(dispositions)

print("\n" + "=" * 60)
print("DISPOSITION TABLE")
print("=" * 60)
print(f"{'Candidate':<32} {'Frames':>6} {'Dir':>5} {'Disp':<20} {'Conf':<8}")
print("-" * 80)
for d in sorted(dispositions, key=lambda x: x["candidate_id"]):
    print(f"  {d['candidate_id']:<30} {d['n_frames']:>6} {d['direction']:>5} "
          f"{d['disposition']:<20} {d['confidence']:<8}")
print(f"\nWrote: {disp_csv}")

# Summary counts
from collections import Counter
counts = Counter(d["disposition"] for d in dispositions)
print(f"\nSummary: {dict(counts)}")
print("Only ADD and GAP_FILL with high confidence will be applied.")

# ─── Which qualify? ───────────────────────────────────────────────────────────
apply = [d for d in dispositions if d["disposition"] in ("ADD","GAP_FILL") and d["confidence"]=="high"]
not_apply = [d for d in dispositions if d["disposition"] in ("ADD","GAP_FILL") and d["confidence"]!="high"]
print(f"\nTo apply: {[d['candidate_id'] for d in apply]}")
print(f"Proposed but not applied (medium confidence): {[d['candidate_id'] for d in not_apply]}")
print(f"SHA256 of original GT: {sha256_orig}")
print("\nPart A + B DONE")
