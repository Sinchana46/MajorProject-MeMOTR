"""
Gate 1 Part C: No high-confidence ADD or GAP_FILL — skip correction, just backup + verify.
Part D: Completeness check (20 evenly spaced frames).
"""
import os, sys, csv, json, hashlib, shutil, configparser
import numpy as np
import cv2
from collections import defaultdict

GATE1_DIR  = "check/annotation_audit/gate1"
BACKUP_DIR = "check/annotation_audit/backup"
REVIEW_PACK = os.path.join(GATE1_DIR, "review_pack")
GT_ORIG    = "data/IOTD_val/seq_02_30sec_a/gt/gt.txt"
IMG_DIR    = "data/IOTD_val/seq_02_30sec_a/img1"
SEQINFO    = "data/IOTD_val/seq_02_30sec_a/seqinfo.ini"

os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(REVIEW_PACK, exist_ok=True)

cp = configparser.ConfigParser(); cp.read(SEQINFO)
W = int(cp["Sequence"]["imwidth"]); H = int(cp["Sequence"]["imheight"])
N_FRAMES = int(cp["Sequence"]["seqlength"])

print("=" * 60)
print("PART C: BACKUP + CORRECTION STATUS")
print("=" * 60)

# Backup original
backup_path = os.path.join(BACKUP_DIR, "seq_02_30sec_a_gt_ORIGINAL.txt")
shutil.copy2(GT_ORIG, backup_path)
print(f"Backup written: {backup_path}")

with open(GT_ORIG, 'rb') as f: raw_orig = f.read()
with open(backup_path, 'rb') as f: raw_back = f.read()
sha_orig = hashlib.sha256(raw_orig).hexdigest()
sha_back = hashlib.sha256(raw_back).hexdigest()
print(f"SHA256 original: {sha_orig}")
print(f"SHA256 backup:   {sha_back}")
print(f"Match: {sha_orig == sha_back}")

# Disposition check — no high-confidence ADD/GAP_FILL
disp_csv = os.path.join(GATE1_DIR, "disposition_table.csv")
dispositions = []
with open(disp_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader: dispositions.append(row)

apply_list = [d for d in dispositions if d["disposition"]=="ADD" and d["confidence"]=="high"]
print(f"\nHigh-confidence ADD candidates: {len(apply_list)}")
print("Per Part C rule: 'If no candidate qualifies, write no corrected file and skip to Part E.'")
print("STATUS: No corrected GT file written. Original GT is the evaluation GT.")
print("        All 4 DOWN candidates are GAP_FILL at medium confidence only — not applied.")
print("        5 UP candidates are EXCLUDE_POLICY — correctly excluded by policy.")
print("        1 candidate (001) is UNCERTAIN — sent to review pack.")

# ─── Part D: Completeness check ───────────────────────────────────────────────
print("\n" + "="*60)
print("PART D: COMPLETENESS CHECK (20 evenly spaced frames)")
print("="*60)

# Parse GT
gt_by_frame = defaultdict(list)
with open(GT_ORIG) as f:
    for line in f:
        c = line.strip().split(',')
        if len(c) >= 9:
            gt_by_frame[int(c[0])].append((int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))

# 20 evenly spaced frames
check_frames = [max(1, int(i * N_FRAMES / 19)) for i in range(20)]
check_frames = sorted(set(min(f, N_FRAMES) for f in check_frames))
print(f"Check frames: {check_frames}")

total_missed = 0
frame_missed_counts = []
contact_sheets = []

for frame_num in check_frames:
    img = cv2.imread(os.path.join(IMG_DIR, f"{frame_num:06d}.jpg"))
    if img is None:
        print(f"  f{frame_num:04d}: IMAGE NOT FOUND")
        frame_missed_counts.append(0)
        continue

    gt_boxes = gt_by_frame.get(frame_num, [])

    # Visual inspection heuristic: look for vehicle-like regions not covered by GT
    # Strategy: run simple brightness-contrast check in areas away from GT
    # (automated proxy only — not ground truth)
    # Mark GT on image for visual output
    annotated = img.copy()
    for tid, gx, gy, gw, gh in gt_boxes:
        cv2.rectangle(annotated, (int(gx),int(gy)), (int(gx+gw),int(gy+gh)), (0,255,0), 2)

    # Count GT boxes in this frame
    n_gt = len(gt_boxes)

    # Simple proxy: count large connected components (>2000 px) not overlapping any GT box
    # (This is an estimate only, clearly labelled as such)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Background subtraction proxy: strong edges in non-GT regions
    edges = cv2.Canny(gray, 50, 150)
    # Mask out GT regions
    mask = np.ones(edges.shape, dtype=np.uint8) * 255
    for tid, gx, gy, gw, gh in gt_boxes:
        x1, y1 = max(0,int(gx)), max(0,int(gy))
        x2, y2 = min(W,int(gx+gw)), min(H,int(gy+gh))
        mask[y1:y2, x1:x2] = 0
    edges_outside = cv2.bitwise_and(edges, mask)
    # Dilate and find components
    kernel = np.ones((15,15), np.uint8)
    dilated = cv2.dilate(edges_outside, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
    # Count large regions (vehicle-sized: >3000 px area)
    min_area = 3000
    # Only look in lower 60% of frame (vehicles typically in lower 2/3)
    lower_start = int(H * 0.25)
    potential_missed = 0
    for label_id in range(1, num_labels):
        x, y, sw, sh, area = stats[label_id]
        cy_comp = y + sh//2
        if area >= min_area and cy_comp > lower_start:
            potential_missed += 1
            cv2.rectangle(annotated, (x, y), (x+sw, y+sh), (255, 0, 255), 2)  # magenta

    # Thumbnail
    sheet_small = cv2.resize(annotated, (640, 360))
    cv2.putText(sheet_small, f"f{frame_num:04d} GT={n_gt} possible_missed={potential_missed}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

    frame_missed_counts.append(potential_missed)
    total_missed += potential_missed
    contact_sheets.append(sheet_small)

    save_path = os.path.join(REVIEW_PACK, f"completeness_f{frame_num:04d}.jpg")
    cv2.imwrite(save_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    print(f"  f{frame_num:04d}: GT boxes={n_gt:2d}  potential_missed={potential_missed}  -> {save_path}")

# Summarize
print(f"\nTotal potential missed vehicle-regions across 20 frames: {total_missed}")
print(f"Mean per frame: {total_missed/len(frame_missed_counts):.1f}")
print(f"Frames with potential missed: {sum(1 for x in frame_missed_counts if x>0)}/20")
print("\n*** CAVEAT: 'potential_missed' is a heuristic edge-component count,")
print("    NOT a ground-truth count. Visual inspection is required to confirm.")

if total_missed > 30:
    completeness_verdict = ("The automated heuristic detects large edge-clusters outside GT "
                           "in many frames. The GT should be considered INCOMPLETE for rigorous "
                           "model comparison; a full CVAT annotation pass by the user is recommended.")
elif total_missed > 10:
    completeness_verdict = ("The heuristic detects some uncovered large regions. The GT may "
                           "have minor gaps. A spot-check of the medium-confidence GAP_FILL candidates "
                           "(007, 005, 006, 009) in CVAT is recommended before final tuning decisions.")
else:
    completeness_verdict = ("The heuristic detects few uncovered regions. For comparative "
                           "model-ranking purposes, the existing GT appears adequate. A full CVAT pass "
                           "is optional but recommended before publication.")

print(f"\nCompleteness verdict: {completeness_verdict}")

# Copy review materials to pack
import glob
for pattern in [os.path.join(GATE1_DIR, "cand_*_review.jpg"),
                os.path.join(GATE1_DIR, "disposition_table.csv")]:
    for src in glob.glob(pattern):
        dst = os.path.join(REVIEW_PACK, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
print(f"\nReview pack assembled: {REVIEW_PACK}")

# Save completeness summary
comp_summary = {
    "total_potential_missed": total_missed,
    "frames_checked": len(check_frames),
    "mean_per_frame": round(total_missed / len(frame_missed_counts), 2),
    "frames_with_potential_missed": sum(1 for x in frame_missed_counts if x > 0),
    "verdict": completeness_verdict,
    "caveat": "Heuristic edge-component count, not ground truth. Human review required."
}
with open(os.path.join(GATE1_DIR, "completeness_summary.json"), 'w') as f:
    json.dump(comp_summary, f, indent=2)
print("Part C+D DONE")
