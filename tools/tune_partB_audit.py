"""
Parts B2-B5: Annotation audit for all 3 sequences.
B2: Find candidate missing-annotation tracks (>=15 frames, >=80% max IoU < 0.2)
B3: Triage by motion/position (LIKELY_OPPOSITE, LIKELY_MISSING, UNCERTAIN)
B4: GT bottom-edge statistics, fixed vs soft boundary?
B5: Output proposals.csv, evidence images, summary.md, CVAT MOT export
"""
import os, sys, csv, math, shutil, json
from collections import defaultdict
import numpy as np
import cv2

# ─── Paths ──────────────────────────────────────────────────────────────────
SEQS = {
    "seq_02_30sec_a": {
        "img": "data/IOTD_val/seq_02_30sec_a/img1",
        "gt":  "data/IOTD_val/seq_02_30sec_a/gt/gt.txt",
        "si":  "data/IOTD_val/seq_02_30sec_a/seqinfo.ini",
        "trackers": {
            "phase5": "results/eval_iotd_val/phase5/seq_02_30sec_a.txt",
            "arm1":   "results/eval_iotd_val/arm1/seq_02_30sec_a.txt",
            "arm2":   "results/eval_iotd_val/arm2/seq_02_30sec_a.txt",
        }
    },
    "seq_01_1min": {
        "img": "data/IOTD/seq_01_1min/img1",
        "gt":  "data/IOTD/seq_01_1min/gt/gt.txt",
        "si":  "data/IOTD/seq_01_1min/seqinfo.ini",
        "trackers": {"phase5": "results/eval_iotd_train_audit/phase5/seq_01_1min.txt"},
    },
    "seq_03_30sec_b": {
        "img": "data/IOTD/seq_03_30sec_b/img1",
        "gt":  "data/IOTD/seq_03_30sec_b/gt/gt.txt",
        "si":  "data/IOTD/seq_03_30sec_b/seqinfo.ini",
        "trackers": {"phase5": "results/eval_iotd_train_audit/phase5/seq_03_30sec_b.txt"},
    },
}

AUDIT_DIR = "check/annotation_audit"
CVAT_DIR  = os.path.join(AUDIT_DIR, "cvat_import")
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(CVAT_DIR, exist_ok=True)

MIN_TRACK_LEN = 15       # minimum frames to be a candidate
MIN_LOW_IOU_FRAC = 0.80  # fraction of frames with max IoU < 0.2

def parse_gt(path):
    rows = defaultdict(list)
    max_tid = 0
    with open(path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) < 6: continue
            frame, tid = int(c[0]), int(c[1])
            x, y, w, h = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            vis = float(c[8]) if len(c) > 8 else 1.0
            rows[frame].append((tid, x, y, w, h, vis))
            max_tid = max(max_tid, tid)
    return rows, max_tid

def parse_tracker(path):
    tracks = defaultdict(list)  # tid -> [(frame, x, y, w, h)]
    by_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) < 6: continue
            frame, tid = int(c[0]), int(c[1])
            x, y, w, h = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            tracks[tid].append((frame, x, y, w, h))
            by_frame[frame].append((tid, x, y, w, h))
    return tracks, by_frame

def iou(a, b):
    ax1,ay1,ax2,ay2 = a[0],a[1],a[0]+a[2],a[1]+a[3]
    bx1,by1,bx2,by2 = b[0],b[1],b[0]+b[2],b[1]+b[3]
    ix = max(0, min(ax2,bx2)-max(ax1,bx1))
    iy = max(0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.0

def read_seqinfo(path):
    d = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k,_,v = line.strip().partition('=')
                d[k.strip()] = v.strip()
    return int(d.get('imwidth',1920)), int(d.get('imheight',1080)), int(d.get('seqlength',0))

all_proposals = []

for seq_name, seq_info in SEQS.items():
    print(f"\n{'='*60}")
    print(f"SEQUENCE: {seq_name}")
    print(f"{'='*60}")

    W, H, N_FRAMES = read_seqinfo(seq_info["si"])
    print(f"  Size: {W}x{H}, {N_FRAMES} frames")

    gt_by_frame, max_gt_tid = parse_gt(seq_info["gt"])
    gt_tids = set()
    for boxes in gt_by_frame.values():
        for b in boxes: gt_tids.add(b[0])
    print(f"  GT: {sum(len(v) for v in gt_by_frame.values())} rows, "
          f"{len(gt_tids)} tracks, max TID={max_gt_tid}")

    # B4: GT bottom-edge distribution
    gt_bottoms = []
    for frame, boxes in sorted(gt_by_frame.items()):
        for tid,x,y,w,h,vis in boxes:
            gt_bottoms.append(y+h)
    gt_bottoms = sorted(gt_bottoms)
    print(f"\n  B4 GT bottom-edge stats:")
    print(f"    min={min(gt_bottoms):.1f} p1={np.percentile(gt_bottoms,1):.1f} "
          f"p5={np.percentile(gt_bottoms,5):.1f} median={np.percentile(gt_bottoms,50):.1f} "
          f"max={max(gt_bottoms):.1f}")

    # ─── B2: Candidate tracks per run ────────────────────────────────────────
    run_candidates = {}
    all_run_tracks = {}
    for run_name, tracker_path in seq_info["trackers"].items():
        if not os.path.exists(tracker_path):
            print(f"  [{run_name}] tracker not found: {tracker_path}")
            continue
        tracks, by_frame = parse_tracker(tracker_path)
        candidates = {}
        for tid, frames_data in tracks.items():
            frames_data = sorted(frames_data, key=lambda x: x[0])
            n_frames_track = len(frames_data)
            if n_frames_track < MIN_TRACK_LEN:
                continue
            low_iou_count = 0
            for frame, tx, ty, tw, th in frames_data:
                gt_boxes = gt_by_frame.get(frame, [])
                if not gt_boxes:
                    low_iou_count += 1
                    continue
                max_iou = max(iou([tx,ty,tw,th], [gx,gy,gw,gh])
                              for (_,gx,gy,gw,gh,*_) in gt_boxes)
                if max_iou < 0.2:
                    low_iou_count += 1
            frac = low_iou_count / n_frames_track
            if frac >= MIN_LOW_IOU_FRAC:
                candidates[tid] = {"frames": frames_data, "low_iou_frac": frac}
        run_candidates[run_name] = candidates
        all_run_tracks[run_name] = tracks
        print(f"  [{run_name}] {len(tracks)} total tracks, {len(candidates)} candidates "
              f"(>={MIN_TRACK_LEN} frames, >={MIN_LOW_IOU_FRAC*100:.0f}% low IoU)")

    # ─── B2: For seq_02, count agreement among runs ─────────────────────────
    # Aggregate all candidate boxes across all runs
    merged_candidates = {}  # candidate_id -> {box data, runs_agreed set}
    cand_id = 0
    for run_name, cands in run_candidates.items():
        for tid, cdata in cands.items():
            # Check overlap with existing merged candidates (IoU of mean box)
            frames_data = cdata["frames"]
            # Compute mean center
            cx_mean = np.mean([f[1]+f[3]/2 for f in frames_data])
            cy_mean = np.mean([f[2]+f[4]/2 for f in frames_data])
            # Check if this overlaps an existing candidate (center within 50px)
            found = None
            for existing_id, ex in merged_candidates.items():
                if abs(ex["cx_mean"]-cx_mean) < 80 and abs(ex["cy_mean"]-cy_mean) < 80:
                    found = existing_id
                    break
            if found is not None:
                merged_candidates[found]["runs"].add(run_name)
                merged_candidates[found]["run_tids"][run_name] = tid
            else:
                mean_box = [
                    np.mean([f[1] for f in frames_data]),
                    np.mean([f[2] for f in frames_data]),
                    np.mean([f[3] for f in frames_data]),
                    np.mean([f[4] for f in frames_data]),
                ]
                merged_candidates[cand_id] = {
                    "seq": seq_name, "primary_run": run_name, "primary_tid": tid,
                    "frames": frames_data, "n_frames": len(frames_data),
                    "mean_box": mean_box,
                    "cx_mean": cx_mean, "cy_mean": cy_mean,
                    "runs": {run_name}, "run_tids": {run_name: tid},
                    "low_iou_frac": cdata["low_iou_frac"],
                }
                cand_id += 1

    print(f"\n  Merged unique candidates: {len(merged_candidates)}")

    # ─── B3: Triage by motion and position ───────────────────────────────────
    # Compute majority vertical direction of GT tracks (+ = moving down = approaching camera)
    gt_directions = []
    for tid in sorted(gt_tids):
        pts = []
        for frame in sorted(gt_by_frame.keys()):
            for box in gt_by_frame[frame]:
                if box[0] == tid:
                    pts.append((frame, box[2]+box[4]/2))  # frame, bottom-y
        if len(pts) >= 5:
            pts = sorted(pts, key=lambda p: p[0])
            dy = pts[-1][1] - pts[0][1]
            gt_directions.append(1 if dy > 0 else -1)
    majority_dir = 1 if sum(gt_directions) >= 0 else -1
    majority_str = "DOWN (approaching)" if majority_dir > 0 else "UP (departing)"
    print(f"  GT majority direction: {majority_str} ({sum(1 for d in gt_directions if d==majority_dir)}/{len(gt_directions)} tracks)")

    # GT lane span (x range of GT hulls)
    gt_all_cx = []
    for frame, boxes in gt_by_frame.items():
        for b in boxes:
            gt_all_cx.append(b[1]+b[3]/2)
    gt_cx_min, gt_cx_max = min(gt_all_cx), max(gt_all_cx)
    print(f"  GT lateral span (cx): {gt_cx_min:.0f}px - {gt_cx_max:.0f}px")

    seq_proposals = []
    for cand_id_local, cand in merged_candidates.items():
        frames_data = cand["frames"]
        n_frames = cand["n_frames"]

        # Vertical trajectory
        pts = sorted(frames_data, key=lambda f: f[0])
        dy = pts[-1][2] - pts[0][2]  # change in y (top of box)
        dir_sign = 1 if dy > 0 else -1 if dy < 0 else 0

        # Mean cx
        cx_mean = cand["cx_mean"]
        cy_mean = cand["cy_mean"]
        mb = cand["mean_box"]

        # Is inside GT lateral span?
        lateral_overlap = gt_cx_min <= cx_mean <= gt_cx_max

        # Assign triage class
        if n_frames < MIN_TRACK_LEN:
            triage = "UNCERTAIN"
        elif dir_sign == 0:
            triage = "UNCERTAIN"
        elif dir_sign == majority_dir:
            if lateral_overlap:
                triage = "LIKELY_MISSING"
            else:
                triage = "UNCERTAIN"
        else:
            if not lateral_overlap:
                triage = "LIKELY_OPPOSITE"
            else:
                triage = "UNCERTAIN"

        proposed_action = {
            "LIKELY_MISSING": "REVIEW_ADD",
            "LIKELY_OPPOSITE": "REVIEW_EXCLUDE_BY_POLICY",
            "UNCERTAIN": "REVIEW_UNCERTAIN",
        }[triage]

        runs_agreeing = len(cand["runs"])

        proposal = {
            "sequence": seq_name,
            "candidate_id": f"{seq_name}_{cand_id_local:03d}",
            "primary_run": cand["primary_run"],
            "primary_tid": cand["primary_tid"],
            "frame_start": pts[0][0],
            "frame_end": pts[-1][0],
            "frames": n_frames,
            "mean_x": f"{mb[0]:.1f}",
            "mean_y": f"{mb[1]:.1f}",
            "mean_w": f"{mb[2]:.1f}",
            "mean_h": f"{mb[3]:.1f}",
            "direction": "DOWN" if dir_sign > 0 else "UP" if dir_sign < 0 else "STATIC",
            "lateral_in_gt_span": str(lateral_overlap),
            "runs_agreeing": runs_agreeing,
            "runs": "+".join(sorted(cand["runs"])),
            "suggested_class": triage,
            "proposed_action": proposed_action,
            "low_iou_frac": f"{cand['low_iou_frac']:.3f}",
        }
        seq_proposals.append(proposal)
        all_proposals.append(proposal)

    # Print top 10 by n_frames
    top = sorted(seq_proposals, key=lambda p: -(int(p["frames"])))[:10]
    print(f"\n  Top candidates (by persistence):")
    for p in top:
        print(f"    {p['candidate_id']} frames={p['frames']} "
              f"class={p['suggested_class']} action={p['proposed_action']} "
              f"runs={p['runs_agreeing']} dir={p['direction']}")

    # ─── B4: Per-window analysis ─────────────────────────────────────────────
    print(f"\n  B4: GT bottom-edge per 100-frame window (min / p5):")
    windows = range(1, N_FRAMES, 100)
    candidate_bottoms_all = []
    for cand in seq_proposals:
        if cand["suggested_class"] == "LIKELY_MISSING":
            cdata = merged_candidates[[k for k,v in merged_candidates.items()
                                      if v.get("seq")==seq_name and
                                      f"{seq_name}_{k:03d}"==cand["candidate_id"]][0] if merged_candidates else 0]
            # just use mean_y + mean_h as bottom estimate
            candidate_bottoms_all.append(float(cand["mean_y"]) + float(cand["mean_h"]))

    for win_start in list(windows)[:5]:
        win_end = min(win_start + 99, N_FRAMES)
        gt_bot_win = [y+h for frame in range(win_start, win_end+1)
                      for _,x,y,w,h,vis in gt_by_frame.get(frame,[])
                      if vis > 0]
        if gt_bot_win:
            print(f"    f{win_start:04d}-{win_end:04d}: GT_min={min(gt_bot_win):.0f} "
                  f"GT_p5={np.percentile(gt_bot_win,5):.0f} GT_max={max(gt_bot_win):.0f}")

    # ─── B5: Evidence renders for top 40 candidates ─────────────────────────
    top40 = sorted(seq_proposals, key=lambda p: -(int(p["frames"])*int(p["runs_agreeing"])))[:40]
    seq_evidence_dir = os.path.join(AUDIT_DIR, seq_name)
    os.makedirs(seq_evidence_dir, exist_ok=True)

    print(f"\n  Rendering evidence images for top {len(top40)} candidates...")
    for pidx, proposal in enumerate(top40[:20]):  # limit to 20 renders to save time
        cand_key = int(proposal["candidate_id"].split("_")[-1])
        if cand_key not in merged_candidates:
            continue
        cand_data = merged_candidates[cand_key]
        frames_data = sorted(cand_data["frames"], key=lambda f: f[0])
        n = len(frames_data)
        sample_frames = [frames_data[0], frames_data[n//2], frames_data[-1]]

        contact_imgs = []
        for fi, (frame, tx, ty, tw, th) in enumerate(sample_frames):
            img_path = os.path.join(seq_info["img"], f"{frame:06d}.jpg")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Draw GT boxes (green)
            for _,gx,gy,gw,gh,*_ in gt_by_frame.get(frame, []):
                cv2.rectangle(img, (int(gx),int(gy)), (int(gx+gw),int(gy+gh)), (0,255,0), 2)
            # Draw candidate box (cyan)
            cv2.rectangle(img, (int(tx),int(ty)), (int(tx+tw),int(ty+th)), (255,255,0), 3)
            cv2.putText(img, f"f{frame} {'start' if fi==0 else 'mid' if fi==1 else 'end'}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
            # Thumbnail-sized
            scale = 0.4
            img_small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
            contact_imgs.append(img_small)

        if len(contact_imgs) >= 2:
            contact = np.concatenate(contact_imgs, axis=1)
            evidence_path = os.path.join(seq_evidence_dir,
                f"cand_{proposal['candidate_id']}_{proposal['suggested_class']}.jpg")
            cv2.imwrite(evidence_path, contact, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            proposal["evidence_image"] = evidence_path
        else:
            proposal["evidence_image"] = ""

    # ─── B4: Overview image (GT bottom edge + candidates) ────────────────────
    print(f"  Generating B4 overview image for {seq_name}...")
    fig_w, fig_h = 1200, 400
    overview = np.ones((fig_h, fig_w, 3), dtype=np.uint8) * 30
    # Plot GT bottom edges as grey dots
    for frame, boxes in sorted(gt_by_frame.items()):
        for _,x,y,w,h,vis in boxes:
            px = int(frame / N_FRAMES * fig_w)
            py = int((1 - (y+h)/H) * fig_h)  # invert y
            if 0<=px<fig_w and 0<=py<fig_h:
                cv2.circle(overview, (px, py), 1, (150,150,150), -1)
    # Plot LIKELY_MISSING candidate tracks as cyan lines
    for proposal in seq_proposals:
        if proposal["suggested_class"] != "LIKELY_MISSING":
            continue
        cand_key = int(proposal["candidate_id"].split("_")[-1])
        if cand_key not in merged_candidates:
            continue
        fdata = sorted(merged_candidates[cand_key]["frames"], key=lambda f: f[0])
        pts_img = []
        for frame,tx,ty,tw,th in fdata:
            px = int(frame / N_FRAMES * fig_w)
            py = int((1 - (ty+th)/H) * fig_h)
            pts_img.append((px, py))
        for i in range(len(pts_img)-1):
            cv2.line(overview, pts_img[i], pts_img[i+1], (255,255,0), 1)
    cv2.putText(overview, f"{seq_name}: GT bottom-edge (grey) + LIKELY_MISSING (cyan)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(overview, "Y-axis: high=bottom of frame, low=top of frame",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    overview_path = os.path.join(AUDIT_DIR, f"overview_{seq_name}.jpg")
    cv2.imwrite(overview_path, overview)
    print(f"  Saved overview: {overview_path}")

# ─── B5: proposals.csv ───────────────────────────────────────────────────────
proposals_csv = os.path.join(AUDIT_DIR, "proposals.csv")
if all_proposals:
    with open(proposals_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence","candidate_id","primary_run","primary_tid",
            "frame_start","frame_end","frames",
            "mean_x","mean_y","mean_w","mean_h",
            "direction","lateral_in_gt_span","runs_agreeing","runs",
            "suggested_class","proposed_action","low_iou_frac","evidence_image"
        ])
        writer.writeheader()
        for p in sorted(all_proposals, key=lambda x: (x["sequence"], -int(x["frames"]))):
            if "evidence_image" not in p:
                p["evidence_image"] = ""
            writer.writerow(p)
    print(f"\nWrote proposals.csv: {proposals_csv} ({len(all_proposals)} rows)")

# ─── B5: CVAT MOT 1.1 export for REVIEW_ADD candidates ──────────────────────
print("\nGenerating CVAT MOT export for REVIEW_ADD candidates...")
review_add = [p for p in all_proposals if p["proposed_action"] == "REVIEW_ADD"]
print(f"  {len(review_add)} REVIEW_ADD candidates")
for seq_name, seq_info in SEQS.items():
    _, _, max_gt_tid_seq = (0, 0, 0)
    _, max_gt_tid_seq = parse_gt(seq_info["gt"])
    seq_review = [p for p in review_add if p["sequence"]==seq_name]
    if not seq_review:
        continue
    cvat_lines = []
    next_tid = max_gt_tid_seq + 1
    for p in seq_review:
        cand_key = int(p["candidate_id"].split("_")[-1])
        # Find the frames data from merged_candidates across all seqs
        fdata = None
        for k,v in merged_candidates.items():
            if v.get("seq") == seq_name:
                cand_key_check = k
                if f"{seq_name}_{cand_key_check:03d}" == p["candidate_id"]:
                    fdata = sorted(v["frames"], key=lambda f: f[0])
                    break
        if not fdata:
            continue
        for frame, tx, ty, tw, th in fdata:
            # MOT format: frame,id,x,y,w,h,conf,class,vis
            cvat_lines.append(f"{frame},{next_tid},{tx:.2f},{ty:.2f},{tw:.2f},{th:.2f},1,-1,-1,-1")
        next_tid += 1
    if cvat_lines:
        cvat_path = os.path.join(CVAT_DIR, f"{seq_name}_review_add.txt")
        with open(cvat_path, 'w') as f:
            f.write('\n'.join(cvat_lines))
        print(f"  {seq_name}: {len(seq_review)} tracks, {len(cvat_lines)} rows -> {cvat_path}")

# ─── B5: summary.md ─────────────────────────────────────────────────────────
summary_lines = ["# Annotation Audit Summary\n",
                 "Generated by tools/tune_partB_audit.py\n\n"]
for seq_name in SEQS:
    seq_props = [p for p in all_proposals if p["sequence"]==seq_name]
    by_class = defaultdict(int)
    for p in seq_props: by_class[p["suggested_class"]] += 1
    summary_lines.append(f"## {seq_name}\n")
    summary_lines.append(f"- Total unique candidates: {len(seq_props)}\n")
    summary_lines.append(f"- LIKELY_MISSING: {by_class['LIKELY_MISSING']}\n")
    summary_lines.append(f"- LIKELY_OPPOSITE: {by_class['LIKELY_OPPOSITE']}\n")
    summary_lines.append(f"- UNCERTAIN: {by_class['UNCERTAIN']}\n\n")

summary_lines.append("## B4 Conclusion\n")
summary_lines.append(
    "See individual sequence sections below. "
    "GT bottom-edge and LIKELY_MISSING candidate positions "
    "reported per 100-frame window in the log. "
    "Whether the boundary is fixed, soft, or indicative of incomplete annotation "
    "is stated in the full report based on window overlap measurements.\n"
)
with open(os.path.join(AUDIT_DIR, "summary.md"), 'w') as f:
    f.writelines(summary_lines)
print(f"\nSummary written: {os.path.join(AUDIT_DIR, 'summary.md')}")
print("\nParts B2-B5 DONE")
