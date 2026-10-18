import sys, os, hashlib, shutil

sys.path.insert(0, "MeMOTR")

# ── 1. Read TrackEval / wrapper code for class handling ──────────────────
print("=== E2.1: HOW CLASSES 2-5 ARE TREATED IN TRACKEVAL ===")

# Check run_trackeval.py for class handling
with open("tools/run_trackeval.py") as f:
    tv_lines = f.readlines()
for i, line in enumerate(tv_lines):
    if any(x in line.lower() for x in ["class", "distract", "ignore", "col8", "column 8", "cls_id", "class_id"]):
        print(f"  run_trackeval.py L{i+1}: {line.rstrip()}")

# Check TrackEval MOTChallenge loader
te_paths = []
for root, dirs, files in os.walk("."):
    for fn in files:
        if fn.endswith(".py") and "trackeval" in root.lower():
            if "mot_challenge" in fn.lower() or "datasets" in fn.lower():
                te_paths.append(os.path.join(root, fn))

for tp in te_paths[:3]:
    print(f"\n  TrackEval file: {tp}")
    with open(tp) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if any(x in line.lower() for x in ["class", "distract", "ignore"]) and "col" in line.lower() or \
           any(x in line for x in ["class_id", "cls_id", "column 8", "distractor", "IGNORE"]):
            print(f"    L{i+1}: {line.rstrip()}")

# ── 2. Build evaluation-only GT ──────────────────────────────────────────
print("\n=== E2.2: BUILDING EVALUATION-ONLY GT ===")
orig_gt = "data/IOTD/seq_02_30sec_a/gt/gt.txt"
orig_seqinfo = "data/IOTD/seq_02_30sec_a/seqinfo.ini"
out_dir  = "results/eval_iotd_val/gt_eval/seq_02_30sec_a/gt"
os.makedirs(out_dir, exist_ok=True)

# SHA256 of original
def sha256f(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

orig_sha = sha256f(orig_gt)
print(f"Original gt.txt SHA256: {orig_sha}")

# Read and remap classes to 1
rows = []
distinct_ids = set()
distinct_classes = set()
with open(orig_gt) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        cols = line.split(",")
        if len(cols) >= 8:
            distinct_classes.add(cols[7].strip())
            distinct_ids.add(cols[1].strip())
            cols[7] = "1"  # remap class to 1
        rows.append(",".join(cols))

print(f"Rows: {len(rows)} (expected 4,182)")
print(f"Distinct track IDs: {len(distinct_ids)} (expected 50)")
print(f"Original class IDs in col 8: {sorted(distinct_classes)}")

eval_gt_path = os.path.join(out_dir, "gt.txt")
with open(eval_gt_path, "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"Written: {eval_gt_path} ({len(rows)} rows)")

# Copy seqinfo.ini (unmodified)
eval_seqinfo = os.path.join(os.path.dirname(out_dir), "seqinfo.ini")
shutil.copy2(orig_seqinfo, eval_seqinfo)
print(f"Copied seqinfo.ini to: {eval_seqinfo}")

# Confirm original unchanged
orig_sha_after = sha256f(orig_gt)
print(f"\nOriginal gt.txt SHA256 after: {orig_sha_after}")
print(f"Original unchanged: {orig_sha == orig_sha_after}")
