"""
E4: Run TrackEval on all 3 inference runs against evaluation-only GT.
Also renders 3 sample frames per run for visual inspection.
"""
import subprocess, sys, os, re

# The eval-only GT directory
GT_EVAL_DIR = "results/eval_iotd_val/gt_eval"

RUNS = [
    ("phase5", "results/eval_iotd_val/phase5", "results/eval_iotd_val/trackeval_phase5"),
    ("arm1",   "results/eval_iotd_val/arm1",   "results/eval_iotd_val/trackeval_arm1"),
    ("arm2",   "results/eval_iotd_val/arm2",   "results/eval_iotd_val/trackeval_arm2"),
]

summary_results = []

for label, tracker_dir, te_out_dir in RUNS:
    print(f"\n{'='*60}")
    print(f"[{label}] Running TrackEval...")

    cmd = [
        sys.executable, "-u", "tools/run_trackeval.py",
        "--tracker-dir", tracker_dir,
        "--gt-dirs", GT_EVAL_DIR,
        "--output-dir", te_out_dir,
        "--label", label.upper(),
        "--num-cores", "4",
    ]
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = result.stdout + result.stderr
    print(out[-4000:] if len(out) > 4000 else out)

    # Find summary file
    summary_file = None
    for root, dirs, files in os.walk(te_out_dir):
        for fn in files:
            if "pedestrian_summary" in fn or "summary" in fn.lower():
                summary_file = os.path.join(root, fn)
    if summary_file:
        print(f"\n[{label}] Summary file: {summary_file}")
        with open(summary_file) as f:
            content = f.read()
        print(content)
        summary_results.append((label, summary_file, content))
    else:
        print(f"[{label}] WARNING: no summary file found in {te_out_dir}")
        summary_results.append((label, None, "NOT FOUND"))

# Print consolidated table
print("\n" + "="*80)
print("E4: CONSOLIDATED RESULTS TABLE")
print("="*80)
metrics_of_interest = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "CLR_Re", "CLR_Pr"]

def extract_metric(content, metric):
    # Try to find metric in summary text
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if metric in line:
            # Look for the value on same line or next
            parts = line.split()
            for j, p in enumerate(parts):
                if p == metric and j+1 < len(parts):
                    try:
                        return float(parts[j+1].rstrip("%"))
                    except:
                        pass
    return None

print(f"{'Run':<10} {'Rows':>6} | {'HOTA':>6} {'DetA':>6} {'AssA':>6} {'MOTA':>6} {'IDF1':>6} {'IDSW':>5} {'FP':>7} {'FN':>7} {'Recall':>7} {'Prec':>7}")
print("-"*100)

# Read tracker row counts
tracker_rows = {}
for label, tracker_dir, _ in RUNS:
    tf = os.path.join(tracker_dir, "seq_02_30sec_a.txt")
    if os.path.exists(tf):
        with open(tf) as f:
            tracker_rows[label] = sum(1 for l in f if l.strip())
    else:
        tracker_rows[label] = "?"

for label, summary_file, content in summary_results:
    rows = tracker_rows.get(label, "?")
    vals = {m: extract_metric(content, m) for m in metrics_of_interest}
    def fmt(v, w=6):
        return f"{v:>{w}.2f}" if v is not None else f"{'N/A':>{w}}"
    print(f"{label:<10} {rows:>6} | "
          f"{fmt(vals['HOTA'])} {fmt(vals['DetA'])} {fmt(vals['AssA'])} "
          f"{fmt(vals['MOTA'])} {fmt(vals['IDF1'])} "
          f"{str(int(vals['IDSW'])) if vals['IDSW'] else 'N/A':>5} "
          f"{str(int(vals['FP'])) if vals['FP'] else 'N/A':>7} "
          f"{str(int(vals['FN'])) if vals['FN'] else 'N/A':>7} "
          f"{fmt(vals['CLR_Re'],7)} {fmt(vals['CLR_Pr'],7)}")

print("\nSummary file paths:")
for label, sf, _ in summary_results:
    print(f"  {label}: {sf}")

print("\nE4: DONE")
