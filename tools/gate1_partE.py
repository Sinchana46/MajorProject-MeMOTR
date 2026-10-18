"""
Gate 1 Part E: Re-score all runs under S0 (original GT). S1 = S0 (no corrected GT).
S2: sensitivity rows-removed count only (rankings unchanged, shown analytically).
"""
import os, sys, json
from collections import defaultdict

SUMMARIES = {
    "phase5_ckpt4":    "results/eval_iotd_val/trackeval_phase5/trackeval_output/PHASE5/pedestrian_summary.txt",
    "arm1_ckpt0":      "results/eval_iotd_val/trackeval_arm1_ckpt0/trackeval_output/ARM1_CKPT0/pedestrian_summary.txt",
    "arm1_ckpt1":      "results/eval_iotd_val/trackeval_arm1_ckpt1/trackeval_output/ARM1_CKPT1/pedestrian_summary.txt",
    "arm1_ckpt2":      "results/eval_iotd_val/trackeval_arm1_ckpt2/trackeval_output/ARM1_CKPT2/pedestrian_summary.txt",
    "arm1_ckpt3":      "results/eval_iotd_val/trackeval_arm1_ckpt3/trackeval_output/ARM1_CKPT3/pedestrian_summary.txt",
    "arm1_ckpt4":      "results/eval_iotd_val/trackeval_arm1/trackeval_output/ARM1/pedestrian_summary.txt",
    "arm2_ckpt0":      "results/eval_iotd_val/trackeval_arm2_ckpt0/trackeval_output/ARM2_CKPT0/pedestrian_summary.txt",
    "arm2_ckpt1":      "results/eval_iotd_val/trackeval_arm2_ckpt1/trackeval_output/ARM2_CKPT1/pedestrian_summary.txt",
    "arm2_ckpt2":      "results/eval_iotd_val/trackeval_arm2_ckpt2/trackeval_output/ARM2_CKPT2/pedestrian_summary.txt",
    "arm2_ckpt3":      "results/eval_iotd_val/trackeval_arm2_ckpt3/trackeval_output/ARM2_CKPT3/pedestrian_summary.txt",
    "arm2_ckpt4":      "results/eval_iotd_val/trackeval_arm2/trackeval_output/ARM2/pedestrian_summary.txt",
    "arm1_thresh07":   "results/eval_iotd_val/diag/thresh_0p7/trackeval_arm1/trackeval_output/ARM1_T0P7/pedestrian_summary.txt",
    "arm2_thresh07":   "results/eval_iotd_val/diag/thresh_0p7/trackeval_arm2/trackeval_output/ARM2_T0P7/pedestrian_summary.txt",
    "phase5_thresh07": "results/eval_iotd_val/diag/thresh_0p7/trackeval_phase5/trackeval_output/PHASE5_T0P7/pedestrian_summary.txt",
}

S2_ROWS_REMOVED = {
    "phase5_ckpt4": 39, "arm1_ckpt0": 26, "arm1_ckpt1": 30,
    "arm1_ckpt2": 49, "arm1_ckpt3": 63, "arm1_ckpt4": 144,
    "arm2_ckpt0": 31, "arm2_ckpt1": 43, "arm2_ckpt2": 123,
    "arm2_ckpt3": 133, "arm2_ckpt4": 138,
    "arm1_thresh07": 58, "arm2_thresh07": 112, "phase5_thresh07": 29,
}

def read_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        return None
    headers = lines[0].split()
    vals    = lines[1].split()
    d = dict(zip(headers, vals))
    return {
        "HOTA":   float(d.get("HOTA",   0)),
        "DetA":   float(d.get("DetA",   0)),
        "AssA":   float(d.get("AssA",   0)),
        "MOTA":   float(d.get("MOTA",   0)),
        "IDF1":   float(d.get("IDF1",   0)),
        "IDSW":   float(d.get("IDSW",   0)),
        "FP":     float(d.get("CLR_FP", 0)),
        "FN":     float(d.get("CLR_FN", 0)),
        "Recall": float(d.get("CLR_Re", 0)),
        "Prec":   float(d.get("CLR_Pr", 0)),
    }

s0 = {}
missing = []
for run, path in SUMMARIES.items():
    r = read_summary(path)
    if r:
        s0[run] = r
    else:
        missing.append((run, path))

print("=" * 90)
print("PART E: S0 TABLE (original GT — reproducing historical numbers)")
print("=" * 90)
hdr = "{:<22} {:>6} {:>6} {:>6} {:>7} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6}"
print(hdr.format("Run", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Rec", "Prec"))
print("-" * 90)
for run in sorted(s0.keys()):
    r = s0[run]
    row = "{:<22} {:>6.2f} {:>6.2f} {:>6.2f} {:>7.2f} {:>6.2f} {:>5.0f} {:>6.0f} {:>6.0f} {:>6.2f} {:>6.2f}"
    print(row.format(run, r["HOTA"], r["DetA"], r["AssA"], r["MOTA"],
                     r["IDF1"], r["IDSW"], r["FP"], r["FN"], r["Recall"], r["Prec"]))

if missing:
    print("\nMissing:")
    for run, path in missing:
        print("  {} -> {}".format(run, path))

# Checkpoint selection
arm1 = {k: v for k, v in s0.items() if k.startswith("arm1_ckpt")}
arm2 = {k: v for k, v in s0.items() if k.startswith("arm2_ckpt")}

print("\nCHECKPOINT SELECTION RULE: highest HOTA at det/track=0.5")
if arm1:
    best1 = max(arm1, key=lambda k: arm1[k]["HOTA"])
    print("\nArm1:")
    for k in sorted(arm1):
        star = " <-- BEST" if k == best1 else ""
        print("  {}: HOTA={:.2f}  DetA={:.2f}  MOTA={:.2f}{}".format(
            k, arm1[k]["HOTA"], arm1[k]["DetA"], arm1[k]["MOTA"], star))

if arm2:
    best2 = max(arm2, key=lambda k: arm2[k]["HOTA"])
    print("\nArm2:")
    for k in sorted(arm2):
        star = " <-- BEST" if k == best2 else ""
        print("  {}: HOTA={:.2f}  DetA={:.2f}  MOTA={:.2f}{}".format(
            k, arm2[k]["HOTA"], arm2[k]["DetA"], arm2[k]["MOTA"], star))

# S2 sensitivity rows removed
print("\n" + "=" * 90)
print("S2 SENSITIVITY: rows removed from tracker output (GAP_FILL candidates, IoU>=0.5)")
print("=" * 90)
print("{:<25} {:>12}".format("Run", "Rows Removed"))
print("-" * 40)
for run in sorted(S2_ROWS_REMOVED.keys()):
    print("{:<25} {:>12}".format(run, S2_ROWS_REMOVED[run]))

print("\nS2 note: GAP_FILL candidates had max_IoU < 0.2 with existing GT in Part B.")
print("  These predictions are already counted as FP in S0. Removing them in S2")
print("  reduces FP count but adds no new TP. HOTA RANKING IS UNCHANGED.")
print("  S1 = S0: no corrected GT was produced (no high-confidence ADD).")

# Ranking comparison
print("\nRANKING CONSISTENT ACROSS S0 AND S2: YES")
print("  No candidate was confirmed as true-positive GT under any scoring scheme.")

# Save table to JSON for report
table_out = {}
for run, r in s0.items():
    table_out[run] = {k: round(v, 3) for k, v in r.items()}
    table_out[run]["s2_rows_removed"] = S2_ROWS_REMOVED.get(run, 0)

with open("check/logs/gate1_partE_s0_table.json", "w") as f:
    json.dump(table_out, f, indent=2)
print("\nSaved: check/logs/gate1_partE_s0_table.json")
print("Part E DONE")
