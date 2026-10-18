#!/usr/bin/env python3
"""
Pre-tuning package: scripts/compare_results.py
Compare new TrackEval summaries (or CSV) against the frozen v1 registry.
Usage:
    python compare_results.py --new_dir <folder_of_new_summaries> --out_dir <output_folder>
    python compare_results.py --new_csv <path_to_csv> --out_dir <output_folder>

Protocol:
- Warns if scoring, threshold, GT hash, or sequence does not match registry.
- Prints difference table for HOTA DetA AssA MOTA IDF1 IDSW FP FN.
- Appends nothing to the frozen registry.
- Writes comparison CSV to out_dir.
"""
import os, sys, csv, json, hashlib, argparse, glob, datetime, re
from collections import defaultdict

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results_registry.csv")
REGISTRY_PATH = os.path.normpath(REGISTRY_PATH)

DIFF_METRICS = ["HOTA","DetA","AssA","MOTA","IDF1","IDSW","FP","FN"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def read_trackeval_summary(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2: return None
    d = dict(zip(lines[0].split(), lines[1].split()))
    return {
        "HOTA": float(d.get("HOTA",0)), "DetA": float(d.get("DetA",0)),
        "AssA": float(d.get("AssA",0)), "MOTA": float(d.get("MOTA",0)),
        "IDF1": float(d.get("IDF1",0)), "IDSW": float(d.get("IDSW",0)),
        "FP":   float(d.get("CLR_FP",0)), "FN": float(d.get("CLR_FN",0)),
    }

def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        print(f"ERROR: registry not found at {REGISTRY_PATH}"); sys.exit(1)
    rows = []
    with open(REGISTRY_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in ["HOTA","DetA","AssA","MOTA","IDF1","threshold"]:
                row[k] = float(row[k])
            for k in ["IDSW","FP","FN"]:
                row[k] = int(float(row[k]))
            rows.append(row)
    return rows

def compare(new_metrics_list, registry, out_dir, label="comparison"):
    """
    new_metrics_list: list of dicts with keys: run, checkpoint, scoring, threshold, GT_sha256, + metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_csv = os.path.join(out_dir, f"{label}_{now}.csv")

    # Registry indexed by (run, checkpoint, scoring, threshold)
    reg_idx = defaultdict(list)
    for r in registry:
        key = (r["run"], r["checkpoint"], r["scoring"], float(r["threshold"]))
        reg_idx[key].append(r)

    # Baseline comparisons (Phase5, Arm1 ckpt3, Arm2 ckpt1) from registry at S0 thresh=0.5
    baselines = {}
    for r in registry:
        if r["scoring"]=="S0" and float(r["threshold"])==0.5:
            if (r["run"]=="Phase5" and r["checkpoint"]=="ckpt_4") or \
               (r["run"]=="Arm1"   and r["checkpoint"]=="ckpt_3") or \
               (r["run"]=="Arm2"   and r["checkpoint"]=="ckpt_1"):
                baselines[(r["run"],r["checkpoint"])] = r

    out_rows = []
    warnings = []

    print(f"\n{'Run':<10} {'Ckpt':<8} {'Metric':<8} {'New':>8} {'Registry':>10} {'Diff':>8}")
    print("-"*55)

    for nm in new_metrics_list:
        key = (nm.get("run","?"), nm.get("checkpoint","?"),
               nm.get("scoring","S0"), float(nm.get("threshold",0.5)))
        reg_matches = reg_idx.get(key, [])

        # Protocol checks
        if nm.get("scoring","S0") != "S0":
            warnings.append(f"WARNING: scoring={nm.get('scoring')} != S0 (registry uses S0)")
        if abs(float(nm.get("threshold",0.5)) - 0.5) > 0.001 and float(nm.get("threshold",0.5)) not in [0.7,0.8]:
            warnings.append(f"WARNING: threshold={nm.get('threshold')} not in registry (0.5/0.7/0.8)")
        if nm.get("GT_sha256") and nm["GT_sha256"] != "17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472":
            warnings.append(f"WARNING: GT hash mismatch for {nm.get('run')} {nm.get('checkpoint')} — unlike protocols!")

        reg_row = reg_matches[0] if reg_matches else None

        for metric in DIFF_METRICS:
            new_val = float(nm.get(metric, 0))
            reg_val = float(reg_row[metric]) if reg_row else None
            diff    = (new_val - reg_val) if reg_val is not None else None
            diff_s  = f"{diff:+.2f}" if diff is not None else "N/A"
            reg_s   = f"{reg_val:.2f}" if reg_val is not None else "NOT IN REGISTRY"
            print(f"  {nm.get('run','?'):<8} {nm.get('checkpoint','?'):<8} {metric:<8} {new_val:>8.2f} {reg_s:>10} {diff_s:>8}")
            out_rows.append({
                "run": nm.get("run","?"), "checkpoint": nm.get("checkpoint","?"),
                "scoring": nm.get("scoring","S0"), "threshold": nm.get("threshold",0.5),
                "metric": metric, "new_value": new_val,
                "registry_value": reg_val, "diff": diff,
                "registry_version": reg_row["registry_version"] if reg_row else "N/A",
            })

    if warnings:
        print("\n" + "\n".join(warnings))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run","checkpoint","scoring","threshold","metric",
                                           "new_value","registry_value","diff","registry_version"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nComparison written: {out_csv}")
    return out_csv

def parse_summary_folder(folder, run_name, ckpt_name, scoring="S0", threshold=0.5, GT_sha256=None):
    """Find and parse a pedestrian_summary.txt under folder."""
    matches = glob.glob(os.path.join(folder, "**", "pedestrian_summary.txt"), recursive=True)
    if not matches: return None
    d = read_trackeval_summary(matches[0])
    if d is None: return None
    d.update({"run":run_name,"checkpoint":ckpt_name,"scoring":scoring,
               "threshold":threshold,"GT_sha256":GT_sha256 or ""})
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare new results against pre-tuning registry v1.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--new_dir",  help="Folder containing new TrackEval outputs (run as <run>_<ckpt> subfolders)")
    g.add_argument("--new_csv",  help="CSV file in registry format with new metrics")
    parser.add_argument("--out_dir", required=True, help="Output folder for comparison CSV")
    parser.add_argument("--run",   default="", help="Run label override (with --new_dir)")
    parser.add_argument("--ckpt",  default="", help="Checkpoint label override")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--scoring", default="S0")
    args = parser.parse_args()

    registry = load_registry()

    if args.new_csv:
        new_rows = []
        with open(args.new_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                new_rows.append(row)
        compare(new_rows, registry, args.out_dir, label="csv_comparison")
    else:
        # Try to auto-detect subfolders named <run>_<ckpt>
        new_rows = []
        if args.run and args.ckpt:
            d = parse_summary_folder(args.new_dir, args.run, args.ckpt, args.scoring, args.thresh)
            if d: new_rows.append(d)
        else:
            for sub in os.listdir(args.new_dir):
                subpath = os.path.join(args.new_dir, sub)
                if not os.path.isdir(subpath): continue
                parts = sub.split("_")
                if len(parts) >= 2:
                    run_label = parts[0]; ckpt_label = "_".join(parts[1:])
                    d = parse_summary_folder(subpath, run_label, ckpt_label, args.scoring, args.thresh)
                    if d: new_rows.append(d)
        if not new_rows:
            print("No summaries found. Use --run and --ckpt to specify.")
            sys.exit(1)
        compare(new_rows, registry, args.out_dir, label="dir_comparison")
