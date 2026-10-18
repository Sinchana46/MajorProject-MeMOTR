#!/usr/bin/env python3
"""
Pre-tuning package: scripts/generate_all.py
Regenerates all figures, tables, and CSVs from the saved evaluation data in one command.
Run from the project root:
    python results/pre_tuning_baseline_v1/scripts/generate_all.py
"""
import os, sys, subprocess, hashlib

PYTHON = sys.executable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SCRIPTS = [
    os.path.join(ROOT, "tools", "pretune_stepA.py"),
    os.path.join(ROOT, "tools", "pretune_stepB.py"),
    os.path.join(ROOT, "tools", "pretune_stepC.py"),
]

DATA_DIR  = os.path.join(ROOT, "results", "pre_tuning_baseline_v1", "data")
TABLE_DIR = os.path.join(ROOT, "results", "pre_tuning_baseline_v1", "tables")
FIG_DIR   = os.path.join(ROOT, "results", "pre_tuning_baseline_v1", "figures")

EXPECTED_CSV = [
    "per_checkpoint_metrics.csv",
    "results_registry.csv",
    "detection_outcomes.csv",
]
EXPECTED_TABLES = [
    "table01_all_checkpoints.csv", "table02_selected_candidates.csv",
    "table03_detection_outcomes.csv", "table04_threshold_exploratory.csv",
    "table05_run_inventory.csv", "table06_dataset_summary.csv",
]
EXPECTED_FIGS = [
    "fig01_hota_vs_steps.png", "fig02_deta_assa_vs_steps.png",
    "fig03_fp_fn_vs_steps.png", "fig04_detection_outcomes.png",
    "fig05_candidate_comparison.png", "fig06_identity_metrics.png",
    "fig07_epochwise_summary.png", "fig08_qualitative.png",
    "pre_tuning_dashboard.png",
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def run_script(script):
    print(f"\n>>> Running {os.path.basename(script)} ...")
    result = subprocess.run(
        [PYTHON, "-u", script],
        cwd=ROOT, capture_output=False, timeout=300
    )
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}")
        return False
    return True

if __name__ == "__main__":
    print("="*60)
    print("generate_all.py: regenerating all pre-tuning baseline outputs")
    print("="*60)

    # Record hashes before
    before_hashes = {}
    for csv_name in EXPECTED_CSV:
        p = os.path.join(DATA_DIR, csv_name)
        if os.path.exists(p): before_hashes[csv_name] = sha256_file(p)

    success = True
    for script in SCRIPTS:
        if not run_script(script):
            success = False
            print(f"FAILED: {script}")

    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    # Check CSVs
    print("\nData CSVs:")
    for csv_name in EXPECTED_CSV:
        p = os.path.join(DATA_DIR, csv_name)
        if os.path.exists(p):
            new_h = sha256_file(p)
            old_h = before_hashes.get(csv_name)
            match = "(same as before)" if old_h and new_h == old_h else "(regenerated)"
            print(f"  OK  {csv_name}  {new_h[:16]}...  {match}")
        else:
            print(f"  MISSING  {csv_name}")

    print("\nTables:")
    for t in EXPECTED_TABLES:
        p = os.path.join(TABLE_DIR, t)
        print(f"  {'OK' if os.path.exists(p) else 'MISSING'}  {t}")

    print("\nFigures:")
    for f in EXPECTED_FIGS:
        p = os.path.join(FIG_DIR, f)
        print(f"  {'OK' if os.path.exists(p) else 'MISSING'}  {f}")

    print(f"\nOverall: {'SUCCESS' if success else 'PARTIAL FAILURE'}")
