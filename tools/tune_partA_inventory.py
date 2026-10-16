"""
Part A: Experiment inventory verification.
Reads configs, logs, result files to verify all claimed facts.
"""
import os, sys, re, glob, hashlib, configparser
import torch

PROJECT = "D:/MajorProject-MeMOTR"

def sha256f(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_yaml_simple(path):
    """Simple key:value YAML loader (no nested dicts)."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or ':' not in line:
                continue
            k, _, v = line.partition(':')
            d[k.strip()] = v.strip()
    return d

print("=" * 70)
print("PART A: EXPERIMENT INVENTORY VERIFICATION")
print("=" * 70)

# ─── A1. Phase 5 checkpoint ────────────────────────────────────────────────
print("\n--- Phase 5 ---")
p5 = "results/phase5_interim_baseline/checkpoint_4.pth"
if os.path.exists(p5):
    size_mb = os.path.getsize(p5) / 1e6
    ckpt = torch.load(p5, map_location="cpu", weights_only=False)
    model_keys = list(ckpt.get("model", ckpt).keys())[:3]
    print(f"  checkpoint_4.pth: {size_mb:.1f} MB  sample_keys={model_keys}")
    # Check for any IOTD-related provenance in metadata
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "lr_scheduler")}
    print(f"  metadata keys: {list(meta.keys())}")
    print(f"  VERIFIED: phase5 checkpoint_4 exists, {size_mb:.0f} MB")
else:
    print("  NOT FOUND")

# ─── A2. Arm 1 config ──────────────────────────────────────────────────────
print("\n--- Arm 1: configs/train_iotd_real.yaml ---")
cfg1 = load_yaml_simple("configs/train_iotd_real.yaml")
print(f"  PRETRAINED_MODEL: {cfg1.get('PRETRAINED_MODEL','?')}")
print(f"  RESUME:           {cfg1.get('RESUME','?')}")
print(f"  EPOCHS:           {cfg1.get('EPOCHS','?')}")
print(f"  LR:               {cfg1.get('LR','?')}")
print(f"  LR_BACKBONE:      {cfg1.get('LR_BACKBONE','?')}")
print(f"  LR_DROP:          {cfg1.get('LR_DROP','?')}")
print(f"  BATCH_SIZE:       {cfg1.get('BATCH_SIZE','?')}")
print(f"  DATASET:          {cfg1.get('DATASET','?')}")
print(f"  DATA_ROOT:        {cfg1.get('DATA_ROOT','?')}")
print(f"  OUTPUT_DIR:       {cfg1.get('OUTPUT_DIR','?')}")
print(f"  SEED:             {cfg1.get('SEED','NOT SET')}")
# Find SPLIT_DIRS or MOT_SEQS
with open("configs/train_iotd_real.yaml") as f:
    raw1 = f.read()
seq_lines = [l.strip() for l in raw1.split('\n') if 'seq' in l.lower() or 'split' in l.lower() or 'seqs' in l.lower()]
print(f"  Sequence-related lines: {seq_lines[:6]}")

# ─── A3. Arm 2 config ──────────────────────────────────────────────────────
print("\n--- Arm 2: configs/train_iotd_real_aug.yaml ---")
cfg2 = load_yaml_simple("configs/train_iotd_real_aug.yaml")
print(f"  PRETRAINED_MODEL: {cfg2.get('PRETRAINED_MODEL','?')}")
print(f"  RESUME:           {cfg2.get('RESUME','?')}")
print(f"  EPOCHS:           {cfg2.get('EPOCHS','?')}")
print(f"  LR:               {cfg2.get('LR','?')}")
print(f"  LR_BACKBONE:      {cfg2.get('LR_BACKBONE','?')}")
print(f"  LR_DROP:          {cfg2.get('LR_DROP','?')}")
print(f"  BATCH_SIZE:       {cfg2.get('BATCH_SIZE','?')}")
print(f"  SEED:             {cfg2.get('SEED','NOT SET')}")
with open("configs/train_iotd_real_aug.yaml") as f:
    raw2 = f.read()
seq_lines2 = [l.strip() for l in raw2.split('\n') if 'seq' in l.lower() or 'split' in l.lower() or 'seqs' in l.lower()]
print(f"  Sequence-related lines: {seq_lines2[:8]}")

# ─── A4. Training logs - seconds per step ─────────────────────────────────
print("\n--- Training Duration from logs ---")
for arm, log_pattern in [
    ("arm1", "check/logs/train_iotd_real*.txt"),
    ("arm2", "check/logs/train_iotd_real_aug*.txt"),
]:
    logs = glob.glob(log_pattern)
    if not logs:
        logs = glob.glob(f"results/train_iotd_real{'_aug' if 'aug' in arm else ''}/log*.txt")
    if not logs:
        logs = glob.glob(f"results/train_iotd_real{'_aug' if 'aug' in arm else ''}/*.txt")
    print(f"  {arm}: found log files: {[os.path.basename(l) for l in logs[:3]]}")
    for log in logs[:1]:
        with open(log) as f:
            content = f.read()
        # Look for time/duration hints
        time_lines = [l for l in content.split('\n') if any(x in l.lower() for x in ['time','duration','elapsed','epoch','step/s','it/s','s/it'])]
        print(f"  {arm} time-related lines (first 5): {time_lines[:5]}")

# ─── A5. Steps per epoch verification ─────────────────────────────────────
print("\n--- Steps per Epoch ---")
# From training logs, find iter counts
for arm, log_pattern, expected_steps in [
    ("arm1", "check/logs/train_iotd_real*.txt", 2528),
    ("arm2", "check/logs/train_iotd_real_aug*.txt", 10112),
]:
    logs = glob.glob(log_pattern)
    if logs:
        with open(logs[0]) as f:
            content = f.read()
        # Find lines mentioning max iter or total steps
        iter_lines = [l for l in content.split('\n') if 'max_it' in l.lower() or 'len(data' in l.lower() or 'dataset size' in l.lower() or 'max_iter' in l.lower()]
        print(f"  {arm}: iter-related lines: {iter_lines[:3]}")
    print(f"  {arm}: claimed steps/epoch = {expected_steps}")

# ─── A6. Clip length / SAMPLE_LENGTHS ─────────────────────────────────────
print("\n--- SAMPLE_LENGTHS ---")
for name, raw in [("arm1", raw1), ("arm2", raw2)]:
    sl = [l for l in raw.split('\n') if 'sample_length' in l.lower() or 'clip' in l.lower()]
    print(f"  {name}: {sl[:4]}")

# ─── A7. Existing result folders ──────────────────────────────────────────
print("\n--- All results/ folders ---")
for root, dirs, files in os.walk("results"):
    dirs.sort()
    level = root.replace("results", "").count(os.sep)
    if level > 2:
        del dirs[:]
        continue
    indent = "  " * level
    txt_files = [f for f in files if f.endswith('.txt')]
    pth_files = [f for f in files if f.endswith('.pth')]
    if txt_files or pth_files or level <= 1:
        print(f"{indent}{os.path.basename(root)}/"
              f"  [{len(pth_files)} .pth, {len(txt_files)} .txt]")

# ─── A8. Random seed in code ──────────────────────────────────────────────
print("\n--- Random Seed in main.py ---")
seed_lines = []
for pyfile in ["MeMOTR/main.py", "MeMOTR/train_engine.py"]:
    if os.path.exists(pyfile):
        with open(pyfile) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'seed' in line.lower() or 'manual_seed' in line.lower() or 'random.seed' in line.lower():
                seed_lines.append(f"  {pyfile}:L{i+1}: {line.rstrip()}")
print('\n'.join(seed_lines[:10]))

# ─── A9. Tracker output files for all thresholds ─────────────────────────
print("\n--- Saved tracker output files ---")
for root, dirs, files in os.walk("results/eval_iotd_val"):
    for f in sorted(files):
        if f.endswith(".txt") and "seq_02" in f:
            full = os.path.join(root, f)
            with open(full) as fh:
                rows = sum(1 for l in fh if l.strip())
            rel = full.replace("results/eval_iotd_val/", "")
            print(f"  {rel}: {rows} rows")

print("\nPart A DONE")
