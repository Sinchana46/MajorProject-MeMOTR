import os, re, sys
import torch

# ─── 4.1 Training log provenance ───────────────────────────────────────────
print("="*70)
print("CHECK 4.1: PRETRAINED WEIGHTS LINE FROM TRAINING LOGS")
print("="*70)

for arm, logpath in [
    ("Arm1", "results/train_iotd_real/train_iotd_real.log"),
    ("Arm2", "results/train_iotd_real_aug/train_iotd_real_aug.log"),
]:
    if not os.path.exists(logpath):
        # Try alternative log location
        for alt in [f"results/{arm.lower()}_train.log",
                    f"results/train_iotd_real{'_aug' if '2' in arm else ''}/train/log.txt"]:
            if os.path.exists(alt):
                logpath = alt
                break
    print(f"\n--- {arm}: {logpath} ---")
    if not os.path.exists(logpath):
        print(f"  LOG NOT FOUND at {logpath}")
        continue
    with open(logpath, errors="replace") as f:
        lines = f.readlines()
    # Find pretrained/resume lines
    for i, line in enumerate(lines):
        if any(x in line.lower() for x in ["pretrain", "resume", "checkpoint", "load", "weight"]):
            if any(x in line for x in ["phase5", "checkpoint_4", "pth", "Loading"]):
                print(f"  L{i+1}: {line.rstrip()}")
            elif i < 50:  # early lines
                print(f"  L{i+1}: {line.rstrip()}")

# ─── 4.2 Sequence loading + iters per epoch ────────────────────────────────
print("\n" + "="*70)
print("CHECK 4.2: SEQUENCE LOADING + ITERATIONS PER EPOCH")
print("="*70)
for arm, logdir in [
    ("Arm1", "results/train_iotd_real/train"),
    ("Arm2", "results/train_iotd_real_aug/train"),
]:
    logpath = os.path.join(logdir, "log.txt")
    print(f"\n--- {arm}: {logpath} ---")
    if not os.path.exists(logpath):
        print("  NOT FOUND")
        continue
    with open(logpath, errors="replace") as f:
        lines = f.readlines()
    # Get sequence loaded lines and epoch iter counts
    for i, line in enumerate(lines[:200]):
        if any(x in line for x in ["VehicleMOT", "Loaded", "sequences", "seq_", "iter", "[Epoch"]):
            print(f"  L{i+1}: {line.rstrip()}")

# ─── 4.3 LR per epoch ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("CHECK 4.3: LR PER EPOCH")
print("="*70)
for arm, logpath in [
    ("Arm1", "results/train_iotd_real/train/log.txt"),
    ("Arm2", "results/train_iotd_real_aug/train/log.txt"),
]:
    print(f"\n--- {arm} ---")
    if not os.path.exists(logpath):
        print("  NOT FOUND"); continue
    with open(logpath, errors="replace") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("[Epoch ") and "lr=" in line:
            print(f"  {line.rstrip()}")

# ─── 4.4 Checkpoint parameter comparison ───────────────────────────────────
print("\n" + "="*70)
print("CHECK 4.4: CHECKPOINT PARAMETER COMPARISON")
print("="*70)

ckpts = {
    "phase5": "results/phase5_interim_baseline/checkpoint_4.pth",
    "arm1":   "results/train_iotd_real/checkpoint_4.pth",
    "arm2":   "results/train_iotd_real_aug/checkpoint_4.pth",
}

loaded = {}
for name, path in ckpts.items():
    print(f"\nLoading {name}: {path}")
    ckpt = torch.load(path, map_location="cpu")
    # Get state dict
    if "model" in ckpt:
        sd = ckpt["model"]
        meta_keys = [k for k in ckpt if k != "model"]
        print(f"  metadata keys: {meta_keys}")
        for mk in meta_keys:
            if mk in ("epoch", "args"):
                print(f"  {mk}: {ckpt[mk]}")
    else:
        sd = ckpt
        print(f"  top-level keys: {list(sd.keys())[:10]}")
    loaded[name] = sd
    # Count params
    total = sum(v.numel() for v in sd.values() if hasattr(v,"numel"))
    print(f"  total params: {total:,}")

# Backbone diff: phase5 vs arm1, phase5 vs arm2
print("\n--- Backbone parameter diffs ---")
for arm in ["arm1", "arm2"]:
    bb_keys = [k for k in loaded["phase5"] if k.startswith("backbone")]
    if not bb_keys:
        bb_keys = [k for k in loaded["phase5"] if "backbone" in k]
    diffs = []
    for k in bb_keys:
        if k in loaded[arm]:
            d = (loaded["phase5"][k].float() - loaded[arm][k].float()).abs().max().item()
            diffs.append(d)
    max_diff = max(diffs) if diffs else -1
    print(f"  phase5 vs {arm}: backbone max abs diff = {max_diff:.8f}  (expected 0.0 = frozen)")

# Non-backbone diff: encoder/decoder/query_updater
print("\n--- Non-backbone parameter diffs ---")
for arm in ["arm1", "arm2"]:
    non_bb_keys = [k for k in loaded["phase5"] if not k.startswith("backbone")]
    diffs = []
    for k in non_bb_keys[:100]:  # sample
        if k in loaded[arm]:
            d = (loaded["phase5"][k].float() - loaded[arm][k].float()).abs().mean().item()
            diffs.append(d)
    mean_diff = sum(diffs)/len(diffs) if diffs else -1
    print(f"  phase5 vs {arm}: non-backbone mean abs diff = {mean_diff:.8f} (expected > 0)")

# Arm1 vs Arm2 diff
print("\n--- Arm1 vs Arm2 ---")
all_keys = [k for k in loaded["arm1"] if k in loaded["arm2"]]
diffs = []
for k in all_keys[:200]:
    d = (loaded["arm1"][k].float() - loaded["arm2"][k].float()).abs().mean().item()
    diffs.append(d)
mean_diff = sum(diffs)/len(diffs) if diffs else -1
max_diff  = max((loaded["arm1"][k].float() - loaded["arm2"][k].float()).abs().max().item() for k in all_keys[:50])
print(f"  arm1 vs arm2: mean abs diff = {mean_diff:.8f}, max = {max_diff:.8f} (expected > 0)")

print("\nCHECK 4: DONE")
