"""
Section 1.3 + 1.4 verification script.
Run from D:/MajorProject-MeMOTR:
    python -u tools/tune_verify.py 2>&1 | Tee-Object check/logs/tune1_verify.txt
"""
import os, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MeMOTR"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Config paths ──────────────────────────────────────────────────────────────
CONFIGS = {
    "arm1":       "configs/train_iotd_real.yaml",
    "seed1":      "configs/train_iotd_real_seed1.yaml",
    "seed2":      "configs/train_iotd_real_seed2.yaml",
    "assoc_lr0":  "configs/train_iotd_real_assoc_lr0.yaml",
    "lr_half":    "configs/train_iotd_real_lr_half.yaml",
}

def load_yaml(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
        return yaml.safe_load(f.read())

print("=" * 70)
print("SECTION 1.3 — CONFIG DIFFS vs train_iotd_real.yaml")
print("=" * 70)

base = load_yaml(CONFIGS["arm1"])
for name, cpath in CONFIGS.items():
    if name == "arm1":
        continue
    cpath_full = os.path.join(ROOT, cpath)
    if not os.path.exists(cpath_full):
        print(f"\n  {name}: FILE NOT FOUND at {cpath}")
        continue
    cfg = load_yaml(cpath)
    diffs = {}
    all_keys = set(list(base.keys()) + list(cfg.keys()))
    for k in sorted(all_keys):
        bv = base.get(k, "<absent>")
        cv = cfg.get(k, "<absent>")
        if bv != cv:
            diffs[k] = (bv, cv)
    print(f"\n  {name} ({cpath})  [{len(diffs)} diff(s)]")
    for k, (bv, cv) in diffs.items():
        print(f"    {k}: arm1={bv!r}  ->  {name}={cv!r}")

# ── Section 1.4 — Optimizer dry check ────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1.4 — OPTIMIZER DRY CHECK (all configs)")
print("=" * 70)

# Read train_engine.get_param_groups source
import importlib
import types

def build_fake_param_groups(config: dict):
    """
    Replicate get_param_groups() logic without loading the actual model.
    Uses the LR_QUERY_UPDATER key if present, else falls back to LR.
    """
    groups = [
        {"name": "lr_backbone",       "lr": config.get("LR_BACKBONE", 0.0)},
        {"name": "lr_points",         "lr": config.get("LR_POINTS",   0.0)},
        {"name": "lr_query_updater",  "lr": config.get("LR_QUERY_UPDATER", config.get("LR", 0.0))},
        {"name": "lr",                "lr": config.get("LR",          0.0)},
    ]
    return groups

print("\nNOTE: param counts not shown (would require CUDA model load).")
print("LR values are read directly from config keys.\n")
print(f"  {'Config':<15} {'lr_backbone':>12} {'lr_points':>12} {'lr_query_upd':>14} {'lr_rest':>12}")
print("  " + "-"*65)

for name, cpath in CONFIGS.items():
    cpath_full = os.path.join(ROOT, cpath)
    if not os.path.exists(cpath_full):
        print(f"  {name:<15}  MISSING")
        continue
    cfg = load_yaml(cpath)
    groups = build_fake_param_groups(cfg)
    lrs = [g["lr"] for g in groups]
    # Mark if LR_QUERY_UPDATER is explicitly in config
    has_qupd_key = "LR_QUERY_UPDATER" in cfg
    qupd_marker = "*" if has_qupd_key else " "
    print(f"  {name:<15} {lrs[0]:>12.2e} {lrs[1]:>12.2e} {lrs[2]:>13.2e}{qupd_marker} {lrs[3]:>12.2e}")

print("\n  * = LR_QUERY_UPDATER key explicitly in config")
print("  Without the key, lr_query_updater falls back to LR.")

# ── Check OUTPUTS_DIR uniqueness ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("OUTPUTS_DIR UNIQUENESS CHECK")
print("=" * 70)
dirs_seen = {}
for name, cpath in CONFIGS.items():
    cpath_full = os.path.join(ROOT, cpath)
    if not os.path.exists(cpath_full):
        continue
    cfg = load_yaml(cpath)
    od = cfg.get("OUTPUTS_DIR", "<absent>")
    conflict = dirs_seen.get(od)
    status = f"CONFLICT with {conflict}" if conflict else "OK"
    print(f"  {name:<15}  {od}  [{status}]")
    dirs_seen[od] = name

# ── Check PRETRAINED_MODEL ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PRETRAINED_MODEL CHECK")
print("=" * 70)
expected_ckpt = os.path.join(ROOT, "results/phase5_interim_baseline/checkpoint_4.pth").replace("\\", "/")
for name, cpath in CONFIGS.items():
    cpath_full = os.path.join(ROOT, cpath)
    if not os.path.exists(cpath_full):
        continue
    cfg = load_yaml(cpath)
    pm = str(cfg.get("PRETRAINED_MODEL", "")).replace("\\", "/")
    resume = cfg.get("RESUME")
    seed = cfg.get("SEED")
    pm_exists = os.path.exists(pm)
    print(f"  {name:<15}  PRETRAINED={pm}  exists={pm_exists}  RESUME={resume}  SEED={seed}")

print("\nAll checks done.")
