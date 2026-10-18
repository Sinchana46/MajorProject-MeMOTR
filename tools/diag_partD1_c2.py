"""
Part D.1: Inspect C2 adapter checkpoint state_dict contents in detail.
Search code for where adapter weights are actually stored/loaded.
"""
import os, sys, torch

# ─── 1. Load and inspect c2_adapter_epoch_4.pth ────────────────────────────
ep4 = "results/phase6_c2_interim/c2_adapter_epoch_4.pth"
print("=" * 70)
print(f"LOADING: {ep4}  ({os.path.getsize(ep4)} bytes)")
print("=" * 70)

ckpt = torch.load(ep4, map_location="cpu", weights_only=False)
print(f"Top-level keys: {list(ckpt.keys())}")
print(f"epoch: {ckpt.get('epoch')}")
print(f"avg_loss: {ckpt.get('avg_loss')}")
print(f"tau_min: {ckpt.get('tau_min')}")
print(f"tau_max: {ckpt.get('tau_max')}")
print(f"label: {ckpt.get('label')}")
print(f"baseline_checkpoint: {ckpt.get('baseline_checkpoint')}")

sd = ckpt.get("state_dict", {})
print(f"\nstate_dict type: {type(sd)}")
print(f"state_dict keys: {list(sd.keys()) if isinstance(sd, dict) else sd}")
if isinstance(sd, dict):
    for k, v in sd.items():
        if hasattr(v, "shape"):
            print(f"  {k}: dtype={v.dtype} shape={tuple(v.shape)} numel={v.numel()}")
        else:
            print(f"  {k}: {type(v).__name__} = {v!r}")

# ─── 2. Check full MeMOTR checkpoints for c2_adapter keys ─────────────────
print("\n" + "=" * 70)
print("SEARCHING FULL MEMOTR CHECKPOINTS FOR c2/adapter/tau KEYS")
print("=" * 70)
for label, path in [
    ("phase5", "results/phase5_interim_baseline/checkpoint_4.pth"),
    ("arm1",   "results/train_iotd_real/checkpoint_4.pth"),
]:
    print(f"\n--- {label}: {path} ---")
    full_ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_sd = full_ckpt.get("model", full_ckpt)
    c2_keys = [k for k in model_sd if any(x in k.lower() for x in ["c2", "adapter", "tau", "threshold", "scene", "brightness"])]
    if c2_keys:
        print(f"  FOUND {len(c2_keys)} matching keys:")
        for k in c2_keys[:10]:
            v = model_sd[k]
            print(f"    {k}: {tuple(v.shape) if hasattr(v,'shape') else v}")
    else:
        print(f"  No c2/adapter/tau keys found (checked {len(model_sd)} keys)")

# ─── 3. Search code for how C2 was loaded during Phase 6 inference ─────────
print("\n" + "=" * 70)
print("CODE SEARCH: how C2 adapter weights are loaded/used")
print("=" * 70)
import subprocess
for pattern in ["c2_adapter_checkpoint", "load.*c2", "ThresholdAdapter", "SceneAnalyzer",
                "tau_min", "tau_max", "adapter_checkpoint"]:
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", pattern, "MeMOTR/", "tools/"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"\nPattern: {pattern!r}")
        for line in result.stdout.strip().split("\n")[:6]:
            print(f"  {line}")

# Also grep run_inference_mot.py specifically
print("\n--- run_inference_mot.py C2 section ---")
with open("tools/run_inference_mot.py") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if any(x in line.lower() for x in ["c2", "adapter", "tau", "threshold_adapter", "scene"]):
        print(f"  L{i+1}: {line.rstrip()}")
