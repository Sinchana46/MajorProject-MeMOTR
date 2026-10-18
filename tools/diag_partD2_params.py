"""
Part D.2: Recompute pairwise mean absolute differences per top-level module
for the non-backbone parameters of phase5, arm1, arm2 checkpoint_4.
"""
import os, sys, torch
from collections import defaultdict

CKPTS = {
    "phase5": "results/phase5_interim_baseline/checkpoint_4.pth",
    "arm1":   "results/train_iotd_real/checkpoint_4.pth",
    "arm2":   "results/train_iotd_real_aug/checkpoint_4.pth",
}

print("=" * 70)
print("LOADING CHECKPOINTS")
print("=" * 70)
state_dicts = {}
for name, path in CKPTS.items():
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    state_dicts[name] = {k: v.float() for k, v in sd.items() if hasattr(v, "float")}
    total = sum(v.numel() for v in state_dicts[name].values())
    print(f"  {name}: {len(state_dicts[name])} tensors, {total:,} params")

# Find shared non-backbone keys
all_keys = set(state_dicts["phase5"].keys())
for n in ["arm1", "arm2"]:
    all_keys &= set(state_dicts[n].keys())
non_bb_keys = [k for k in sorted(all_keys) if not k.startswith("backbone")]
bb_keys     = [k for k in sorted(all_keys) if k.startswith("backbone")]
print(f"\nShared keys total: {len(all_keys)}")
print(f"Backbone keys: {len(bb_keys)}")
print(f"Non-backbone keys: {len(non_bb_keys)}")

def pairwise_diff(sd_a, sd_b, keys):
    total_mad = 0.0
    total_numel = 0
    for k in keys:
        if k in sd_a and k in sd_b:
            d = (sd_a[k] - sd_b[k]).abs()
            total_mad += d.sum().item()
            total_numel += d.numel()
    return total_mad / total_numel if total_numel > 0 else 0.0

def pairwise_max(sd_a, sd_b, keys):
    m = 0.0
    for k in keys:
        if k in sd_a and k in sd_b:
            m = max(m, (sd_a[k] - sd_b[k]).abs().max().item())
    return m

# Per-top-level-module breakdown
def top_module(key):
    return key.split(".")[0]

modules = sorted(set(top_module(k) for k in non_bb_keys))

print("\n" + "=" * 70)
print("PAIRWISE MEAN ABSOLUTE DIFFERENCES (non-backbone params)")
print("=" * 70)
pairs = [("phase5", "arm1"), ("phase5", "arm2"), ("arm1", "arm2")]
for a, b in pairs:
    mad = pairwise_diff(state_dicts[a], state_dicts[b], non_bb_keys)
    mx  = pairwise_max(state_dicts[a], state_dicts[b], non_bb_keys)
    print(f"  {a} vs {b}: mean abs diff = {mad:.8f}  max = {mx:.8f}")

print("\nCheck triangle inequality:")
d_p5_a1 = pairwise_diff(state_dicts["phase5"], state_dicts["arm1"], non_bb_keys)
d_p5_a2 = pairwise_diff(state_dicts["phase5"], state_dicts["arm2"], non_bb_keys)
d_a1_a2 = pairwise_diff(state_dicts["arm1"],   state_dicts["arm2"], non_bb_keys)
print(f"  |d(p5,a2) - d(p5,a1)| = {abs(d_p5_a2 - d_p5_a1):.8f}")
print(f"  d(a1,a2) reported previously = 0.0000271")
print(f"  d(a1,a2) recomputed          = {d_a1_a2:.8f}")
print(f"  Previous report was {'WRONG' if abs(d_a1_a2 - 0.0000271) > 1e-5 else 'OK'}")

print("\n" + "=" * 70)
print("PER-MODULE BREAKDOWN")
print("=" * 70)
print(f"{'Module':<35} {'p5-a1 MAD':>12} {'p5-a2 MAD':>12} {'a1-a2 MAD':>12}")
print("-" * 75)
for mod in modules:
    mod_keys = [k for k in non_bb_keys if top_module(k) == mod]
    d1 = pairwise_diff(state_dicts["phase5"], state_dicts["arm1"], mod_keys)
    d2 = pairwise_diff(state_dicts["phase5"], state_dicts["arm2"], mod_keys)
    d3 = pairwise_diff(state_dicts["arm1"],   state_dicts["arm2"], mod_keys)
    print(f"  {mod:<33} {d1:>12.8f} {d2:>12.8f} {d3:>12.8f}")

print("\n" + "=" * 70)
print("BACKBONE DIFF CONFIRM")
print("=" * 70)
for a, b in pairs:
    mx = pairwise_max(state_dicts[a], state_dicts[b], bb_keys)
    print(f"  {a} vs {b}: backbone max abs diff = {mx:.10f}")
