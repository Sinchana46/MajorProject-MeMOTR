"""
Part A: Verify checkpoint epoch metadata and print step counts.
Also print SHA256 of both arm checkpoint_4 files (for Part D.2).
"""
import os, sys, hashlib
import torch

def sha256f(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

CKPTS = {
    "arm1": "results/train_iotd_real",
    "arm2": "results/train_iotd_real_aug",
    "phase5": "results/phase5_interim_baseline",
}

# Steps per epoch
ARM1_STEPS = 2528
ARM2_STEPS = 10112

print("=" * 65)
print("PART A.1: CHECKPOINT EPOCH METADATA")
print("=" * 65)

meta = {}
for arm, base in CKPTS.items():
    print(f"\n--- {arm} ({base}) ---")
    for n in range(5):
        path = os.path.join(base, f"checkpoint_{n}.pth")
        if not os.path.exists(path):
            print(f"  checkpoint_{n}.pth: NOT FOUND")
            continue
        size_mb = os.path.getsize(path) / 1e6
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", ckpt.get("args", {}).get("epoch", "?"))
        # Check states/args for epoch
        if "states" in ckpt:
            states = ckpt["states"]
            if isinstance(states, dict) and "epoch" in states:
                epoch = states["epoch"]
        # Calculate cumulative steps completed (after saving):
        # checkpoint_N is saved at the END of epoch N
        if arm == "arm1":
            steps = (n + 1) * ARM1_STEPS
        elif arm == "arm2":
            steps = (n + 1) * ARM2_STEPS
        else:
            steps = "N/A (Phase5 = 0 IOTD steps)"
        
        print(f"  checkpoint_{n}.pth: epoch_key={epoch!r}  size={size_mb:.1f}MB  "
              f"cumulative_steps={steps}")
        meta[(arm, n)] = {"epoch": epoch, "steps": steps}

# SHA256 for arm checkpoint_4 files
print("\n" + "=" * 65)
print("CHECKPOINT_4 SHA256 (for Part D.2)")
print("=" * 65)
for arm, base in [("arm1", "results/train_iotd_real"),
                   ("arm2", "results/train_iotd_real_aug")]:
    p = os.path.join(base, "checkpoint_4.pth")
    h = sha256f(p)
    print(f"  {arm}/checkpoint_4.pth: {h}")

print("\n" + "=" * 65)
print("EQUAL-STEP PAIRS")
print("=" * 65)
print(f"Arm 1 steps/epoch: {ARM1_STEPS}")
print(f"Arm 2 steps/epoch: {ARM2_STEPS}")
print(f"Equal-step at 10,112 steps:")
print(f"  Arm 1 checkpoint_3 = end of epoch 3 = {4*ARM1_STEPS} steps")
print(f"  Arm 2 checkpoint_0 = end of epoch 0 = {1*ARM2_STEPS} steps")
print(f"  Match: {4*ARM1_STEPS == ARM2_STEPS}")
print(f"\nAll arm1 checkpoints vs arm2 by steps:")
print(f"{'Ckpt':>8} | {'Arm1 steps':>12} | {'Arm2 steps':>12}")
for n in range(5):
    a1 = (n+1)*ARM1_STEPS
    a2 = (n+1)*ARM2_STEPS
    print(f"  ckpt_{n} | {a1:>12,} | {a2:>12,}")
