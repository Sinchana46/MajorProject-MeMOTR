import os, sys

C2_DIR = "results/phase6_c2_interim"
files = sorted([f for f in os.listdir(C2_DIR) if f.endswith(".pth")])

print("=== C2 ADAPTER CHECKPOINT SIZES ===")
for fn in files:
    path = os.path.join(C2_DIR, fn)
    size = os.path.getsize(path)
    print(f"  {fn}: {size} bytes")

# Try loading epoch_4
ep4 = os.path.join(C2_DIR, "c2_adapter_epoch_4.pth")
print(f"\n=== LOADING {ep4} ===")
size4 = os.path.getsize(ep4)
print(f"File size: {size4} bytes")

if size4 == 0:
    print("File is 0 bytes - cannot load. This is likely an empty placeholder.")
    print("Plausible for 209 params? NO - 209 float32 params = ~836 bytes minimum (weights only),")
    print("  ~1-3 KB with PyTorch serialization overhead.")
    print("CHECK 2: C2 checkpoints are 0 bytes - INVALID, cannot be used for inference.")
    sys.exit(0)

import torch
try:
    ckpt = torch.load(ep4, map_location="cpu")
    print(f"Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Keys: {list(ckpt.keys())}")
        total_params = 0
        for k, v in ckpt.items():
            if hasattr(v, "numel"):
                print(f"  {k}: shape={tuple(v.shape)}, numel={v.numel()}")
                total_params += v.numel()
        print(f"Total parameters: {total_params}")
        print(f"Expected: 209 (3-16-8-1 MLP with biases)")
        exp = 3*16+16 + 16*8+8 + 8*1+1  # = 48+16+128+8+8+1 = 209
        print(f"3-16-8-1 MLP param count: {exp}")
        print(f"Plausible for {size4} bytes: {'YES' if total_params <= 300 and size4 > 500 else 'MISMATCH'}")
    else:
        print(f"Content: {ckpt}")
except Exception as e:
    print(f"torch.load failed: {e}")
