import json, os, sys, math
import numpy as np

AUG_ROOT = "data/IOTD/augmented_train_v2"
SRC_ROOT  = "data/IOTD"

# Read all aug_params.json
copies = []
for d in sorted(os.listdir(AUG_ROOT)):
    dp = os.path.join(AUG_ROOT, d)
    pf = os.path.join(dp, "aug_params.json")
    if os.path.exists(pf):
        with open(pf) as f:
            params = json.load(f)
        copies.append((d, dp, params))

print("=== AUG_PARAMS.JSON ===")
for name, path, params in copies:
    print(f"  {name}: {json.dumps(params)}")

# Determine source sequence for each aug copy
def get_source(aug_name):
    if "seq_01" in aug_name: return "seq_01_1min", 1599
    if "seq_03" in aug_name: return "seq_03_30sec_b", 931
    raise ValueError(aug_name)

# Pearson correlation between two flat arrays
def pearson(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    num = ((a-ma)*(b-mb)).sum()
    den = math.sqrt(((a-ma)**2).sum() * ((b-mb)**2).sum())
    return num/den if den > 0 else 0.0

print("\n=== PEARSON CORRELATION CHECK (plain vs flipped) ===")
any_flipped = False

for aug_name, aug_path, params in copies:
    src_seq, src_total = get_source(aug_name)
    # pick 3 test frames
    if src_total >= 900:
        test_frames = [100, 400, 800]
    else:
        test_frames = [100, 400, min(900, src_total-1)]
    
    print(f"\n{aug_name}:")
    flipped = False
    for fidx in test_frames:
        fname = f"{fidx:06d}.jpg"
        src_path = os.path.join(SRC_ROOT, src_seq, "img1", fname)
        aug_path_img = os.path.join(aug_path, "img1", fname)
        
        if not os.path.exists(src_path) or not os.path.exists(aug_path_img):
            print(f"  frame {fname}: FILE MISSING")
            continue
        
        # Load grayscale (manual to avoid PIL dependency issues)
        import struct, zlib
        try:
            import cv2
            src_gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            aug_gray = cv2.imread(aug_path_img, cv2.IMREAD_GRAYSCALE)
        except:
            from PIL import Image
            src_gray = np.array(Image.open(src_path).convert("L"))
            aug_gray = np.array(Image.open(aug_path_img).convert("L"))
        
        src_flat  = src_gray.flatten()
        aug_flat  = aug_gray.flatten()
        src_flip_flat = np.fliplr(src_gray).flatten()
        
        r_plain  = pearson(src_flat, aug_flat)
        r_flip   = pearson(src_flip_flat, aug_flat)
        is_flip  = r_flip > r_plain
        print(f"  frame {fname}: r_plain={r_plain:.6f}  r_flip={r_flip:.6f}  FLIPPED={is_flip}")
        if is_flip:
            flipped = True
    
    status = "FLIPPED" if flipped else "NOT_FLIPPED"
    print(f"  => {aug_name}: {status}")
    if flipped:
        any_flipped = True

print("\n=== RESULT ===")
if any_flipped:
    print("CHECK 1: FAIL - at least one copy is FLIPPED. STOP.")
    sys.exit(1)
else:
    print("CHECK 1: PASS - no flipping detected in any of the 6 copies")
