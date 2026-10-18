import os, glob
import numpy as np

IMG_DIR = "data/IOTD_val/seq_02_30sec_a/img1"
frames = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
step = max(1, len(frames) // 20)
sample = frames[::step][:20]

import cv2
lumas = []
for fp in sample:
    img = cv2.imread(fp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lumas.append(gray.mean())

print(f"Frames sampled: {len(lumas)}")
print(f"Mean luma: {np.mean(lumas):.2f}")
print(f"Min: {np.min(lumas):.2f}  Max: {np.max(lumas):.2f}")
label = "NIGHT" if np.mean(lumas) < 60 else ("DUSK" if np.mean(lumas) < 100 else "DAY")
print(f"Classification: {label}")
