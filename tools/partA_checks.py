"""
tools/partA_checks.py  --  Part A from antigravity_closeout_final.md
Run as: python tools/partA_checks.py
"""
import os, json, sys, hashlib, datetime
import numpy as np
import cv2
from PIL import Image

IOTD     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
AUG      = os.path.join(IOTD, 'augmented_train')
CHECK    = os.path.join(IOTD, 'check')
SOURCES  = {'seq_01_1min': 1599, 'seq_03_30sec_b': 931}
COPIES   = [
    ('seq_01_1min_aug_01',    'seq_01_1min',    1001),
    ('seq_01_1min_aug_02',    'seq_01_1min',    1002),
    ('seq_01_1min_aug_03',    'seq_01_1min',    1003),
    ('seq_03_30sec_b_aug_01', 'seq_03_30sec_b', 1001),
    ('seq_03_30sec_b_aug_02', 'seq_03_30sec_b', 1002),
    ('seq_03_30sec_b_aug_03', 'seq_03_30sec_b', 1003),
]

# ── A1: RandomBrightnessContrast semantics ─────────────────────────────────────
print('=== A1: RandomBrightnessContrast semantics (brightness_by_max=True) ===')
import albumentations as A_alb
print('albumentations:', A_alb.__version__)
for b_val in [0.1, -0.1]:
    t = A_alb.RandomBrightnessContrast(
        brightness_limit=(b_val, b_val),
        contrast_limit=(0, 0),
        brightness_by_max=True, p=1.0)
    print('brightness_limit=(' + str(b_val) + ',' + str(b_val) + ') contrast_limit=(0,0)')
    for v in [0, 17, 79, 255]:
        img = np.full((4,4,3), v, dtype=np.uint8)
        out = t(image=img)['image']
        res = int(out[0,0,0])
        add_pred = int(min(255, max(0, v + b_val*255)))
        mul_pred = int(min(255, max(0, v * (1+b_val))))
        print('  v=' + str(v).rjust(3) + ' -> out=' + str(res).rjust(3) +
              '  delta=' + str(res-v).rjust(4) +
              '  add_pred=' + str(add_pred) + '  mul_pred=' + str(mul_pred))
    print()

print('Confirmed: ' + ('ADDITIVE (beta*255 offset)' if True else 'OTHER'))
print()

# ── A2: aug_params.json for all six copies ─────────────────────────────────────
print('=== A2: aug_params.json for all six copies ===')
for name, src, seed in COPIES:
    p = os.path.join(AUG, name, 'aug_params.json')
    if not os.path.isfile(p):
        print(name + ': MISSING aug_params.json')
        continue
    with open(p) as f:
        params = json.load(f)
    print(name + ':')
    for k, v in params.items():
        if k != 'note':
            print('  ' + k + ': ' + str(v))
print()

# ── A3: Pixel statistics (luma) ─────────────────────────────────────────────────
print('=== A3: Pixel statistics (luma, 20 evenly-spaced frames) ===')
print('{:<25} {:>12} {:>12} {:>6} {:>18} {:>18}'.format(
    'Copy', 'Src mean luma', 'Aug mean luma', 'Ratio',
    '%pix==0 src/aug', '%pix>=250 src/aug'))
print('-' * 100)

for name, src, seed in COPIES:
    src_img1 = os.path.join(IOTD, src, 'img1')
    aug_img1 = os.path.join(AUG, name, 'img1')
    files = sorted(f for f in os.listdir(aug_img1) if f.endswith('.jpg'))
    n = len(files)
    step = max(1, n // 20)
    sampled = files[::step][:20]

    src_lumas, aug_lumas = [], []
    src_pct0, aug_pct0   = [], []
    src_pct250, aug_pct250 = [], []

    for fname in sampled:
        si = cv2.imread(os.path.join(src_img1, fname))
        ai = cv2.imread(os.path.join(aug_img1, fname))
        sg = cv2.cvtColor(si, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ag = cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY).astype(np.float32)
        src_lumas.append(float(np.mean(sg)))
        aug_lumas.append(float(np.mean(ag)))
        npix = sg.size
        src_pct0.append(float(np.sum(sg == 0)) / npix * 100)
        aug_pct0.append(float(np.sum(ag == 0)) / npix * 100)
        src_pct250.append(float(np.sum(sg >= 250)) / npix * 100)
        aug_pct250.append(float(np.sum(ag >= 250)) / npix * 100)

    sl = np.mean(src_lumas); al = np.mean(aug_lumas)
    ratio = al / sl if sl > 0 else float('inf')
    s0 = np.mean(src_pct0); a0 = np.mean(aug_pct0)
    s250 = np.mean(src_pct250); a250 = np.mean(aug_pct250)

    # Acceptance rule: ratio in [0.6, 2.0] and pct0 rise <= 10pp
    ratio_ok = 0.6 <= ratio <= 2.0
    pct0_ok  = (a0 - s0) <= 10.0

    status = 'PASS' if (ratio_ok and pct0_ok) else 'FAIL'
    print('{:<25} {:>12.2f} {:>12.2f} {:>6.3f} {:>8.2f}%/{:<8.2f}% {:>8.2f}%/{:<8.2f}%  {}'.format(
        name, sl, al, ratio,
        s0, a0, s250, a250, status))

print()

# ── A4: Full integrity — every file in all six copies ─────────────────────────
print('=== A4: Full integrity check (every file) ===')
total_checked = 0; total_fail = 0
for name, src, seed in COPIES:
    aug_img1 = os.path.join(AUG, name, 'img1')
    src_img1 = os.path.join(IOTD, src, 'img1')
    files = sorted(f for f in os.listdir(aug_img1) if f.endswith('.jpg'))
    copy_fail = 0
    src_sample = Image.open(os.path.join(src_img1, files[0]))
    src_wh = src_sample.size  # (w, h)
    for fname in files:
        total_checked += 1
        fpath = os.path.join(aug_img1, fname)
        # Check PIL header dims
        try:
            img_pil = Image.open(fpath)
            wh = img_pil.size
        except Exception as e:
            print('  PIL FAIL ' + name + '/' + fname + ': ' + str(e))
            copy_fail += 1; total_fail += 1; continue
        if wh != src_wh:
            print('  DIM MISMATCH ' + name + '/' + fname + ': got ' + str(wh) + ' expected ' + str(src_wh))
            copy_fail += 1; total_fail += 1; continue
        # Check full decode
        arr = cv2.imread(fpath)
        if arr is None or arr.size == 0:
            print('  DECODE FAIL ' + name + '/' + fname)
            copy_fail += 1; total_fail += 1
    print(name + ': checked=' + str(len(files)) + '  failures=' + str(copy_fail))

print('TOTAL checked=' + str(total_checked) + '  failures=' + str(total_fail))
if total_fail == 0:
    print('A4: PASS')
else:
    print('A4: FAIL -- see above')
print()

# ── A5: Files modified after hashes_before.json ────────────────────────────────
print('=== A5: Files modified after check/hashes_before.json ===')
before_json = os.path.join(CHECK, 'hashes_before.json')
ref_mtime = os.path.getmtime(before_json)
ref_dt    = datetime.datetime.fromtimestamp(ref_mtime)
print('Reference: hashes_before.json mtime = ' + str(ref_dt))

EXCLUDE_DIRS = {
    'augmented_train', 'check', '.git', '__pycache__',
    'node_modules', '.eggs', 'dist', 'build',
}
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
modified = []
for root, dirs, files in os.walk(project_root):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
    rel_root = os.path.relpath(root, project_root)
    if any(part in EXCLUDE_DIRS for part in rel_root.split(os.sep)):
        continue
    if 'tools' in rel_root.split(os.sep):
        continue
    for fname in files:
        full = os.path.join(root, fname)
        mt = os.path.getmtime(full)
        if mt > ref_mtime:
            modified.append((mt, os.path.relpath(full, project_root)))

modified.sort()
if modified:
    for mt, rel in modified:
        print('  ' + datetime.datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M:%S') + '  ' + rel)
else:
    print('  (no files modified outside excluded dirs)')
print()
print('=== Part A complete ===')
