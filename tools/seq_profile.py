"""
tools/seq_profile.py
--------------------
Profile IOTD sequences: track stats, class distribution, visibility values,
mean brightness, lighting estimate.

Usage:
    python tools/seq_profile.py
Outputs:
    stdout  -- full table
    data/IOTD/seq_profile.csv
"""

import os, csv, collections
import cv2
import numpy as np

IOTD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
SEQS = ['seq_01_1min', 'seq_02_30sec_a', 'seq_03_30sec_b']
LABELS = {1:'car', 2:'two_wheeler', 3:'auto', 4:'bus', 5:'truck'}

def parse_gt(gt_path):
    rows = []
    with open(gt_path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) < 9:
                continue
            rows.append({
                'frame':  int(c[0]),
                'tid':    int(c[1]),
                'x':      float(c[2]),
                'y':      float(c[3]),
                'w':      float(c[4]),
                'h':      float(c[5]),
                'conf':   int(c[6]),     # col 7 = not-ignored flag
                'cls':    int(c[7]),     # col 8 = class_id
                'vis':    float(c[8]),   # col 9 = visibility
            })
    return rows

def mean_brightness_sample(img1_dir, n_sample=50):
    """Sample n_sample evenly-spaced frames, return mean brightness."""
    files = sorted(f for f in os.listdir(img1_dir) if f.endswith('.jpg'))
    if not files:
        return None
    step = max(1, len(files) // n_sample)
    sampled = files[::step][:n_sample]
    brightnesses = []
    for fname in sampled:
        img = cv2.imread(os.path.join(img1_dir, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(np.mean(gray)))
    return float(np.mean(brightnesses)) if brightnesses else None

def lighting_label(mean_lum):
    if mean_lum is None:    return 'unknown'
    if mean_lum < 60:       return 'night'
    if mean_lum < 120:      return 'dusk'
    return 'day'

profiles = []
for seq in SEQS:
    seq_dir  = os.path.join(IOTD_ROOT, seq)
    img1_dir = os.path.join(seq_dir, 'img1')
    gt_path  = os.path.join(seq_dir, 'gt', 'gt.txt')

    rows = parse_gt(gt_path)
    n_rows   = len(rows)
    tids     = sorted(set(r['tid'] for r in rows))
    n_tracks = len(tids)
    frames   = sorted(set(r['frame'] for r in rows))

    cls_count = collections.Counter(r['cls'] for r in rows)
    vis_vals  = sorted(set(r['vis'] for r in rows))

    n_jpegs = len([f for f in os.listdir(img1_dir) if f.endswith('.jpg')])

    print(f'\n{"="*60}')
    print(f'SEQUENCE: {seq}')
    print(f'{"="*60}')
    print(f'  JPEG frames on disk    : {n_jpegs}')
    print(f'  gt.txt rows            : {n_rows}')
    print(f'  Annotated frames (uniq): {len(frames)}  [{min(frames)}..{max(frames)}]')
    print(f'  Distinct track IDs     : {n_tracks}  -> {tids}')
    print()

    print(f'  Class distribution (col 8 = class_id):')
    for cls in sorted(cls_count):
        print(f'    class {cls} ({LABELS.get(cls,"?"):>12}): {cls_count[cls]:>5} rows')
    print()

    print(f'  Visibility values (col 9, distinct): {vis_vals}')
    print()

    print(f'  Computing mean brightness (sampling 50 frames) ...')
    mb = mean_brightness_sample(img1_dir)
    ll = lighting_label(mb)
    print(f'  Mean brightness (0-255) : {mb:.1f}' if mb else '  Mean brightness: N/A')
    print(f'  Lighting estimate       : {ll}')

    profiles.append({
        'sequence':     seq,
        'jpegs':        n_jpegs,
        'gt_rows':      n_rows,
        'ann_frames':   len(frames),
        'frame_min':    min(frames),
        'frame_max':    max(frames),
        'n_tracks':     n_tracks,
        'track_ids':    str(tids),
        'cls1_car':     cls_count.get(1, 0),
        'cls2_two_wh':  cls_count.get(2, 0),
        'cls3_auto':    cls_count.get(3, 0),
        'cls4_bus':     cls_count.get(4, 0),
        'cls5_truck':   cls_count.get(5, 0),
        'vis_vals':     str(vis_vals),
        'mean_brightness': round(mb, 2) if mb else '',
        'lighting':     ll,
    })

# Save CSV
csv_path = os.path.join(IOTD_ROOT, 'seq_profile.csv')
fieldnames = list(profiles[0].keys())
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(profiles)

print(f'\n\nCSV saved: {csv_path}')
print('\n=== SPLIT ASSESSMENT ===')
print('Provisional: train=[seq_01_1min, seq_03_30sec_b]  val=[seq_02_30sec_a]')
train_frames = sum(p['jpegs'] for p in profiles if p['sequence'] in ['seq_01_1min','seq_03_30sec_b'])
val_frames   = sum(p['jpegs'] for p in profiles if p['sequence'] == 'seq_02_30sec_a')
print(f'  Train frames (real): {train_frames}  (seq_01={profiles[0]["jpegs"]} + seq_03={profiles[2]["jpegs"]})')
print(f'  Val   frames (real): {val_frames}')
print(f'  Train x (1+3 aug copies) = {train_frames * 4} total training samples')
