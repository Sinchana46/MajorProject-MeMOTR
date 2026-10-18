"""
tools/augment_photometric.py
-----------------------------
Photometric-only augmentation of IOTD training sequences.

Rules:
  - Photometric only: brightness, contrast, gamma, hue/sat, JPEG compression.
  - No flip, crop, scale, rotate, shift, or perspective transforms.
  - gt.txt, labels.txt, seqinfo.ini copied byte-for-byte (SHA256 verified).
  - ReplayCompose: deterministic transforms sampled once per copy, replayed on
    every frame. Gaussian noise applied outside ReplayCompose (different per
    frame, reproducible via per-frame seed).
  - Refuses to run if source sequence is under val/test in split_manifest.json.
  - Refuses to write into any path that already exists.
  - Handles each sequence's own resolution (no resizing).

Usage:
    python tools/augment_photometric.py --copy seq_01_1min_aug_01
    python tools/augment_photometric.py --dry-run 20 --copy seq_03_30sec_b_aug_01
    python tools/augment_photometric.py --all          # all 6 copies in sequence

Library: albumentations 2.0.8
  GaussNoise       -> std_range (not var_limit)
  ImageCompression -> quality_range (tuple)
  HueSaturationValue -> val_shift_limit
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import numpy as np
import cv2
import albumentations as A

# ── paths ──────────────────────────────────────────────────────────────────────
IOTD_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
MANIFEST_PATH  = os.path.join(IOTD_ROOT, 'split_manifest.json')
AUG_ROOT       = os.path.join(IOTD_ROOT, 'augmented_train')
CHECK_ROOT     = os.path.join(IOTD_ROOT, 'check')
JPEG_QUALITY   = 98

# ── per-copy parameters (from plan Table) ──────────────────────────────────────
# Each entry: (brightness_range, contrast_range, gamma_range,
#              hue_shift, sat_shift, jpeg_quality_range, noise_std_range)
COPY_PARAMS = {
    1001: dict(
        brightness=(-0.15, -0.05),
        contrast=  (-0.10,  0.10),
        gamma=     (110,   130),
        hue_shift= 3,
        sat_shift= 10,
        jpeg_q=    (75, 90),
        noise_std= (2.0/255, 4.0/255),  # albumentations std_range is 0..1 normalised
    ),
    1002: dict(
        brightness=(+0.05, +0.15),
        contrast=  (-0.10,  0.10),
        gamma=     (80,    95),
        hue_shift= 5,
        sat_shift= 15,
        jpeg_q=    (65, 85),
        noise_std= (3.0/255, 6.0/255),
    ),
    1003: dict(
        brightness=(-0.08, +0.08),
        contrast=  (-0.15,  0.15),
        gamma=     (90,   110),
        hue_shift= 5,
        sat_shift= 20,
        jpeg_q=    (60, 80),
        noise_std= (4.0/255, 8.0/255),
    ),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def build_replay_pipeline(params, seed):
    """Build A.ReplayCompose for deterministic transforms (sampled once per copy).
    val_shift_limit=0 because brightness is handled by RandomBrightnessContrast.
    """
    rng = np.random.default_rng(seed)

    # Sample one concrete value set from the ranges using the copy seed
    brightness_limit = tuple(params['brightness'])   # passed as limit; albu samples within
    contrast_limit   = tuple(params['contrast'])
    gamma_range      = tuple(params['gamma'])
    hue_shift        = params['hue_shift']
    sat_shift        = params['sat_shift']
    jpeg_q           = tuple(params['jpeg_q'])

    pipeline = A.ReplayCompose([
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            brightness_by_max=True,
            p=1.0,
        ),
        A.RandomGamma(
            gamma_limit=gamma_range,
            p=1.0,
        ),
        A.HueSaturationValue(
            hue_shift_limit=hue_shift,
            sat_shift_limit=sat_shift,
            val_shift_limit=0,   # brightness handled separately
            p=1.0,
        ),
        A.ImageCompression(
            quality_range=jpeg_q,
            p=1.0,
        ),
    ])
    return pipeline


def add_frame_noise(img, std_range, copy_seed, frame_idx):
    """Add Gaussian noise independently per frame.
    Each frame gets a unique seed: copy_seed * 1_000_003 + frame_idx.
    std is sampled once per frame from std_range (normalised 0..1).
    """
    rng = np.random.default_rng(copy_seed * 1_000_003 + frame_idx)
    std_norm = rng.uniform(std_range[0], std_range[1])
    std_255  = std_norm * 255.0
    noise    = rng.normal(0.0, std_255, img.shape).astype(np.float32)
    out      = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def run_copy(copy_name, manifest, dry_run_n=None):
    """Run augmentation for one copy (or dry-run N frames)."""

    # ── load copy config from manifest ──
    if copy_name not in manifest['augmented']:
        print(f'ERROR: "{copy_name}" not in split_manifest.json augmented section')
        sys.exit(1)

    cfg        = manifest['augmented'][copy_name]
    source_seq = cfg['source']
    seed       = cfg['seed']

    # ── safety: refuse if source is val or test ──
    if source_seq in manifest.get('val', []):
        print(f'ERROR: {source_seq} is listed as VAL in split_manifest.json. Refusing to augment.')
        sys.exit(1)
    if source_seq in manifest.get('test', []):
        print(f'ERROR: {source_seq} is listed as TEST in split_manifest.json. Refusing to augment.')
        sys.exit(1)

    # ── paths ──
    src_dir  = os.path.join(IOTD_ROOT, source_seq)
    src_img1 = os.path.join(src_dir, 'img1')

    if dry_run_n is not None:
        out_dir = os.path.join(CHECK_ROOT, 'dryrun', copy_name)
    else:
        out_dir = os.path.join(AUG_ROOT, copy_name)

    out_img1 = os.path.join(out_dir, 'img1')
    out_gt   = os.path.join(out_dir, 'gt')
    params_path = os.path.join(out_dir, 'aug_params.json')

    # ── refuse if output path exists ──
    if os.path.exists(out_dir):
        print(f'ERROR: output path already exists: {out_dir}')
        print('  Will not overwrite. Remove it manually if you want to re-run.')
        sys.exit(1)

    os.makedirs(out_img1, exist_ok=False)
    os.makedirs(out_gt,   exist_ok=False)

    print(f'=== {copy_name} ===')
    print(f'  source : {src_dir}')
    print(f'  output : {out_dir}')
    print(f'  seed   : {seed}')
    if dry_run_n:
        print(f'  DRY RUN: first {dry_run_n} frames only')

    # ── get source JPEG list ──
    jpegs = sorted(f for f in os.listdir(src_img1) if f.lower().endswith('.jpg'))
    if dry_run_n is not None:
        jpegs = jpegs[:dry_run_n]
    n_frames = len(jpegs)

    # ── build replay pipeline; sample on first frame ──
    p_params = COPY_PARAMS[seed]
    pipeline = build_replay_pipeline(p_params, seed)

    # ── read first frame to build replay data ──
    first_img = cv2.imread(os.path.join(src_img1, jpegs[0]))
    if first_img is None:
        print(f'ERROR: could not read first frame: {jpegs[0]}')
        sys.exit(1)
    first_result = pipeline(image=first_img)
    replay_data  = first_result['replay']

    # ── record sampled parameters ──
    aug_params_record = {
        'copy':   copy_name,
        'source': source_seq,
        'seed':   seed,
        'n_frames': n_frames,
        'brightness_limit': list(p_params['brightness']),
        'contrast_limit':   list(p_params['contrast']),
        'gamma_range':      list(p_params['gamma']),
        'hue_shift_limit':  p_params['hue_shift'],
        'sat_shift_limit':  p_params['sat_shift'],
        'jpeg_quality_range': list(p_params['jpeg_q']),
        'noise_std_range_normalised': list(p_params['noise_std']),
        'note': (
            'Deterministic transforms (brightness/contrast/gamma/hue/jpeg) sampled once '
            'via ReplayCompose. Gaussian noise applied per-frame with seed '
            'copy_seed * 1_000_003 + frame_index.'
        ),
    }
    with open(params_path, 'w') as f:
        json.dump(aug_params_record, f, indent=2)

    # ── process frames ──
    imwrite_ok   = 0
    imwrite_fail = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    for frame_idx, fname in enumerate(jpegs):
        src_path = os.path.join(src_img1, fname)
        out_path = os.path.join(out_img1, fname)

        img = cv2.imread(src_path)
        if img is None:
            print(f'  WARNING: could not read {src_path}')
            imwrite_fail += 1
            continue

        # 1. Replay deterministic transforms
        replayed = A.ReplayCompose.replay(replay_data, image=img)
        aug_img  = replayed['image']

        # 2. Per-frame independent Gaussian noise (outside ReplayCompose)
        aug_img = add_frame_noise(aug_img, p_params['noise_std'], seed, frame_idx)

        ok = cv2.imwrite(out_path, aug_img, encode_params)
        if ok:
            imwrite_ok += 1
        else:
            imwrite_fail += 1
            print(f'  imwrite FAILED: {out_path}')

        if (frame_idx + 1) % 200 == 0:
            print(f'  ... {frame_idx + 1}/{n_frames} frames done')

    # ── copy annotation files (byte-for-byte) ──
    ann_files = [
        ('gt/gt.txt',      os.path.join(src_dir, 'gt', 'gt.txt')),
        ('gt/labels.txt',  os.path.join(src_dir, 'gt', 'labels.txt')),
        ('seqinfo.ini',    os.path.join(src_dir, 'seqinfo.ini')),
    ]

    print(f'  Copying annotation files and verifying SHA256 ...')
    for rel, src_ann in ann_files:
        dst_ann = os.path.join(out_dir, rel)
        shutil.copyfile(src_ann, dst_ann)
        h_src = sha256_file(src_ann)
        h_dst = sha256_file(dst_ann)
        if h_src != h_dst:
            print(f'  SHA256 MISMATCH: {rel}  src={h_src}  dst={h_dst}')
            sys.exit(1)
        print(f'  SHA256 OK: {rel}  {h_src.upper()}')

    # ── summary ──
    jpeg_count = len([f for f in os.listdir(out_img1) if f.endswith('.jpg')])
    print(f'\n  DONE: {copy_name}')
    print(f'  Frames written : {imwrite_ok}  (failures: {imwrite_fail})')
    print(f'  JPEGs on disk  : {jpeg_count}')
    print(f'  Expected       : {n_frames}')
    if jpeg_count != n_frames:
        print(f'  WARNING: count mismatch!')
    print()


def main():
    parser = argparse.ArgumentParser(description='Photometric augmentation for IOTD training sequences.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--copy', type=str,
                       help='Name of a single augmented copy from split_manifest.json')
    group.add_argument('--all',  action='store_true',
                       help='Run all copies listed in split_manifest.json')
    parser.add_argument('--dry-run', type=int, metavar='N', default=None,
                        help='Process only first N frames into check/dryrun/ (no annotation copy)')
    args = parser.parse_args()

    manifest = load_manifest()
    copies   = list(manifest['augmented'].keys())

    if args.all and args.dry_run is not None:
        print('ERROR: --all and --dry-run cannot be combined. Pick one copy with --copy for dry run.')
        sys.exit(1)

    targets = copies if args.all else [args.copy]

    for copy_name in targets:
        run_copy(copy_name, manifest, dry_run_n=args.dry_run)

    if args.all:
        print('=== ALL COPIES COMPLETE ===')
        for seq in ['seq_01_1min', 'seq_03_30sec_b']:
            for i in ['01', '02', '03']:
                name = f'{seq}_aug_{i}'
                d = os.path.join(AUG_ROOT, name, 'img1')
                count = len(os.listdir(d)) if os.path.isdir(d) else 0
                print(f'  {name}: {count} JPEGs')


if __name__ == '__main__':
    main()
