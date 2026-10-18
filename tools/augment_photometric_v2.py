"""
tools/augment_photometric_v2.py
--------------------------------
Part B: Photometric augmentation v2 — pure OpenCV/NumPy, no albumentations.
Multiplicative exposure gain (not additive offset) so night scenes are preserved.

Pipeline per frame (in order):
  1. Exposure gain:  clip(gain * x, 0, 255)           -- multiplicative
  2. Gamma LUT:      255 * (x/255)**gamma              -- 256-entry uint8 LUT
  3. Hue/sat shift:  OpenCV HSV, hue 0..179, val untouched
  4. JPEG round-trip: cv2.imencode / cv2.imdecode at sampled quality
  5. Gaussian noise: np.random.default_rng(seed*1_000_003 + frame_idx), different per frame
  6. Save at JPEG quality 98, names 000001.jpg onward

Per-copy parameters (from plan):
  aug_01: seed=1001, gain=0.85-0.95, gamma=1.02-1.08, hue±3/sat±10,  q=75-90, noise=2-4
  aug_02: seed=1002, gain=1.10-1.30, gamma=0.92-0.98, hue±5/sat±15,  q=65-85, noise=3-6
  aug_03: seed=1003, gain=0.95-1.10, gamma=0.95-1.05, hue±5/sat±20,  q=60-80, noise=4-8

Safety:
  - Refuses val/test sources (reads split_manifest.json)
  - Refuses to write into any existing path
  - No resizing (seq_03 is 1080x2336 portrait)
  - Annotation files copied byte-for-byte, SHA256 verified
  - --dry-run N: outputs to check/dryrun_v2/<copy_name>/, first N frames only

Usage:
  python tools/augment_photometric_v2.py --copy seq_01_1min_aug_01
  python tools/augment_photometric_v2.py --dry-run 20 --copy seq_03_30sec_b_aug_01
  python tools/augment_photometric_v2.py --all
  python tools/augment_photometric_v2.py --b1-check   # disk precondition only
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import numpy as np
import cv2

# ── paths ──────────────────────────────────────────────────────────────────────
IOTD_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
MANIFEST_PATH = os.path.join(IOTD_ROOT, 'split_manifest.json')
AUG_ROOT      = os.path.join(IOTD_ROOT, 'augmented_train_v2')
CHECK_ROOT    = os.path.join(IOTD_ROOT, 'check')
JPEG_SAVE_Q   = 98

# ── per-copy parameters (from plan B2 table) ───────────────────────────────────
COPY_PARAMS = {
    1001: dict(gain=(0.85, 0.95), gamma=(1.02, 1.08), hue=3,  sat=10, jpeg_q=(75, 90), noise_sigma=(2.0, 4.0)),
    1002: dict(gain=(1.10, 1.30), gamma=(0.92, 0.98), hue=5,  sat=15, jpeg_q=(65, 85), noise_sigma=(3.0, 6.0)),
    1003: dict(gain=(0.95, 1.10), gamma=(0.95, 1.05), hue=5,  sat=20, jpeg_q=(60, 80), noise_sigma=(4.0, 8.0)),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_gamma_lut(gamma: float) -> np.ndarray:
    """Build a 256-entry uint8 LUT: 255 * (x/255)**gamma."""
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(255.0 * (x ** gamma), 0, 255).astype(np.uint8)
    return lut


def apply_frame(img: np.ndarray, gain: float, gamma_lut: np.ndarray,
                hue_shift: int, sat_shift: int, jpeg_quality: int,
                noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply the full augmentation pipeline to one BGR uint8 frame.
    rng is a per-frame Generator (different seed per frame index).
    """
    # 1. Exposure gain (multiplicative)
    out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    # 2. Gamma via LUT
    out = cv2.LUT(out, gamma_lut)

    # 3. Hue / saturation shift in HSV (OpenCV hue: 0..179)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
    # value (channel 2) is untouched
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. JPEG round trip
    _, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 5. Per-frame Gaussian noise (different per frame — rng already seeded per frame)
    noise = rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def b1_check():
    """Disk precondition check."""
    aug_v1 = os.path.join(IOTD_ROOT, 'augmented_train')
    total_v1 = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(aug_v1) for f in files
    )
    free = shutil.disk_usage('D:/').free
    required = 1.5 * total_v1
    print('=== B1: Disk Precondition Check ===')
    print('  v1 size on disk  : {:.2f} GB'.format(total_v1 / (1024**3)))
    print('  Required (1.5x)  : {:.2f} GB'.format(required / (1024**3)))
    print('  Free on D:       : {:.2f} GB'.format(free / (1024**3)))
    if free < required:
        print('  STOP: insufficient disk space')
        sys.exit(1)
    v2_path = AUG_ROOT
    if os.path.exists(v2_path):
        print('  STOP: augmented_train_v2/ already exists: ' + v2_path)
        sys.exit(1)
    print('  Disk check: OK')
    print('  augmented_train_v2/ does not exist: OK')


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def run_copy(copy_name: str, manifest: dict, dry_run_n: int = None):
    """Augment one copy (or dry-run first N frames)."""

    if copy_name not in manifest['augmented']:
        print('ERROR: "' + copy_name + '" not in split_manifest.json augmented section')
        sys.exit(1)

    cfg        = manifest['augmented'][copy_name]
    source_seq = cfg['source']
    seed       = cfg['seed']

    # Safety: refuse val / test sources
    if source_seq in manifest.get('val', []):
        print('ERROR: ' + source_seq + ' is listed as VAL. Refusing.')
        sys.exit(1)
    if source_seq in manifest.get('test', []):
        print('ERROR: ' + source_seq + ' is listed as TEST. Refusing.')
        sys.exit(1)

    src_dir  = os.path.join(IOTD_ROOT, source_seq)
    src_img1 = os.path.join(src_dir, 'img1')

    if dry_run_n is not None:
        out_dir = os.path.join(CHECK_ROOT, 'dryrun_v2', copy_name)
    else:
        out_dir = os.path.join(AUG_ROOT, copy_name)

    out_img1 = os.path.join(out_dir, 'img1')
    out_gt   = os.path.join(out_dir, 'gt')

    # Refuse if output path already exists
    if os.path.exists(out_dir):
        print('ERROR: output path already exists: ' + out_dir)
        print('       Will not overwrite. Remove manually if needed.')
        sys.exit(1)

    os.makedirs(out_img1, exist_ok=False)
    os.makedirs(out_gt,   exist_ok=False)

    print('=== ' + copy_name + ' ===')
    print('  source  : ' + src_dir)
    print('  output  : ' + out_dir)
    print('  seed    : ' + str(seed))
    if dry_run_n:
        print('  DRY RUN: first ' + str(dry_run_n) + ' frames only')

    # Sample parameters once per copy using copy seed
    rng_copy = np.random.default_rng(seed)
    params   = COPY_PARAMS[seed]
    gain      = float(rng_copy.uniform(*params['gain']))
    gamma_val = float(rng_copy.uniform(*params['gamma']))
    hue_shift = int(rng_copy.integers(-params['hue'], params['hue'] + 1))
    sat_shift = int(rng_copy.integers(-params['sat'], params['sat'] + 1))
    jpeg_q    = int(rng_copy.integers(*params['jpeg_q']))
    noise_sig = float(rng_copy.uniform(*params['noise_sigma']))

    gamma_lut = build_gamma_lut(gamma_val)

    aug_params_record = {
        'copy':        copy_name,
        'source':      source_seq,
        'seed':        seed,
        'gain':        round(gain, 6),
        'gamma':       round(gamma_val, 6),
        'hue_shift':   hue_shift,
        'sat_shift':   sat_shift,
        'jpeg_quality':jpeg_q,
        'noise_sigma': round(noise_sig, 6),
        'note': (
            'All params sampled once per copy from np.random.default_rng(seed). '
            'Noise: np.random.default_rng(seed*1_000_003 + frame_idx) per frame.'
        ),
    }

    params_path = os.path.join(out_dir, 'aug_params.json')
    with open(params_path, 'w') as f:
        json.dump(aug_params_record, f, indent=2)

    print('  Sampled: gain={:.4f} gamma={:.4f} hue={:+d} sat={:+d} '
          'jpeg_q={} noise_sigma={:.2f}'.format(
              gain, gamma_val, hue_shift, sat_shift, jpeg_q, noise_sig))

    # Get sorted JPEG list
    jpegs = sorted(f for f in os.listdir(src_img1) if f.lower().endswith('.jpg'))
    if dry_run_n is not None:
        jpegs = jpegs[:dry_run_n]
    n_frames = len(jpegs)

    ok_count   = 0
    fail_count = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_SAVE_Q]

    for frame_idx, fname in enumerate(jpegs):
        src_path = os.path.join(src_img1, fname)
        out_path = os.path.join(out_img1, fname)

        img = cv2.imread(src_path)
        if img is None:
            print('  WARNING: could not read ' + src_path)
            fail_count += 1
            continue

        # Per-frame RNG (independent noise)
        frame_rng = np.random.default_rng(seed * 1_000_003 + frame_idx)

        aug = apply_frame(img, gain, gamma_lut, hue_shift, sat_shift,
                          jpeg_q, noise_sig, frame_rng)

        ok = cv2.imwrite(out_path, aug, encode_params)
        if ok:
            ok_count += 1
        else:
            print('  imwrite FAILED: ' + out_path)
            fail_count += 1

        if (frame_idx + 1) % 200 == 0:
            print('  ... {}/{} frames'.format(frame_idx + 1, n_frames))

    # Annotation files — byte-for-byte copy + SHA256 verify
    print('  Copying annotations and verifying SHA256 ...')
    for rel in ['gt/gt.txt', 'gt/labels.txt', 'seqinfo.ini']:
        src_ann = os.path.join(src_dir, *rel.split('/'))
        dst_ann = os.path.join(out_dir, rel)
        shutil.copyfile(src_ann, dst_ann)
        h_src = sha256_file(src_ann).upper()
        h_dst = sha256_file(dst_ann).upper()
        if h_src != h_dst:
            print('  SHA256 MISMATCH: ' + rel + ' src=' + h_src + ' dst=' + h_dst)
            sys.exit(1)
        print('  SHA256 OK: ' + rel + '  ' + h_src)

    jpeg_on_disk = len([f for f in os.listdir(out_img1) if f.endswith('.jpg')])
    print()
    print('  DONE: ' + copy_name)
    print('  Frames written : {}  (failures: {})'.format(ok_count, fail_count))
    print('  JPEGs on disk  : {}  expected: {}'.format(jpeg_on_disk, n_frames))
    if jpeg_on_disk != n_frames or fail_count > 0:
        print('  WARNING: count or failure mismatch!')
    print()


def main():
    parser = argparse.ArgumentParser(description='IOTD photometric augmentation v2 (multiplicative gain, no albumentations)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--copy',     type=str, help='Name of a single copy from split_manifest.json')
    group.add_argument('--all',      action='store_true', help='Run all 6 copies')
    group.add_argument('--b1-check', action='store_true', help='Disk precondition check only')
    parser.add_argument('--dry-run', type=int, metavar='N', default=None,
                        help='Process first N frames into check/dryrun_v2/ (per copy, no full run)')
    args = parser.parse_args()

    if args.b1_check:
        b1_check()
        return

    if args.all and args.dry_run is not None:
        print('ERROR: --all and --dry-run cannot be combined. Use --copy for dry runs.')
        sys.exit(1)

    manifest = load_manifest()
    all_copies = list(manifest['augmented'].keys())
    targets    = all_copies if args.all else [args.copy]

    for copy_name in targets:
        run_copy(copy_name, manifest, dry_run_n=args.dry_run)

    if args.all and args.dry_run is None:
        print('=== ALL COPIES COMPLETE ===')
        for name in all_copies:
            d = os.path.join(AUG_ROOT, name, 'img1')
            count = len(os.listdir(d)) if os.path.isdir(d) else 0
            print('  {}: {} JPEGs'.format(name, count))


if __name__ == '__main__':
    main()
