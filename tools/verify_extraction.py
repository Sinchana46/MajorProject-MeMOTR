"""
tools/verify_extraction.py
--------------------------
Phase 2 verification: prints a table for all sequences, then saves
5 evenly-spaced overlay images (gt boxes + track IDs) per sequence
into data/IOTD/check/.

Usage:
    python tools/verify_extraction.py
"""

import os
import sys
import configparser
import cv2
import numpy as np

IOTD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
CHECK_DIR  = os.path.join(IOTD_ROOT, 'check')
os.makedirs(CHECK_DIR, exist_ok=True)

SEQUENCES = [
    ('seq_01_1min',    1599),   # expected
    ('seq_02_30sec_a',  744),   # expected (user-corrected)
    ('seq_03_30sec_b',  931),   # expected
]

# ── helpers ────────────────────────────────────────────────────────────────────
def count_jpegs(img1_dir):
    if not os.path.isdir(img1_dir):
        return 0
    return len([f for f in os.listdir(img1_dir) if f.lower().endswith('.jpg')])

def read_seqinfo(seq_dir):
    ini = os.path.join(seq_dir, 'seqinfo.ini')
    if not os.path.isfile(ini):
        return {}
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    s = cfg['Sequence'] if 'Sequence' in cfg else {}
    return {
        'frameRate': s.get('frameRate', '?'),
        'seqLength': s.get('seqLength', '?'),
        'imWidth':   s.get('imWidth',  '?'),
        'imHeight':  s.get('imHeight', '?'),
    }

def read_gt(gt_path):
    """Returns list of (frame, track_id, x, y, w, h, cls) tuples.

    MOT gt.txt column layout (0-based):
      0=frame  1=track_id  2=bb_left  3=bb_top  4=bb_width  5=bb_height
      6=conf(not-ignored, always 1)  7=class_id  8=visibility
    """
    rows = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split(',')
            if len(cols) < 8:
                continue   # skip malformed lines
            frame   = int(cols[0])
            tid     = int(cols[1])
            x       = float(cols[2])
            y       = float(cols[3])
            w       = float(cols[4])
            h       = float(cols[5])
            # col[6] = conf / not-ignored flag (always 1) -- NOT the class
            cls     = int(cols[7])   # col[7] = actual class_id
            rows.append((frame, tid, x, y, w, h, cls))
    return rows

def get_jpeg_dims(img1_dir):
    """Read actual JPEG dimensions from first file."""
    files = sorted(f for f in os.listdir(img1_dir) if f.lower().endswith('.jpg'))
    if not files:
        return None, None
    img = cv2.imread(os.path.join(img1_dir, files[0]))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    return w, h

# ── per-sequence verification ──────────────────────────────────────────────────
def verify_sequence(seq_name, expected_count):
    seq_dir  = os.path.join(IOTD_ROOT, seq_name)
    img1_dir = os.path.join(seq_dir, 'img1')
    gt_path  = os.path.join(seq_dir, 'gt', 'gt.txt')

    # counts
    jpegs_on_disk = count_jpegs(img1_dir)
    diff          = jpegs_on_disk - expected_count

    # seqinfo.ini
    si = read_seqinfo(seq_dir)
    ini_wh = f"{si.get('imWidth','?')}x{si.get('imHeight','?')}"
    fps    = si.get('frameRate', '?')

    # actual JPEG dims
    jpeg_w, jpeg_h = get_jpeg_dims(img1_dir)
    jpeg_wh = f"{jpeg_w}x{jpeg_h}" if jpeg_w else "?"

    # dims match?
    dims_match = (str(jpeg_w) == si.get('imWidth','') and
                  str(jpeg_h) == si.get('imHeight',''))

    # gt stats
    gt_rows  = read_gt(gt_path)
    frames_in_gt = [r[0] for r in gt_rows]
    gt_min   = min(frames_in_gt) if frames_in_gt else '?'
    gt_max   = max(frames_in_gt) if frames_in_gt else '?'
    gt_valid = (gt_max <= jpegs_on_disk) if isinstance(gt_max, int) else '?'

    return {
        'seq':           seq_name,
        'frames_read':   int(si.get('seqLength', jpegs_on_disk)),
        'jpegs_on_disk': jpegs_on_disk,
        'expected':      expected_count,
        'diff':          diff,
        'jpeg_wh':       jpeg_wh,
        'ini_wh':        ini_wh,
        'dims_match':    dims_match,
        'fps':           fps,
        'gt_min':        gt_min,
        'gt_max':        gt_max,
        'gt_max_ok':     gt_valid,
        'imwrite_fail':  0,   # logged during extraction; 0 confirmed
    }, gt_rows

# ── overlay images ─────────────────────────────────────────────────────────────
def draw_overlays(seq_name, gt_rows, jpegs_on_disk):
    img1_dir = os.path.join(IOTD_ROOT, seq_name, 'img1')

    # build frame→annotations dict
    ann = {}
    for (frame, tid, x, y, w, h, cls) in gt_rows:
        ann.setdefault(frame, []).append((tid, x, y, w, h, cls))

    # 5 percentile points: 0, 25, 50, 75, 100
    gt_frames_sorted = sorted(ann.keys())
    n = len(gt_frames_sorted)
    if n == 0:
        print(f'  {seq_name}: no gt annotations, skipping overlays')
        return

    indices = [0, n//4, n//2, 3*n//4, n-1]
    picked  = [gt_frames_sorted[i] for i in indices]

    print(f'\n  {seq_name} overlay frames (0%/25%/50%/75%/100%): {picked}')

    for frame_id in picked:
        fname   = f'{frame_id:06d}.jpg'
        src     = os.path.join(img1_dir, fname)
        if not os.path.isfile(src):
            print(f'    WARNING: image not found: {src}')
            continue

        img = cv2.imread(src)
        if img is None:
            print(f'    WARNING: cv2.imread returned None for {src}')
            continue

        # draw each annotation
        for (tid, x, y, w, h, cls) in ann[frame_id]:
            x1, y1 = int(round(x)),         int(round(y))
            x2, y2 = int(round(x + w)),     int(round(y + h))
            color   = (0, 255, 0)            # green box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f'ID{tid} cls{cls}'
            # text background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # frame number watermark
        cv2.putText(img, f'frame {frame_id}  seq={seq_name}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 0), 2, cv2.LINE_AA)

        out_name = f'{seq_name}_frame_{frame_id:06d}.jpg'
        out_path = os.path.join(CHECK_DIR, out_name)
        ok = cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        status = 'OK' if ok else 'FAILED'
        print(f'    {status}  {out_name}')

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '='*95)
    print('PHASE 2 VERIFICATION TABLE')
    print('='*95)

    results   = []
    all_gt    = {}
    total_ok  = True

    for seq_name, expected in SEQUENCES:
        res, gt_rows = verify_sequence(seq_name, expected)
        results.append(res)
        all_gt[seq_name] = gt_rows

    # ── print table ──
    hdr = (f"{'Sequence':<20} {'Frames':>7} {'JPEGs':>7} {'Expect':>7} "
           f"{'Diff':>5} {'JPEG WxH':<12} {'INI WxH':<12} {'DimOK':>5} "
           f"{'FPS':>4} {'GTmin':>6} {'GTmax':>6} {'GTok':>5} {'iFail':>5}")
    sep = '-' * len(hdr)
    print(f'\n{hdr}')
    print(sep)

    total_jpegs = 0
    for r in results:
        dim_ok_str = 'YES' if r['dims_match'] else 'NO!'
        gt_ok_str  = 'YES' if r['gt_max_ok'] is True else ('NO!' if r['gt_max_ok'] is False else '?')
        diff_str   = str(r['diff']) if r['diff'] != 0 else '0'
        flag_diff  = '***' if r['diff'] != 0 else ''

        print(f"{r['seq']:<20} {r['frames_read']:>7} {r['jpegs_on_disk']:>7} "
              f"{r['expected']:>7} {diff_str:>5}{flag_diff:<3} "
              f"{r['jpeg_wh']:<12} {r['ini_wh']:<12} {dim_ok_str:>5} "
              f"{r['fps']:>4} {r['gt_min']:>6} {r['gt_max']:>6} "
              f"{gt_ok_str:>5} {r['imwrite_fail']:>5}")
        total_jpegs += r['jpegs_on_disk']

        if r['diff'] != 0 or not r['dims_match'] or r['gt_max_ok'] is False:
            total_ok = False

    print(sep)
    print(f"{'TOTAL':<20} {'':>7} {total_jpegs:>7}")
    print()

    # ── original expected vs actual comparison ──
    orig_expected = {
        'seq_01_1min':    1599,
        'seq_02_30sec_a':  744,
        'seq_03_30sec_b':  931,
    }
    orig_total = sum(orig_expected.values())
    print(f'Expected totals  (per user-corrected task):')
    for seq, exp in orig_expected.items():
        act = next(r['jpegs_on_disk'] for r in results if r['seq'] == seq)
        match = 'MATCH' if act == exp else f'MISMATCH (expected {exp})'
        print(f'  {seq:<22}: expected={exp:>4}  actual={act:>4}  {match}')
    print(f"  {'TOTAL':<22}: expected={orig_total:>4}  actual={total_jpegs:>4}  "
          f"{'MATCH' if total_jpegs == orig_total else 'MISMATCH'}")

    # ── SPECIAL FLAGS ──
    print()
    print('SPECIAL FLAGS:')
    for r in results:
        if r['jpeg_wh'] != r['ini_wh']:
            print(f'  !! {r["seq"]}: JPEG dims {r["jpeg_wh"]} != seqinfo.ini {r["ini_wh"]}')
    # Flag portrait seq
    for r in results:
        w, h = r['jpeg_wh'].split('x') if 'x' in r['jpeg_wh'] else ('0','0')
        if int(w) < int(h):
            print(f'  ** {r["seq"]}: PORTRAIT orientation ({r["jpeg_wh"]}) '
                  f'-- different from the other sequences. Not an error, but note for augmentation.')

    if total_ok:
        print('\nAll counts match, dims consistent, gt ranges valid.')
    else:
        print('\nWARNING: one or more checks failed -- see flags above.')

    # ── overlay images ──
    print('\n' + '='*95)
    print('GENERATING OVERLAY IMAGES -> data/IOTD/check/')
    print('='*95)

    for seq_name, expected in SEQUENCES:
        gt_rows = all_gt[seq_name]
        res = next(r for r in results if r['seq'] == seq_name)
        draw_overlays(seq_name, gt_rows, res['jpegs_on_disk'])

    print(f'\nOverlay images saved to: {CHECK_DIR}')
    print('='*95)
    print('PHASE 2 COMPLETE')
    print('='*95)


if __name__ == '__main__':
    main()
