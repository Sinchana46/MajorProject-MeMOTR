"""
tools/extract_frames.py
-----------------------
Extract frames from a single video into <seq_dir>/img1/ as 6-digit JPEG files.

Rules enforced:
  - cap.isOpened() checked immediately; aborts if False.
  - Sequential cap.read() only; never cap.set(POS_FRAMES).
  - Frame count from actual reads; CAP_PROP_FRAME_COUNT is only advisory.
  - Output: 000001.jpg ... (1-based, JPEG quality 98).
  - Every cv2.imwrite return value is checked.
  - img1/ must not already contain files (stops and reports).
  - seqinfo.ini created only if absent; never overwritten.
  - All events logged to data/IOTD/extraction_log.txt.

Usage:
    python tools/extract_frames.py <seq_name>
E.g.:
    python tools/extract_frames.py seq_01_1min
"""

import sys
import os
import cv2
import configparser
import hashlib
import datetime

# ── constants ──────────────────────────────────────────────────────────────────
IOTD_ROOT  = os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD')
IOTD_ROOT  = os.path.abspath(IOTD_ROOT)
VIDEO_DIR  = os.path.join(IOTD_ROOT, 'videos')
LOG_PATH   = os.path.join(IOTD_ROOT, 'extraction_log.txt')
JPEG_QUALITY = 98

VIDEO_MAP = {
    'seq_01_1min':    'seq_01_1min.mov',
    'seq_02_30sec_a': 'seq_02_30sec_a.mov',
    'seq_03_30sec_b': 'seq_03_30sec_b.mp4',
}

# ── logging ────────────────────────────────────────────────────────────────────
os.makedirs(IOTD_ROOT, exist_ok=True)
_log_file = open(LOG_PATH, 'a', encoding='utf-8')

def log(msg, also_print=True):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    _log_file.write(line + '\n')
    _log_file.flush()
    if also_print:
        print(line)

def abort(msg):
    log(f'ABORT: {msg}')
    _log_file.close()
    sys.exit(1)

# ── main ───────────────────────────────────────────────────────────────────────
def extract(seq_name):
    log(f'===== START extraction: {seq_name} =====')

    # -- locate video --
    if seq_name not in VIDEO_MAP:
        abort(f'Unknown sequence "{seq_name}". Known: {list(VIDEO_MAP.keys())}')
    video_file = os.path.join(VIDEO_DIR, VIDEO_MAP[seq_name])
    if not os.path.isfile(video_file):
        abort(f'Video file not found: {video_file}')
    log(f'Video file : {video_file}')
    log(f'Video size : {os.path.getsize(video_file):,} bytes')

    # -- locate sequence dir --
    seq_dir  = os.path.join(IOTD_ROOT, seq_name)
    img1_dir = os.path.join(seq_dir, 'img1')
    gt_path  = os.path.join(seq_dir, 'gt', 'gt.txt')
    seq_ini  = os.path.join(seq_dir, 'seqinfo.ini')

    if not os.path.isfile(gt_path):
        abort(f'gt.txt missing: {gt_path}')

    # -- guard: img1 must not have existing files --
    if os.path.isdir(img1_dir):
        existing = [f for f in os.listdir(img1_dir)
                    if os.path.isfile(os.path.join(img1_dir, f))]
        if existing:
            abort(
                f'img1/ already contains {len(existing)} file(s). '
                f'First 5: {existing[:5]}. '
                f'Will NOT overwrite. Remove them manually if you want to re-extract.'
            )
    os.makedirs(img1_dir, exist_ok=True)

    # -- SHA256 of gt.txt before we touch anything --
    with open(gt_path, 'rb') as f:
        sha_before = hashlib.sha256(f.read()).hexdigest().upper()
    log(f'gt.txt SHA256 (before) : {sha_before}')

    # -- open video --
    log(f'Opening video with cv2.VideoCapture ...')
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        abort(
            f'cap.isOpened() returned False for: {video_file}\n'
            f'  Check the file is a valid, untruncated video and that '
            f'opencv ({cv2.__version__}) has the required codec.'
        )
    log(f'cap.isOpened() : True')

    # -- advisory properties (not trusted for frame count) --
    fps_prop    = cap.get(cv2.CAP_PROP_FPS)
    width_prop  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_prop = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prop_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f'CAP_PROP_FPS         : {fps_prop}  (advisory)')
    log(f'CAP_PROP_FRAME_WIDTH : {width_prop}')
    log(f'CAP_PROP_FRAME_HEIGHT: {height_prop}')
    log(f'CAP_PROP_FRAME_COUNT : {prop_count}  (advisory, NOT used for actual count)')

    # -- sequential read loop --
    frames_read    = 0
    imwrite_ok     = 0
    imwrite_fail   = 0
    actual_width   = None
    actual_height  = None
    encode_params  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    log(f'Starting sequential cap.read() loop ...')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1

        # Record actual dimensions from first frame
        if frames_read == 1:
            actual_height, actual_width = frame.shape[:2]
            log(f'First frame actual dimensions: {actual_width}x{actual_height}')

        # 6-digit 1-based filename
        filename  = f'{frames_read:06d}.jpg'
        out_path  = os.path.join(img1_dir, filename)

        ok = cv2.imwrite(out_path, frame, encode_params)
        if ok:
            imwrite_ok += 1
        else:
            imwrite_fail += 1
            log(f'  imwrite FAILED for frame {frames_read}: {out_path}')

        # Progress every 200 frames
        if frames_read % 200 == 0:
            log(f'  ... {frames_read} frames read so far')

    cap.release()
    log(f'cap.release() called.')

    # -- summarise --
    jpeg_on_disk = len([f for f in os.listdir(img1_dir)
                        if f.lower().endswith('.jpg')])

    log(f'--- EXTRACTION RESULT: {seq_name} ---')
    log(f'Frames read         : {frames_read}')
    log(f'imwrite OK          : {imwrite_ok}')
    log(f'imwrite FAILURES    : {imwrite_fail}')
    log(f'JPEG files on disk  : {jpeg_on_disk}')
    log(f'Actual resolution   : {actual_width}x{actual_height}')

    # -- seqinfo.ini: create only if absent --
    if os.path.isfile(seq_ini):
        log(f'seqinfo.ini already exists — reading to check differences ...')
        cfg = configparser.ConfigParser()
        cfg.read(seq_ini)
        existing_seq_len = cfg.get('Sequence', 'seqLength', fallback='MISSING')
        existing_fps     = cfg.get('Sequence', 'frameRate', fallback='MISSING')
        existing_w       = cfg.get('Sequence', 'imWidth',   fallback='MISSING')
        existing_h       = cfg.get('Sequence', 'imHeight',  fallback='MISSING')
        diffs = []
        if str(frames_read) != existing_seq_len:
            diffs.append(f'seqLength: ini={existing_seq_len}  actual={frames_read}')
        if str(round(fps_prop)) != existing_fps and existing_fps != 'MISSING':
            diffs.append(f'frameRate: ini={existing_fps}  cap={fps_prop:.2f}')
        if str(actual_width)  != existing_w:
            diffs.append(f'imWidth:   ini={existing_w}  actual={actual_width}')
        if str(actual_height) != existing_h:
            diffs.append(f'imHeight:  ini={existing_h}  actual={actual_height}')
        if diffs:
            log(f'seqinfo.ini DIFFERENCES found (NOT overwritten):')
            for d in diffs:
                log(f'  {d}')
        else:
            log(f'seqinfo.ini: no differences detected.')
    else:
        log(f'seqinfo.ini absent — creating ...')
        cfg = configparser.ConfigParser()
        cfg['Sequence'] = {
            'name':      seq_name,
            'imDir':     'img1',
            'frameRate': str(round(fps_prop)),
            'seqLength': str(frames_read),
            'imWidth':   str(actual_width),
            'imHeight':  str(actual_height),
            'imExt':     '.jpg',
        }
        with open(seq_ini, 'w') as f:
            cfg.write(f)
        log(f'seqinfo.ini written: frameRate={round(fps_prop)}, '
            f'seqLength={frames_read}, '
            f'{actual_width}x{actual_height}')

    # -- SHA256 of gt.txt after (must be unchanged) --
    with open(gt_path, 'rb') as f:
        sha_after = hashlib.sha256(f.read()).hexdigest().upper()
    log(f'gt.txt SHA256 (after) : {sha_after}')
    if sha_before != sha_after:
        abort(f'gt.txt SHA256 CHANGED during extraction — this should never happen!')
    else:
        log(f'gt.txt SHA256 MATCH: unchanged')

    log(f'===== END extraction: {seq_name}  frames={frames_read} =====')
    print(f'\nFRAME COUNT FOR {seq_name}: {frames_read}')
    return frames_read


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python tools/extract_frames.py <seq_name>')
        print('  seq_name must be one of:', list(VIDEO_MAP.keys()))
        sys.exit(1)

    count = extract(sys.argv[1])
    _log_file.close()
