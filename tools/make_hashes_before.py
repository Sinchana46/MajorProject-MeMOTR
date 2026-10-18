"""
tools/make_hashes_before.py
---------------------------
Compute SHA256 of every protected file in the three real sequences and save
to data/IOTD/check/hashes_before.json.

Protected files per sequence:
  img1/*.jpg       (all extracted frames)
  gt/gt.txt
  gt/labels.txt
  seqinfo.ini
"""

import os, json, hashlib, datetime

IOTD_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
CHECK_DIR  = os.path.join(IOTD_ROOT, 'check')
OUT_PATH   = os.path.join(CHECK_DIR, 'hashes_before.json')
SEQS       = ['seq_01_1min', 'seq_02_30sec_a', 'seq_03_30sec_b']

os.makedirs(CHECK_DIR, exist_ok=True)

if os.path.exists(OUT_PATH):
    print(f'ERROR: {OUT_PATH} already exists. Will not overwrite.')
    raise SystemExit(1)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    'created': datetime.datetime.utcnow().isoformat() + 'Z',
    'note':    'SHA256 of all protected files before any augmentation. Do not delete.',
    'sequences': {}
}

total_files = 0
for seq in SEQS:
    seq_dir  = os.path.join(IOTD_ROOT, seq)
    img1_dir = os.path.join(seq_dir, 'img1')
    entry = {'annotation_files': {}, 'img1': {}}

    # annotation files
    for rel in ['gt/gt.txt', 'gt/labels.txt', 'seqinfo.ini']:
        full = os.path.join(seq_dir, rel)
        if os.path.isfile(full):
            entry['annotation_files'][rel] = sha256_file(full)
        else:
            entry['annotation_files'][rel] = 'MISSING'

    # JPEGs
    jpegs = sorted(f for f in os.listdir(img1_dir) if f.lower().endswith('.jpg'))
    print(f'{seq}: hashing {len(jpegs)} JPEGs ...', flush=True)
    for i, fname in enumerate(jpegs):
        full = os.path.join(img1_dir, fname)
        entry['img1'][fname] = sha256_file(full)
        if (i + 1) % 500 == 0:
            print(f'  ... {i+1}/{len(jpegs)}', flush=True)

    jpeg_count = len(entry['img1'])
    ann_count  = len(entry['annotation_files'])
    seq_total  = jpeg_count + ann_count
    total_files += seq_total

    manifest['sequences'][seq] = entry

    print(f'  Done: {ann_count} annotation files + {jpeg_count} JPEGs = {seq_total} files')
    for rel, h in entry['annotation_files'].items():
        print(f'    {rel}: {h.upper()}')

# Save
with open(OUT_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'\n=== INTEGRITY MANIFEST WRITTEN ===')
print(f'Path       : {OUT_PATH}')
print(f'Size       : {os.path.getsize(OUT_PATH):,} bytes')
print(f'Total files: {total_files}')
print(f'Breakdown  :')
for seq in SEQS:
    n_ann  = len(manifest["sequences"][seq]["annotation_files"])
    n_jpg  = len(manifest["sequences"][seq]["img1"])
    print(f'  {seq}: {n_ann} annotation + {n_jpg} JPEGs = {n_ann + n_jpg}')
