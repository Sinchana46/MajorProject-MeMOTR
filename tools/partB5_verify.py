"""
tools/partB5_verify.py
-----------------------
B5 verification checks for augmented_train_v2/:
  1. File count per copy
  2. Header dims (PIL) + full decode (cv2) of every file
  3. SHA256 of 3 annotation files per copy vs source (18 total)
  4. A3 luma table on 20 evenly-spaced frames per copy, pass/fail per copy
  5. Re-hash real sequences vs hashes_before.json (changed/missing/added must be 0)
  6. 18 overlays -> check/aug_overlays_v2/
  7. Frame count totals
"""
import os, json, hashlib, sys
import numpy as np
import cv2
from PIL import Image

IOTD   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
AUG    = os.path.join(IOTD, 'augmented_train_v2')
CHECK  = os.path.join(IOTD, 'check')
SEQS_REAL = ['seq_01_1min', 'seq_02_30sec_a', 'seq_03_30sec_b']
SOURCES   = {'seq_01_1min': 1599, 'seq_03_30sec_b': 931}
COPIES    = [
    ('seq_01_1min_aug_01',    'seq_01_1min',    1001),
    ('seq_01_1min_aug_02',    'seq_01_1min',    1002),
    ('seq_01_1min_aug_03',    'seq_01_1min',    1003),
    ('seq_03_30sec_b_aug_01', 'seq_03_30sec_b', 1001),
    ('seq_03_30sec_b_aug_02', 'seq_03_30sec_b', 1002),
    ('seq_03_30sec_b_aug_03', 'seq_03_30sec_b', 1003),
]
LABELS = {1:'car', 2:'two_whl', 3:'auto', 4:'bus', 5:'truck'}

def sha256f(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest().upper()

def parse_gt(path):
    ann = {}
    with open(path) as f:
        for line in f:
            c = line.strip().split(',')
            if len(c) < 8: continue
            ann.setdefault(int(c[0]),[]).append(
                (int(c[1]),float(c[2]),float(c[3]),float(c[4]),float(c[5]),int(c[7])))
    return ann

all_ok = True

# ── CHECK 1: File counts ───────────────────────────────────────────────────────
print('=== CHECK 1: FILE COUNT ===')
for name, src, seed in COPIES:
    d = os.path.join(AUG, name, 'img1')
    n = len([f for f in os.listdir(d) if f.endswith('.jpg')])
    exp = SOURCES[src]
    flag = 'OK' if n == exp else 'MISMATCH'
    if flag != 'OK': all_ok = False
    print(name + ': ' + str(n) + '  expected=' + str(exp) + '  ' + flag)
print()

# ── CHECK 2: Dims + full decode of every file ─────────────────────────────────
print('=== CHECK 2: DIMENSIONS AND FULL DECODE (every file) ===')
src_dims_cache = {}
total_checked = 0; total_fail = 0
for name, src, seed in COPIES:
    aug_img1 = os.path.join(AUG, name, 'img1')
    src_img1 = os.path.join(IOTD, src, 'img1')
    if src not in src_dims_cache:
        ref = Image.open(os.path.join(src_img1, sorted(os.listdir(src_img1))[0]))
        src_dims_cache[src] = ref.size  # (w, h)
    src_wh = src_dims_cache[src]
    files = sorted(f for f in os.listdir(aug_img1) if f.endswith('.jpg'))
    fail_copy = 0
    for fname in files:
        fpath = os.path.join(aug_img1, fname)
        total_checked += 1
        try:
            pil_wh = Image.open(fpath).size
        except Exception as e:
            print('  PIL FAIL ' + name + '/' + fname + ': ' + str(e))
            fail_copy += 1; total_fail += 1; continue
        if pil_wh != src_wh:
            print('  DIM ' + name + '/' + fname + ': got ' + str(pil_wh) + ' expected ' + str(src_wh))
            fail_copy += 1; total_fail += 1; continue
        arr = cv2.imread(fpath)
        if arr is None or arr.size == 0:
            print('  DECODE FAIL ' + name + '/' + fname)
            fail_copy += 1; total_fail += 1
    print(name + ': checked=' + str(len(files)) + '  failures=' + str(fail_copy))
if total_fail > 0: all_ok = False
print('TOTAL checked=' + str(total_checked) + '  failures=' + str(total_fail))
print('CHECK 2: ' + ('PASS' if total_fail == 0 else 'FAIL'))
print()

# ── CHECK 3: Annotation SHA256 (18 comparisons) ───────────────────────────────
print('=== CHECK 3: ANNOTATION SHA256 (18 comparisons) ===')
src_hashes = {}
for seq in SOURCES:
    src_hashes[seq] = {
        'gt/gt.txt':     sha256f(os.path.join(IOTD, seq, 'gt', 'gt.txt')),
        'gt/labels.txt': sha256f(os.path.join(IOTD, seq, 'gt', 'labels.txt')),
        'seqinfo.ini':   sha256f(os.path.join(IOTD, seq, 'seqinfo.ini')),
    }
ann_fail = 0
for name, src, seed in COPIES:
    base = os.path.join(AUG, name)
    for rel in ['gt/gt.txt','gt/labels.txt','seqinfo.ini']:
        h = sha256f(os.path.join(base, *rel.split('/')))
        exp = src_hashes[src][rel]
        flag = 'OK' if h == exp else '*** MISMATCH ***'
        if flag != 'OK': ann_fail += 1; all_ok = False
        print(name + '/' + rel + ': ' + flag)
print('CHECK 3: ' + ('PASS (0 mismatches)' if ann_fail == 0 else 'FAIL (' + str(ann_fail) + ' mismatches)'))
print()

# ── CHECK 4: A3 luma table (20 evenly-spaced frames) ─────────────────────────
print('=== CHECK 4: A3 LUMA TABLE (20 evenly-spaced frames) ===')
print('{:<25} {:>12} {:>12} {:>6} {:>18} {:>18} {:>5}'.format(
    'Copy','Src mean luma','Aug mean luma','Ratio','%pix==0 src/aug','%pix>=250 src/aug','PASS?'))
print('-'*105)
a3_all_pass = True
for name, src, seed in COPIES:
    src_img1 = os.path.join(IOTD, src, 'img1')
    aug_img1 = os.path.join(AUG, name, 'img1')
    files = sorted(f for f in os.listdir(aug_img1) if f.endswith('.jpg'))
    step = max(1, len(files)//20); sampled = files[::step][:20]
    sl_list,al_list,sp0,ap0,sp250,ap250 = [],[],[],[],[],[]
    for fname in sampled:
        si = cv2.imread(os.path.join(src_img1, fname))
        ai = cv2.imread(os.path.join(aug_img1, fname))
        sg = cv2.cvtColor(si, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ag = cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY).astype(np.float32)
        npix = float(sg.size)
        sl_list.append(np.mean(sg)); al_list.append(np.mean(ag))
        sp0.append(np.sum(sg==0)/npix*100); ap0.append(np.sum(ag==0)/npix*100)
        sp250.append(np.sum(sg>=250)/npix*100); ap250.append(np.sum(ag>=250)/npix*100)
    sl=float(np.mean(sl_list)); al=float(np.mean(al_list))
    ratio = al/sl if sl>0 else float('inf')
    s0=float(np.mean(sp0)); a0=float(np.mean(ap0))
    s250=float(np.mean(sp250)); a250=float(np.mean(ap250))
    passed = (0.6<=ratio<=2.0) and ((a0-s0)<=10.0)
    if not passed: a3_all_pass=False; all_ok=False
    print('{:<25} {:>12.2f} {:>12.2f} {:>6.3f} {:>8.2f}%/{:<8.2f}% {:>8.2f}%/{:<8.2f}%  {}'.format(
        name,sl,al,ratio,s0,a0,s250,a250,'PASS' if passed else 'FAIL'))
print('CHECK 4: ' + ('ALL PASS' if a3_all_pass else 'FAILURES'))
print()

# ── CHECK 5: Re-hash real sequences vs hashes_before.json ────────────────────
print('=== CHECK 5: REAL SEQUENCES VS hashes_before.json ===')
with open(os.path.join(CHECK, 'hashes_before.json')) as f:
    before = json.load(f)
changed=0; missing=0; added=0
for seq in SEQS_REAL:
    if seq not in before['sequences']: continue
    seq_dir = os.path.join(IOTD, seq)
    img1_dir = os.path.join(seq_dir, 'img1')
    b_ann = before['sequences'][seq]['annotation_files']
    b_img = before['sequences'][seq]['img1']
    for rel, bh in b_ann.items():
        ah = sha256f(os.path.join(seq_dir, *rel.split('/')))
        if ah.lower() != bh.lower():
            print('CHANGED: ' + seq + '/' + rel); changed+=1
    cur_jpegs = {f: sha256f(os.path.join(img1_dir,f))
                 for f in os.listdir(img1_dir) if f.endswith('.jpg')}
    for fname, bh in b_img.items():
        if fname not in cur_jpegs:
            print('MISSING: ' + seq + '/img1/' + fname); missing+=1
        elif cur_jpegs[fname].lower() != bh.lower():
            print('CHANGED: ' + seq + '/img1/' + fname); changed+=1
    for fname in cur_jpegs:
        if fname not in b_img:
            print('ADDED:   ' + seq + '/img1/' + fname); added+=1
if changed+missing+added > 0: all_ok=False
print('changed=' + str(changed) + '  missing=' + str(missing) + '  added=' + str(added))
print('CHECK 5: ' + ('PASS' if changed+missing+added==0 else 'FAIL'))
print()

# ── CHECK 6: 18 overlays ──────────────────────────────────────────────────────
print('=== CHECK 6: OVERLAYS (first/middle/last per copy) -> check/aug_overlays_v2/ ===')
OVL = os.path.join(CHECK, 'aug_overlays_v2')
os.makedirs(OVL, exist_ok=True)
ovl_ok = 0; ovl_fail = 0
for name, src, seed in COPIES:
    copy_dir = os.path.join(AUG, name)
    ann = parse_gt(os.path.join(copy_dir, 'gt', 'gt.txt'))
    img1_dir = os.path.join(copy_dir, 'img1')
    files = sorted(f for f in os.listdir(img1_dir) if f.endswith('.jpg'))
    n = len(files)
    picks = [files[0], files[n//2], files[n-1]]
    for fname in picks:
        fid = int(fname.replace('.jpg',''))
        img = cv2.imread(os.path.join(img1_dir, fname))
        if img is None: ovl_fail+=1; continue
        for (tid,x,y,w,h,cls) in ann.get(fid,[]):
            x1,y1,x2,y2=int(round(x)),int(round(y)),int(round(x+w)),int(round(y+h))
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            lbl='ID'+str(tid)+' '+LABELS.get(cls,'?')
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            cv2.rectangle(img,(x1,y1-th-4),(x1+tw+2,y1),(0,255,0),-1)
            cv2.putText(img,lbl,(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
        cv2.putText(img,'v2_'+name+'  f='+str(fid),(15,45),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,255,255),2)
        out_name='v2_'+name+'_f'+str(fid).zfill(6)+'.jpg'
        ok=cv2.imwrite(os.path.join(OVL,out_name),img,[cv2.IMWRITE_JPEG_QUALITY,88])
        if ok: ovl_ok+=1; print('OK  '+out_name)
        else:  ovl_fail+=1; print('FAIL '+out_name)
if ovl_fail>0: all_ok=False
print('Overlays: ' + str(ovl_ok) + ' OK  ' + str(ovl_fail) + ' FAIL  (expected 18 OK)')
print('CHECK 6: ' + ('PASS' if ovl_fail==0 else 'FAIL'))
print()

# ── CHECK 7: Frame count totals ───────────────────────────────────────────────
print('=== CHECK 7: FRAME COUNT TOTALS ===')
real_train = sum(len([f for f in os.listdir(os.path.join(IOTD,s,'img1')) if f.endswith('.jpg')])
                 for s in ['seq_01_1min','seq_03_30sec_b'])
real_val   = len([f for f in os.listdir(os.path.join(IOTD,'seq_02_30sec_a','img1')) if f.endswith('.jpg')])
aug_total  = sum(len([f for f in os.listdir(os.path.join(AUG,d,'img1')) if f.endswith('.jpg')])
                 for d in os.listdir(AUG) if os.path.isdir(os.path.join(AUG,d,'img1')))
total_train = real_train + aug_total
print('Real train frames   : ' + str(real_train) + '  (expected 2530)')
print('Real val frames     : ' + str(real_val)   + '  (expected 744)')
print('Augmented frames    : ' + str(aug_total)  + '  (expected 7590)')
print('Total train samples : ' + str(total_train) + '  (expected 10120)')
counts_ok = (real_train==2530 and real_val==744 and aug_total==7590 and total_train==10120)
if not counts_ok: all_ok=False
print('CHECK 7: ' + ('PASS' if counts_ok else 'FAIL'))
print()

print('=== B5 OVERALL: ' + ('ALL CHECKS PASS' if all_ok else 'FAILURES PRESENT — see above') + ' ===')
if not all_ok:
    sys.exit(1)
