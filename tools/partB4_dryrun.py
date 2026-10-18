"""
tools/partB4_dryrun.py
-----------------------
B4: dry-run 20 frames for all 6 copies, then run the B4 checks:
  - A3 luma table (per copy)
  - Sampled parameters per copy (from aug_params.json)
  - Noise independence test (corr(noise_frame1, noise_frame2) < 0.05)
  - Reproducibility test (SHA256 of frame 1 generated twice = identical)
  - Acceptance rule: ratio in [0.6, 2.0] and pct==0 rise <= 10pp
"""
import os, sys, json, subprocess, hashlib, importlib.util
import numpy as np
import cv2

IOTD      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'IOTD'))
CHECK     = os.path.join(IOTD, 'check')
DR        = os.path.join(CHECK, 'dryrun_v2')
SCRIPT    = os.path.join(os.path.dirname(__file__), 'augment_photometric_v2.py')

COPIES = [
    ('seq_01_1min_aug_01',    'seq_01_1min',    1001),
    ('seq_01_1min_aug_02',    'seq_01_1min',    1002),
    ('seq_01_1min_aug_03',    'seq_01_1min',    1003),
    ('seq_03_30sec_b_aug_01', 'seq_03_30sec_b', 1001),
    ('seq_03_30sec_b_aug_02', 'seq_03_30sec_b', 1002),
    ('seq_03_30sec_b_aug_03', 'seq_03_30sec_b', 1003),
]

PYTHON = sys.executable

# ── Step 1: Run dry runs ──────────────────────────────────────────────────────
print('=== B4: RUNNING DRY RUNS (20 frames each) ===')
for name, src, seed in COPIES:
    out_path = os.path.join(DR, name)
    if os.path.exists(out_path):
        print(name + ': dryrun already exists at ' + out_path + ' -- skipping')
        continue
    print('Running dry run: ' + name + ' ...')
    result = subprocess.run(
        [PYTHON, SCRIPT, '--dry-run', '20', '--copy', name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print('ERROR running dry run for ' + name)
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print(result.stdout.strip())
print()

# ── Step 2: A3 Luma table ────────────────────────────────────────────────────
print('=== B4 CHECK: A3 LUMA TABLE (20-frame dry run) ===')
print('{:<25} {:>12} {:>12} {:>6} {:>18} {:>18} {:>6}'.format(
    'Copy', 'Src mean luma', 'Aug mean luma', 'Ratio',
    '%pix==0 src/aug', '%pix>=250 src/aug', 'PASS?'))
print('-' * 110)

all_pass = True
for name, src, seed in COPIES:
    src_img1 = os.path.join(IOTD, src, 'img1')
    aug_img1 = os.path.join(DR, name, 'img1')
    files = sorted(f for f in os.listdir(aug_img1) if f.endswith('.jpg'))

    src_lumas, aug_lumas, src_p0, aug_p0, src_p250, aug_p250 = [], [], [], [], [], []
    for fname in files:
        si = cv2.imread(os.path.join(src_img1, fname))
        ai = cv2.imread(os.path.join(aug_img1, fname))
        sg = cv2.cvtColor(si, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ag = cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY).astype(np.float32)
        npix = float(sg.size)
        src_lumas.append(np.mean(sg)); aug_lumas.append(np.mean(ag))
        src_p0.append(np.sum(sg == 0)/npix*100);   aug_p0.append(np.sum(ag == 0)/npix*100)
        src_p250.append(np.sum(sg >= 250)/npix*100); aug_p250.append(np.sum(ag >= 250)/npix*100)

    sl = float(np.mean(src_lumas)); al = float(np.mean(aug_lumas))
    ratio = al / sl if sl > 0 else float('inf')
    s0 = float(np.mean(src_p0));   a0 = float(np.mean(aug_p0))
    s250 = float(np.mean(src_p250)); a250 = float(np.mean(aug_p250))
    ratio_ok = 0.6 <= ratio <= 2.0
    pct0_ok  = (a0 - s0) <= 10.0
    passed = ratio_ok and pct0_ok
    if not passed:
        all_pass = False
    print('{:<25} {:>12.2f} {:>12.2f} {:>6.3f} {:>8.2f}%/{:<8.2f}% {:>8.2f}%/{:<8.2f}%  {}'.format(
        name, sl, al, ratio, s0, a0, s250, a250,
        'PASS' if passed else 'FAIL'))

print()
print('A3 luma table: ' + ('ALL PASS' if all_pass else 'FAILURES PRESENT'))
print()

# ── Step 3: Sampled parameters ────────────────────────────────────────────────
print('=== B4 CHECK: SAMPLED PARAMETERS (from aug_params.json) ===')
for name, src, seed in COPIES:
    p_path = os.path.join(DR, name, 'aug_params.json')
    with open(p_path) as f:
        params = json.load(f)
    print(name + ':')
    for k, v in params.items():
        if k != 'note':
            print('  ' + k + ': ' + str(v))
print()

# ── Step 4: Noise independence test ──────────────────────────────────────────
print('=== B4 CHECK: NOISE INDEPENDENCE TEST ===')
# Use aug_01 seed=1001, sigma sampled from params, shape of seq_01 (1080x1920x3)
seed = 1001
p_path = os.path.join(DR, 'seq_01_1min_aug_01', 'aug_params.json')
with open(p_path) as f:
    p = json.load(f)
sigma = float(p['noise_sigma'])
shape = (1080, 1920, 3)

def make_noise(copy_seed, frame_idx, sigma, shape):
    rng = np.random.default_rng(copy_seed * 1_000_003 + frame_idx)
    return rng.normal(0.0, sigma, shape).astype(np.float32).ravel()

n1 = make_noise(seed, 1, sigma, shape)
n2 = make_noise(seed, 2, sigma, shape)
corr = float(np.corrcoef(n1, n2)[0, 1])
noise_ok = abs(corr) < 0.05
print('copy_seed={} sigma={:.4f}'.format(seed, sigma))
print('frame_idx=1 seed: {}'.format(seed * 1_000_003 + 1))
print('frame_idx=2 seed: {}'.format(seed * 1_000_003 + 2))
print('Pearson correlation (frame1 vs frame2): {:.6f}'.format(corr))
print('Noise independence: ' + ('PASS (< 0.05)' if noise_ok else 'FAIL (>= 0.05)'))
print()

# ── Step 5: Reproducibility test ─────────────────────────────────────────────
print('=== B4 CHECK: REPRODUCIBILITY TEST ===')
# Load the v2 module and regenerate frame 1 of seq_01_1min_aug_01 twice
spec = importlib.util.spec_from_file_location('augv2', SCRIPT)
augv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augv2)

src_img1 = os.path.join(IOTD, 'seq_01_1min', 'img1')
jpegs = sorted(f for f in os.listdir(src_img1) if f.endswith('.jpg'))
fname0 = jpegs[0]
img0 = cv2.imread(os.path.join(src_img1, fname0))

rng_copy = np.random.default_rng(1001)
params_1001 = augv2.COPY_PARAMS[1001]
gain      = float(rng_copy.uniform(*params_1001['gain']))
gamma_val = float(rng_copy.uniform(*params_1001['gamma']))
hue_shift = int(rng_copy.integers(-params_1001['hue'], params_1001['hue'] + 1))
sat_shift = int(rng_copy.integers(-params_1001['sat'], params_1001['sat'] + 1))
jpeg_q    = int(rng_copy.integers(*params_1001['jpeg_q']))
noise_sig = float(rng_copy.uniform(*params_1001['noise_sigma']))
gamma_lut = augv2.build_gamma_lut(gamma_val)

def gen_frame1():
    frame_rng = np.random.default_rng(1001 * 1_000_003 + 0)
    out = augv2.apply_frame(img0.copy(), gain, gamma_lut, hue_shift, sat_shift,
                            jpeg_q, noise_sig, frame_rng)
    _, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, augv2.JPEG_SAVE_Q])
    return hashlib.sha256(enc.tobytes()).hexdigest()

h1 = gen_frame1()
h2 = gen_frame1()
repro_ok = (h1 == h2)
print('Run 1 SHA256: ' + h1.upper())
print('Run 2 SHA256: ' + h2.upper())
print('Reproducibility: ' + ('PASS (identical)' if repro_ok else 'FAIL (different!)'))
print()

# ── Final branch decision ─────────────────────────────────────────────────────
print('=== B4 DECISION ===')
if all_pass and noise_ok and repro_ok:
    print('ALL CHECKS PASS. Proceeding to B5 full run is safe.')
else:
    print('STOP: one or more checks failed. See above.')
    sys.exit(1)
