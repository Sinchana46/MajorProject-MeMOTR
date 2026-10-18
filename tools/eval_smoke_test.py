"""
E3 Smoke test: run phase5 checkpoint on first 50 frames of seq_02_30sec_a.
"""
import subprocess, sys, os, tempfile, shutil

# Create a temporary sequence directory with only 50 frames
ORIG_IMG1 = "data/IOTD_val/seq_02_30sec_a/img1"
ORIG_SEQINFO = "data/IOTD_val/seq_02_30sec_a/seqinfo.ini"

# Build a temp dir
smoke_dir = "results/eval_iotd_val/smoke_test/seq_02_30sec_a"
os.makedirs(os.path.join(smoke_dir, "img1"), exist_ok=True)

# Symlink/copy 50 frames
import glob
frames = sorted(glob.glob(os.path.join(ORIG_IMG1, "*.jpg")))[:50]
for src in frames:
    dst = os.path.join(smoke_dir, "img1", os.path.basename(src))
    if not os.path.exists(dst):
        import shutil
        shutil.copy2(src, dst)

# Write seqinfo.ini with seqlength=50
with open(ORIG_SEQINFO) as f:
    seqinfo = f.read()
seqinfo_50 = seqinfo.replace("seqlength = 744", "seqlength = 50")
with open(os.path.join(smoke_dir, "seqinfo.ini"), "w") as f:
    f.write(seqinfo_50)

print(f"Smoke dir: {smoke_dir} ({len(frames)} frames)")

# Run inference
smoke_out = "results/eval_iotd_val/smoke_output"
os.makedirs(smoke_out, exist_ok=True)

cmd = [
    sys.executable, "-u", "tools/run_inference_mot.py",
    "--checkpoint", "results/phase5_interim_baseline/checkpoint_4.pth",
    "--split-dirs", os.path.dirname(smoke_dir),
    "--output-dir", smoke_out,
    "--config", "configs/train_iotd_real.yaml",
    "--label", "SMOKE_PHASE5",
    "--det-thresh", "0.5",
    "--track-thresh", "0.5",
    "--miss-tol", "30",
]
print("CMD:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
stdout = result.stdout + result.stderr
print(stdout[-3000:] if len(stdout) > 3000 else stdout)

# Check output
out_file = os.path.join(smoke_out, "seq_02_30sec_a.txt")
if os.path.exists(out_file):
    with open(out_file) as f:
        rows = [l for l in f.readlines() if l.strip()]
    print(f"\nTracker output: {out_file}")
    print(f"  Rows: {len(rows)}")
    print(f"  First row: {rows[0].rstrip()}")
    print(f"  Last row:  {rows[-1].rstrip()}")
    cols = rows[0].split(",")
    print(f"  Columns: {len(cols)}")
    frames_out = sorted(set(int(r.split(',')[0]) for r in rows))
    print(f"  Frame range: {frames_out[0]} to {frames_out[-1]}")
    print("SMOKE TEST: PASS")
else:
    print(f"ERROR: output file not found at {out_file}")
    sys.exit(1)
