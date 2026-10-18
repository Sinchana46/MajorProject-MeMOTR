"""
E3 Full inference: run all 3 checkpoints on IOTD val (seq_02_30sec_a, 744 frames).
Writes to results/eval_iotd_val/phase5/, arm1/, arm2/ — refuses to overwrite.
"""
import subprocess, sys, os

DATA_DIR = "data/IOTD_val"   # junction pointing to seq_02_30sec_a
RUNS = [
    ("phase5", "results/phase5_interim_baseline/checkpoint_4.pth",
               "results/eval_iotd_val/phase5",
               "configs/train_iotd_real.yaml"),
    ("arm1",   "results/train_iotd_real/checkpoint_4.pth",
               "results/eval_iotd_val/arm1",
               "configs/train_iotd_real.yaml"),
    ("arm2",   "results/train_iotd_real_aug/checkpoint_4.pth",
               "results/eval_iotd_val/arm2",
               "configs/train_iotd_real_aug.yaml"),
]

for label, ckpt, out_dir, cfg in RUNS:
    out_file = os.path.join(out_dir, "seq_02_30sec_a.txt")
    if os.path.exists(out_file):
        print(f"[{label}] Output already exists: {out_file} — SKIP (refuse to overwrite)")
        continue

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "tools/run_inference_mot.py",
        "--checkpoint", ckpt,
        "--split-dirs", DATA_DIR,
        "--output-dir", out_dir,
        "--config", cfg,
        "--label", label.upper(),
        "--det-thresh", "0.5",
        "--track-thresh", "0.5",
        "--miss-tol", "30",
    ]
    print(f"\n[{label}] Starting inference...")
    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=1800)
    if result.returncode not in (0, 1):  # 1 = PyTorch FutureWarning via stderr
        print(f"[{label}] ERROR: returncode={result.returncode}")
    else:
        print(f"[{label}] Done.")

    # Verify
    if os.path.exists(out_file):
        with open(out_file) as f:
            rows = [l for l in f.readlines() if l.strip()]
        frames = sorted(set(int(r.split(",")[0]) for r in rows))
        print(f"[{label}] Output: {len(rows)} rows, frames {frames[0]}-{frames[-1]}, cols={len(rows[0].split(','))}")
    else:
        print(f"[{label}] ERROR: output file not found!")
        sys.exit(1)

print("\nAll 3 inference runs complete.")
