import os
import sys
import subprocess

def run(cmd):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    return res.returncode == 0

print("Installing gdown and huggingface_hub...")
run(f"{sys.executable} -m pip install gdown huggingface_hub")

# 1. Download Checkpoint
ckpt_dir = os.path.join("checkpoints", "pretrained")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "memotr_dancetrack.pth")

if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 100000000:
    print("Downloading MeMOTR DanceTrack checkpoint with gdown...")
    run(f"{sys.executable} -m gdown 1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2 -O {ckpt_path}")
    if os.path.exists(ckpt_path):
        print(f"Checkpoint downloaded successfully! Size: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")
    else:
        print("gdown failed, trying MOT17 checkpoint...")
        ckpt_path_mot17 = os.path.join(ckpt_dir, "memotr_mot17.pth")
        run(f"{sys.executable} -m gdown 1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew -O {ckpt_path_mot17}")

# 2. Download DanceTrack Val Sequence (dancetrack0001)
import huggingface_hub
target_dir = os.path.join("data", "DanceTrack")
seq_name = "dancetrack0001"
seq_dir = os.path.join(target_dir, "val", seq_name)

print(f"Downloading DanceTrack val sequence {seq_name}...")
try:
    huggingface_hub.snapshot_download(
        repo_id="noahcao/dancetrack",
        repo_type="dataset",
        allow_patterns=[f"val/{seq_name}/*"],
        local_dir=target_dir,
    )
    print(f"Sequence {seq_name} downloaded successfully.")
except Exception as e:
    print("snapshot_download error:", e)

# Also create seqmap
seqmap_path = os.path.join(target_dir, "val_seqmap.txt")
os.makedirs(target_dir, exist_ok=True)
with open(seqmap_path, "w") as f:
    f.write("name\n")
    f.write(f"{seq_name}\n")

print("Asset setup completed.")
