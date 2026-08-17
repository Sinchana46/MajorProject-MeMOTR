import os

ckpt = os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth")
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    print(f"Checkpoint size: {size_mb:.2f} MB")
else:
    print("Checkpoint file does not exist.")
