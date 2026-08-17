import os

print("=== Checkpoint Status ===")
ckpt = os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth")
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    print(f"  memotr_dancetrack.pth: {size_mb:.2f} MB")
else:
    print("  memotr_dancetrack.pth: Not present")

print("\n=== Dataset Status ===")
dt_val = os.path.join("data", "DanceTrack", "val")
if os.path.exists(dt_val):
    seqs = os.listdir(dt_val)
    print(f"  Sequences in {dt_val}: {seqs}")
    for seq in seqs:
        seq_path = os.path.join(dt_val, seq)
        if os.path.isdir(seq_path):
            img_dir = os.path.join(seq_path, "img1")
            gt_file = os.path.join(seq_path, "gt", "gt.txt")
            num_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
            has_gt = os.path.exists(gt_file)
            print(f"    - {seq}: {num_imgs} frames, gt.txt exists: {has_gt}")
else:
    print(f"  {dt_val}: Not present")
