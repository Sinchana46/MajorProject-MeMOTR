import os
import subprocess
import glob
from pathlib import Path

# 1. Check Get-Process python
print("=== 1. PROCESS STATUS ===")
try:
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU | Format-Table -AutoSize | Out-String"],
        capture_output=True, text=True, timeout=10
    )
    output = res.stdout.strip()
    if output:
        print("Running python processes:")
        print(output)
    else:
        print("No Python process currently running (Get-Process python returned nothing).")
except Exception as e:
    print(f"Error checking processes: {e}")

# Check log
log_path = Path(r"C:\Users\admin\.gemini\antigravity-ide\brain\0cdf6ded-9433-4efc-b775-b64f278aca82\.system_generated\tasks\task-2175.log")
print(f"\nLog file: {log_path} (exists: {log_path.exists()})")
if log_path.exists():
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Total lines in log: {len(lines)}")
    print("Last 20 lines of log:")
    for line in lines[-20:]:
        print("  " + line.rstrip())

# 2. Check each of the six folders under data/IOTD/augmented_train_v2/
print("\n=== 2. AUGMENTED FOLDERS STATUS ===")
base_dir = Path("data/IOTD/augmented_train_v2")
folders = [
    ("seq_01_1min_aug_01", 1599),
    ("seq_01_1min_aug_02", 1599),
    ("seq_01_1min_aug_03", 1599),
    ("seq_03_30sec_b_aug_01", 931),
    ("seq_03_30sec_b_aug_02", 931),
    ("seq_03_30sec_b_aug_03", 931),
]

all_expected = True
counts = {}
for folder_name, expected in folders:
    folder_path = base_dir / folder_name
    img_dir = folder_path / "img1"
    if img_dir.exists():
        jpgs = list(img_dir.glob("*.jpg"))
        jpg_count = len(jpgs)
    else:
        jpg_count = 0
    
    gt_file = (folder_path / "gt" / "gt.txt").exists()
    labels_file = (folder_path / "gt" / "labels.txt").exists()
    seqinfo_file = (folder_path / "seqinfo.ini").exists()
    aug_params_file = (folder_path / "aug_params.json").exists()
    
    counts[folder_name] = jpg_count
    match = (jpg_count == expected)
    if not match:
        all_expected = False
    print(f"Folder: {folder_name}")
    print(f"  .jpg in img1/: {jpg_count} / {expected} (matches: {match})")
    print(f"  gt/gt.txt: {gt_file}")
    print(f"  gt/labels.txt: {labels_file}")
    print(f"  seqinfo.ini: {seqinfo_file}")
    print(f"  aug_params.json: {aug_params_file}")

print(f"\nAll 6 folders at expected counts: {all_expected}")
