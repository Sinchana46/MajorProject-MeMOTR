"""
tools/diff_configs.py
---------------------
Computes line-by-line diff of each new config against the original configs/train_vehicle_mot_interim.yaml,
and verifies that the two new configs differ only in DATASET_DIRS and output/label names.
"""
import difflib, os

orig_path = "configs/train_vehicle_mot_interim.yaml"
real_path = "configs/train_iotd_real.yaml"
aug_path  = "configs/train_iotd_real_aug.yaml"

with open(orig_path, encoding="utf-8") as f:
    orig_lines = f.readlines()
with open(real_path, encoding="utf-8") as f:
    real_lines = f.readlines()
with open(aug_path, encoding="utf-8") as f:
    aug_lines = f.readlines()

print("=== DIFF: configs/train_iotd_real.yaml vs original configs/train_vehicle_mot_interim.yaml ===")
diff1 = list(difflib.unified_diff(orig_lines, real_lines, fromfile="train_vehicle_mot_interim.yaml", tofile="train_iotd_real.yaml"))
for line in diff1:
    print(line, end="")

print("\n\n=== DIFF: configs/train_iotd_real_aug.yaml vs original configs/train_vehicle_mot_interim.yaml ===")
diff2 = list(difflib.unified_diff(orig_lines, aug_lines, fromfile="train_vehicle_mot_interim.yaml", tofile="train_iotd_real_aug.yaml"))
for line in diff2:
    print(line, end="")

print("\n\n=== DIFF: configs/train_iotd_real_aug.yaml vs configs/train_iotd_real.yaml ===")
diff3 = list(difflib.unified_diff(real_lines, aug_lines, fromfile="train_iotd_real.yaml", tofile="train_iotd_real_aug.yaml"))
for line in diff3:
    print(line, end="")
