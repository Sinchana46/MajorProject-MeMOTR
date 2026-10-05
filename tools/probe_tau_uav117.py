"""
probe_tau_uav117.py
Diagnostic: sample frames from uav0000117_02622_v and log tau_adaptive
values produced by the trained ThresholdAdapter, confirming clamp behaviour.
"""
import sys
import glob
import torch

sys.path.insert(0, "d:/MajorProject-MeMOTR/MeMOTR")
sys.path.insert(0, "d:/MajorProject-MeMOTR")

from PIL import Image
import torchvision.transforms.functional as TF
from modules.scene_analyzer import SceneAnalyzer
from modules.threshold_adapter import ThresholdAdapter

ADAPTER_CKPT = "d:/MajorProject-MeMOTR/results/phase6_c2_interim/c2_adapter_epoch_4.pth"
SEQ_DIR      = "d:/MajorProject-MeMOTR/data/VisDrone_Vehicles/val/uav0000117_02622_v/img1"

# ── Load adapter ───────────────────────────────────────────────────────────────
ckpt = torch.load(ADAPTER_CKPT, map_location="cpu", weights_only=False)
tau_min = ckpt["tau_min"]
tau_max = ckpt["tau_max"]
adapter = ThresholdAdapter(tau_min=tau_min, tau_max=tau_max)
adapter.load_state_dict(ckpt["state_dict"])
adapter.eval()
analyzer = SceneAnalyzer(max_tracks=50)

print(f"Adapter: tau_min={tau_min}, tau_max={tau_max}")
print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
print()

# ── Sample frames ──────────────────────────────────────────────────────────────
def frame_key(path):
    name = path.replace("\\", "/").split("/")[-1]
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0

all_files = sorted(
    glob.glob(SEQ_DIR + "/*.jpg") + glob.glob(SEQ_DIR + "/*.png"),
    key=frame_key
)
step = max(1, len(all_files) // 30)
sample_files = all_files[::step][:30]
print(f"Total frames in sequence: {len(all_files)}")
print(f"Sampling {len(sample_files)} frames (every {step})")
print()

header = f"{'Frame':<25}  {'brightness':>10}  {'contrast':>10}  {'density':>10}  {'tau_pred':>10}  {'in_range':>10}"
print(header)
print("-" * len(header))

tau_values    = []
raw_mlp_outs  = []

# We also want to inspect the raw MLP output before sigmoid-and-scale
# Monkey-patch to capture
_orig_forward = adapter.forward

def _patched_forward(scene_vec):
    with torch.no_grad():
        raw = adapter._mlp(scene_vec)
        sig = torch.sigmoid(raw)
        scaled = tau_min + (tau_max - tau_min) * sig
    return scaled

for f in sample_files:
    img   = Image.open(f).convert("RGB")
    raw_t = TF.to_tensor(img)          # (3,H,W) float [0,1]
    with torch.no_grad():
        scene_vec = analyzer(raw_t, active_track_count=0)
        tau       = adapter(scene_vec).item()

        # Also capture intermediate values
        raw_mlp   = adapter._mlp(scene_vec)
        sig_val   = torch.sigmoid(raw_mlp).item()

    tau_values.append(tau)
    b, c, d = scene_vec.tolist()
    in_range = tau_min <= tau <= tau_max
    name = f.replace("\\", "/").split("/")[-1]

    flag = "OK" if in_range else "OUT_OF_RANGE"
    print(f"{name:<25}  {b:>10.4f}  {c:>10.4f}  {d:>10.4f}  {tau:>10.6f}  {flag:>10}")

print()
print("=" * 60)
print(f"tau_min clamp  : {tau_min}")
print(f"tau_max clamp  : {tau_max}")
print(f"Observed min   : {min(tau_values):.6f}")
print(f"Observed max   : {max(tau_values):.6f}")
n_out = sum(1 for t in tau_values if not (tau_min <= t <= tau_max))
print(f"Out-of-range   : {n_out} / {len(tau_values)}")

if n_out == 0:
    print()
    print("RESULT: All tau values WITHIN [tau_min, tau_max]. Clamp is working.")
    print("=> The 61k-detection explosion is NOT a clamp bug.")
    print("   Even with tau~0.30 (the minimum allowed), MVI/UAV frames at this")
    print("   resolution push nearly all det-queries above threshold.")
    print("   Root cause: the proxy training did not penalise low-brightness UAV")
    print("   sequences where tau=0.30 is still far too permissive.")
else:
    print()
    print("BUG: Some tau values are OUTSIDE the clamp range. Investigate forward().")

print()
print("Brightness distribution across sampled frames:")
brightnesses = []
for f in sample_files:
    img   = Image.open(f).convert("RGB")
    raw_t = TF.to_tensor(img)
    with torch.no_grad():
        scene_vec = analyzer(raw_t, active_track_count=0)
    brightnesses.append(scene_vec[0].item())
avg_b = sum(brightnesses) / len(brightnesses)
print(f"  mean brightness = {avg_b:.4f}")
print(f"  min  brightness = {min(brightnesses):.4f}")
print(f"  max  brightness = {max(brightnesses):.4f}")
print()
print("For reference, tau = tau_min + (tau_max - tau_min) * sigmoid(mlp_out)")
print(f"  With tau_min={tau_min}, tau_max={tau_max}:")
print(f"  At brightness=0.00 (black): tau -> ~{tau_min:.3f}  (sigmoid(mlp)~0)")
print(f"  At brightness=1.00 (white): tau -> ~{tau_max:.3f}  (sigmoid(mlp)~1)")
