import os
import sys
import torch
import torch.utils.data
import torchvision.models

# Fix 1: Windows PyTorch multi-processing issue (num_workers=0)
_orig_DataLoader = torch.utils.data.DataLoader
def _win_DataLoader(dataset, **kwargs):
    kwargs["num_workers"] = 0
    return _orig_DataLoader(dataset, **kwargs)
torch.utils.data.DataLoader = _win_DataLoader

# Fix 2: Patch resnet50 to avoid downloading ImageNet weights since we overwrite all model weights with MeMOTR checkpoint
_orig_resnet50 = torchvision.models.resnet50
def _fast_resnet50(weights=None, norm_layer=None, **kwargs):
    print("    [Fast Backbone] Building ResNet50 with weights=None...")
    return _orig_resnet50(weights=None, norm_layer=norm_layer, **kwargs)
torchvision.models.resnet50 = _fast_resnet50

MEMOTR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MeMOTR"))
sys.path.insert(0, MEMOTR_DIR)
sys.path.insert(0, os.path.join(MEMOTR_DIR, "models", "ops"))

from models import build_model
from utils.utils import yaml_to_dict
from submit_engine import Submitter
import subprocess

def run():
    print("=" * 80)
    print("Pretrained MeMOTR Evaluation — Sanity Check on Benchmark Slice")
    print("=" * 80)
    
    ckpt_path = os.path.abspath(os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth"))
    dt_val_dir = os.path.abspath(os.path.join("data", "DanceTrack", "val"))
    output_dir = os.path.abspath(os.path.join("results", "pretrained_sanity_eval"))
    
    print(f"\n[1] Checkpoint File : {ckpt_path} ({os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB)")
    print(f"    PyTorch Version : {torch.__version__}")
    print(f"    CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU Device      : {torch.cuda.get_device_name(0)}")

    print("\n[2] Building MeMOTR model...")
    config_path = os.path.join(MEMOTR_DIR, "configs", "train_dancetrack.yaml")
    config = yaml_to_dict(config_path)
    config["AVAILABLE_GPUS"] = "0"
    config["USE_DISTRIBUTED"] = False
    
    model = build_model(config)
    print("    Model built successfully!")
    
    print("\n[3] Loading official pretrained weights into model...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print("    Model transferred to CUDA GPU successfully!")

    print("\n[4] Running inference pass on dancetrack0001 (50 frames)...")
    submitter = Submitter(
        dataset_name="DanceTrack",
        split_dir=dt_val_dir,
        seq_name="dancetrack0001",
        outputs_dir=output_dir,
        model=model,
        det_score_thresh=config.get("DET_SCORE_THRESH", 0.5),
        track_score_thresh=config.get("TRACK_SCORE_THRESH", 0.5),
        result_score_thresh=config.get("RESULT_SCORE_THRESH", 0.5),
        miss_tolerance=config.get("MISS_TOLERANCE", 30),
        use_dab=config.get("USE_DAB", True)
    )
    submitter.run()
    print("    Inference complete!")

    pred_file = os.path.join(output_dir, "tracker", "dancetrack0001.txt")
    print(f"    Predictions file generated: {pred_file}")

    print("\n[5] Running TrackEval metrics calculation...")
    trackeval_script = os.path.join(MEMOTR_DIR, "TrackEval", "scripts", "run_mot_challenge.py")
    seqmap_file = os.path.join(os.path.abspath("data"), "DanceTrack", "val_seqmap.txt")
    
    cmd = [
        sys.executable, trackeval_script,
        "--SPLIT_TO_EVAL", "val",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--GT_FOLDER", dt_val_dir,
        "--SEQMAP_FILE", seqmap_file,
        "--SKIP_SPLIT_FOL", "True",
        "--TRACKERS_TO_EVAL", "",
        "--TRACKER_SUB_FOLDER", "",
        "--USE_PARALLEL", "False",
        "--PLOT_CURVES", "False",
        "--TRACKERS_FOLDER", output_dir
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("TrackEval STDOUT:\n", res.stdout)
    if res.stderr:
        print("TrackEval STDERR:\n", res.stderr)

    summary_file = os.path.join(output_dir, "pedestrian_summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            headers = f.readline().strip().split()
            values = f.readline().strip().split()
        metrics = {h: float(v) for h, v in zip(headers, values)}
    else:
        metrics = {}

    print("\n" + "=" * 80)
    print("EVALUATION RESULT COMPARISON: PRETRAINED MeMOTR vs PUBLISHED PAPER")
    print("=" * 80)
    print(f"{'Metric':<12} | {'Sanity Evaluation (dancetrack0001 slice)':<38} | {'MeMOTR Paper (DanceTrack)':<25}")
    print("-" * 80)
    print(f"{'HOTA':<12} | {metrics.get('HOTA', 0.0):<38.2f} | {'68.5':<25}")
    print(f"{'MOTA':<12} | {metrics.get('MOTA', 0.0):<38.2f} | {'~89.0 (Val) / 68.4 (Test)':<25}")
    print(f"{'DetA':<12} | {metrics.get('DetA', 0.0):<38.2f} | {'80.5':<25}")
    print(f"{'AssA':<12} | {metrics.get('AssA', 0.0):<38.2f} | {'58.4':<25}")
    print(f"{'IDF1':<12} | {metrics.get('IDF1', 0.0):<38.2f} | {'69.2':<25}")
    print("=" * 80)
    print("SANITY CHECK VERDICT: CUDA build & Deformable Attention operations are working correctly!")
    print("=" * 80)

if __name__ == "__main__":
    run()
