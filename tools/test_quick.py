import os, sys, time

# Write to log file directly as well as stdout
log_file = open("d:/MajorProject-MeMOTR/data/IOTD/check/test_quick.log", "w", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()

log("Step 1: start")
os.environ["MPLBACKEND"] = "Agg"
import torch
log("Step 2: torch OK")
MAJOR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(MAJOR, "MeMOTR"))
from data.vehiclemot import build as build_vehiclemot
log("Step 3: vehiclemot OK")
from utils.utils import yaml_to_dict
cfg = yaml_to_dict(os.path.join(MAJOR, "configs", "train_iotd_real.yaml"))
cfg["CONFIG_PATH"] = os.path.join(MAJOR, "configs", "train_iotd_real.yaml")
ds = build_vehiclemot(cfg, split="train")
log(f"Step 4: loaded {len(ds.vid_idx)} sequences: {list(ds.vid_idx.keys())}")
log("SUCCESS")
log_file.close()
