"""
tools/sweep_run.py — Inference-time threshold sweep, one setting per call.

Usage:  python -u tools/sweep_run.py <RUN_ID>      (RUN_ID in S_A .. S_G)

Arm 1 checkpoint_3, configs/train_iotd_real.yaml, data/IOTD_val (seq_02_30sec_a), use_c1/use_c2 off.
Same tools as the baseline package: tools/run_inference_mot.py + tools/run_trackeval.py
(evaluation-only class-1 GT results/eval_iotd_val/gt_eval, DO_PREPROC=False).
Outputs: results/inference_sweep/<RUN_ID>/seq_02_30sec_a.txt and .../<RUN_ID>/trackeval/. Refuses to overwrite.
"""
import os, sys, subprocess, time

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
CKPT   = os.path.join(ROOT, "results", "train_iotd_real", "checkpoint_3.pth")
CONFIG = os.path.join(ROOT, "configs", "train_iotd_real.yaml")
SPLIT  = os.path.join(ROOT, "data", "IOTD_val")
GT_EVAL = os.path.join(ROOT, "results", "eval_iotd_val", "gt_eval")
OUT_ROOT = os.path.join(ROOT, "results", "inference_sweep")

SETTINGS = {  # run_id: (det_thresh, track_thresh, miss_tol)
    "S_A": (0.5, 0.5, 30), "S_B": (0.6, 0.6, 30), "S_C": (0.7, 0.7, 30),
    "S_D": (0.5, 0.5, 20), "S_E": (0.5, 0.5, 40), "S_F": (0.6, 0.6, 20), "S_G": (0.7, 0.7, 20),
}

def env():
    e = os.environ.copy()
    memotr = os.path.join(ROOT, "MeMOTR")
    e["PYTHONPATH"] = memotr + os.pathsep + os.path.join(memotr, "models", "ops")
    e["PYTHONIOENCODING"] = "utf-8"; e["PYTHONUTF8"] = "1"
    if CUDA_BIN not in e.get("PATH", ""):
        e["PATH"] = CUDA_BIN + os.pathsep + e.get("PATH", "")
    return e

def preflight():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    procs = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True).stdout
    others = [l for l in procs.splitlines() if l.startswith('"python.exe"') and f'"{os.getpid()}"' not in l]
    print(f"GPU: {smi}")
    print(f"other python.exe processes: {len(others)}")
    for l in others:
        print("   ", l)
    return len(others) == 0

def main():
    run_id = sys.argv[1]
    det, trk, miss = SETTINGS[run_id]
    out = os.path.join(OUT_ROOT, run_id)
    if os.path.exists(out) and os.listdir(out):
        print(f"REFUSE: {out} exists and is not empty"); sys.exit(2)
    print(f"{run_id}: det_thresh={det} track_thresh={trk} miss_tol={miss}  ckpt={CKPT}")
    if not preflight():
        print("REFUSE: another python process is running"); sys.exit(3)
    os.makedirs(out, exist_ok=True)

    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_inference_mot.py"),
           "--checkpoint", CKPT, "--split-dirs", SPLIT, "--output-dir", out, "--config", CONFIG,
           "--label", f"ARM1_CKPT3_{run_id}", "--det-thresh", str(det), "--track-thresh", str(trk),
           "--miss-tol", str(miss)]
    print("CMD:", " ".join(cmd)); t0 = time.time()
    with open(os.path.join(out, "inference.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, "MeMOTR"), env=env(), stdout=f,
                           stderr=subprocess.STDOUT, timeout=1800)
    trk_txt = os.path.join(out, "seq_02_30sec_a.txt")
    if r.returncode != 0 or not os.path.exists(trk_txt):
        print(f"inference FAILED rc={r.returncode}"); sys.exit(1)
    n = sum(1 for _ in open(trk_txt))
    print(f"inference done in {time.time()-t0:.0f}s, {n} rows -> {trk_txt}")

    label = f"SWEEP_{run_id}"
    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_trackeval.py"), "--tracker-dir", out,
           "--gt-dirs", GT_EVAL, "--output-dir", os.path.join(out, "trackeval"), "--label", label,
           "--num-cores", "1"]
    print("CMD:", " ".join(cmd))
    with open(os.path.join(out, "trackeval.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=f, stderr=subprocess.STDOUT, timeout=300)
    summ = os.path.join(out, "trackeval", "trackeval_output", label, "pedestrian_summary.txt")
    if r.returncode != 0 or not os.path.exists(summ):
        print(f"TrackEval FAILED rc={r.returncode}"); sys.exit(1)
    lines = [l.split() for l in open(summ) if l.strip()]
    d = dict(zip(lines[0], lines[1]))
    print(f"summary: {summ}")
    print("  ".join(f"{k}={d[k]}" for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW",
                                            "CLR_FP", "CLR_FN", "CLR_Re", "CLR_Pr")))

if __name__ == "__main__":
    main()
