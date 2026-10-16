"""
tools/after_tuning_sweep_run.py — corrected inference sweep (output-score filter as upstream MeMOTR), one call per run.

    python -u tools/after_tuning_sweep_run.py S_A            # corrected run -> results_after_tuning/sweep/S_A/
    python -u tools/after_tuning_sweep_run.py S_A --filters-off
        # equivalence check -> results_after_tuning/verify_filters_off/S_A/ ; must reproduce
        # results/inference_sweep/S_A/seq_02_30sec_a.txt byte for byte

Same as the previous sweep (tools/sweep_run.py) except the inference tool: tools/run_inference_mot_filtered.py,
with RESULT_SCORE_THRESH from configs/train_iotd_real.yaml (0.5) and area > 100 px, as upstream submit_engine.py.
Arm 1 checkpoint_3, data/IOTD_val (seq_02_30sec_a), use_c1/use_c2 off, scoring tools/run_trackeval.py with
results/eval_iotd_val/gt_eval (class-1 eval copy of the S0 GT), DO_PREPROC=False. Refuses to overwrite.
"""
import os, sys, subprocess, time, hashlib

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
CKPT   = os.path.join(ROOT, "results", "train_iotd_real", "checkpoint_3.pth")
CONFIG = os.path.join(ROOT, "configs", "train_iotd_real.yaml")
SPLIT  = os.path.join(ROOT, "data", "IOTD_val")
GT_EVAL = os.path.join(ROOT, "results", "eval_iotd_val", "gt_eval")
OUT_ROOT = os.path.join(ROOT, "results_after_tuning")
SETTINGS = {"S_A": (0.5, 0.5, 30), "S_B": (0.6, 0.6, 30), "S_C": (0.7, 0.7, 30), "S_D": (0.5, 0.5, 20),
            "S_E": (0.5, 0.5, 40), "S_F": (0.6, 0.6, 20), "S_G": (0.7, 0.7, 20)}

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
    own = (f'"{os.getpid()}"', f'"{os.getppid()}"')
    others = [l for l in procs.splitlines() if l.startswith('"python.exe"') and not any(o in l for o in own)]
    print(f"GPU: {smi}; other python.exe processes (excluding self and parent): {len(others)}")
    for l in others:
        print("   ", l)
    return len(others) == 0

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main():
    run_id = sys.argv[1]; off = "--filters-off" in sys.argv
    det, trk, miss = SETTINGS[run_id]
    out = os.path.join(OUT_ROOT, "verify_filters_off" if off else "sweep", run_id)
    if os.path.exists(out) and os.listdir(out):
        print(f"REFUSE: {out} exists and is not empty"); sys.exit(2)
    print(f"{run_id}: det={det} track={trk} miss_tol={miss} filters={'OFF' if off else 'ON (config)'} ckpt={CKPT}")
    if not preflight():
        print("REFUSE: another python process is running"); sys.exit(3)
    os.makedirs(out, exist_ok=True)
    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_inference_mot_filtered.py"),
           "--checkpoint", CKPT, "--split-dirs", SPLIT, "--output-dir", out, "--config", CONFIG,
           "--label", f"ARM1_CKPT3_{run_id}_{'NOFILTER' if off else 'FILTERED'}", "--det-thresh", str(det),
           "--track-thresh", str(trk), "--miss-tol", str(miss)]
    if off:
        cmd += ["--result-score-thresh", "-1", "--area-thresh", "-1"]
    print("CMD:", " ".join(cmd)); t0 = time.time()
    with open(os.path.join(out, "inference.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, "MeMOTR"), env=env(), stdout=f, stderr=subprocess.STDOUT,
                           timeout=1800)
    trk_txt = os.path.join(out, "seq_02_30sec_a.txt")
    if r.returncode != 0 or not os.path.exists(trk_txt):
        print(f"inference FAILED rc={r.returncode}"); sys.exit(1)
    for l in open(os.path.join(out, "inference.log"), encoding="utf-8", errors="replace"):
        if "output filter" in l.lower():
            print("  log:", l.rstrip())
    print(f"inference done in {time.time()-t0:.0f}s, {sum(1 for _ in open(trk_txt))} rows, sha256={sha(trk_txt)}")
    if off:
        ref = os.path.join(ROOT, "results", "inference_sweep", run_id, "seq_02_30sec_a.txt")
        same = sha(ref) == sha(trk_txt)
        print(f"EQUIVALENCE vs {ref}: sha256={sha(ref)} identical={same}")
        sys.exit(0 if same else 4)

    label = f"AFTER_{run_id}"
    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_trackeval.py"), "--tracker-dir", out,
           "--gt-dirs", GT_EVAL, "--output-dir", os.path.join(out, "trackeval"), "--label", label, "--num-cores", "1"]
    with open(os.path.join(out, "trackeval.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=f, stderr=subprocess.STDOUT, timeout=300)
    summ = os.path.join(out, "trackeval", "trackeval_output", label, "pedestrian_summary.txt")
    if r.returncode != 0 or not os.path.exists(summ):
        print(f"TrackEval FAILED rc={r.returncode}"); sys.exit(1)
    l = [x.split() for x in open(summ) if x.strip()]; d = dict(zip(l[0], l[1]))
    print(f"summary: {summ}")
    print("  ".join(f"{k}={d[k]}" for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN",
                                            "CLR_Re", "CLR_Pr")))

if __name__ == "__main__":
    main()
