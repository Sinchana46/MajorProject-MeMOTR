"""
tools/final_review_run.py — one corrected-procedure inference + TrackEval job for the final review.

    python -u tools/final_review_run.py <job_id>

Jobs are defined in JOBS below. Every job uses tools/run_inference_mot_filtered.py (RESULT_SCORE_THRESH from the
config = 0.5, area > 100 px), det 0.5 / track 0.5 / miss 30, use_c1 off, and tools/run_trackeval.py
(DO_PREPROC=False). C2-on jobs add --c2-adapter-checkpoint and record per-frame tau via a read-only hook.
Outputs: results_after_tuning/final_review/runs/<job_id>/. Refuses to overwrite.
"""
import os, sys, subprocess, time, hashlib

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
OUT_ROOT = os.path.join(ROOT, "results_after_tuning", "final_review", "runs")
R = lambda *p: os.path.join(ROOT, *p)
CFG_REAL, CFG_AUG = R("configs", "train_iotd_real.yaml"), R("configs", "train_iotd_real_aug.yaml")
ADAPTER = R("results", "phase6_c2_interim", "c2_adapter_epoch_4.pth")
VAL_SPLIT, VAL_GT = R("data", "IOTD_val"), R("results", "eval_iotd_val", "gt_eval")
C2_SPLIT = lambda s: R("results", "c2_eval", "splits", s)
C2_GT = R("results", "c2_eval", "gt_eval")

CKPT = {"Phase5": (R("results", "phase5_interim_baseline"), CFG_REAL),
        "Arm1": (R("results", "train_iotd_real"), CFG_REAL),
        "Arm2": (R("results", "train_iotd_real_aug"), CFG_AUG),
        "assoc_lr0": (R("results", "tuning_stage1", "assoc_lr0"), CFG_REAL),
        "lr_half": (R("results", "tuning_stage1", "lr_half"), CFG_REAL)}

JOBS = {}
JOBS["Phase5_ckpt_4"] = dict(model="Phase5", ckpt=4, seq="seq_02_30sec_a", split=VAL_SPLIT, gt=VAL_GT, c2="off")
for m in ("Arm1", "Arm2", "assoc_lr0", "lr_half"):
    for i in range(5):
        if m == "Arm1" and i == 3:
            continue        # reused: results_after_tuning/sweep/S_A (same runner, ckpt, thresholds, GT, protocol)
        JOBS[f"{m}_ckpt_{i}"] = dict(model=m, ckpt=i, seq="seq_02_30sec_a", split=VAL_SPLIT, gt=VAL_GT, c2="off")
for s in ("seq_01_1min", "seq_02_30sec_a", "seq_03_30sec_b"):
    for c2 in ("off", "on"):
        if s == "seq_02_30sec_a" and c2 == "off":
            continue        # identical to job Phase5_ckpt_4 (reused)
        JOBS[f"C2_{s}_{c2}"] = dict(model="Phase5", ckpt=4, seq=s, split=C2_SPLIT(s), gt=C2_GT, c2=c2)

def env():
    e = os.environ.copy()
    memotr = R("MeMOTR")
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
    print(f"GPU: {smi}; other python.exe processes (excluding self and parent driver): {len(others)}")
    return len(others) == 0

def main():
    jid = sys.argv[1]; j = JOBS[jid]
    ck_dir, cfg = CKPT[j["model"]]
    ckpt = os.path.join(ck_dir, f"checkpoint_{j['ckpt']}.pth")
    out = os.path.join(OUT_ROOT, jid)
    if os.path.exists(out) and os.listdir(out):
        print(f"REFUSE: {out} exists and is not empty"); sys.exit(2)
    print(f"{jid}: ckpt={ckpt} config={cfg} seq={j['seq']} c2={j['c2']} det/track 0.5 miss 30 filtered")
    if not os.path.exists(ckpt):
        print("BLOCKED: checkpoint missing"); sys.exit(5)
    if not preflight():
        print("REFUSE: another python process is running"); sys.exit(3)
    os.makedirs(out, exist_ok=True)
    tool = "final_review_c2_tau.py" if j["c2"] == "on" else "run_inference_mot_filtered.py"
    cmd = [PYTHON, "-u", R("tools", tool), "--checkpoint", ckpt, "--split-dirs", j["split"], "--output-dir", out,
           "--config", cfg, "--label", jid, "--det-thresh", "0.5", "--track-thresh", "0.5", "--miss-tol", "30"]
    if j["c2"] == "on":
        cmd += ["--c2-adapter-checkpoint", ADAPTER]
    print("CMD:", " ".join(cmd)); t0 = time.time()
    with open(os.path.join(out, "inference.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=R("MeMOTR"), env=env(), stdout=f, stderr=subprocess.STDOUT, timeout=1800)
    trk = os.path.join(out, f"{j['seq']}.txt")
    if r.returncode != 0 or not os.path.exists(trk):
        print(f"inference FAILED rc={r.returncode}"); sys.exit(1)
    for l in open(os.path.join(out, "inference.log"), encoding="utf-8", errors="replace"):
        if "output filter" in l.lower() or "tau over" in l or "C2 adapter loaded" in l:
            print("  log:", l.rstrip())
    print(f"inference done in {time.time()-t0:.0f}s, rows={sum(1 for _ in open(trk))}, "
          f"sha256={hashlib.sha256(open(trk, 'rb').read()).hexdigest()}")
    label = jid.upper()
    cmd = [PYTHON, "-u", R("tools", "run_trackeval.py"), "--tracker-dir", out, "--gt-dirs", j["gt"],
           "--output-dir", os.path.join(out, "trackeval"), "--label", label, "--num-cores", "1"]
    with open(os.path.join(out, "trackeval.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=f, stderr=subprocess.STDOUT, timeout=300)
    summ = os.path.join(out, "trackeval", "trackeval_output", label, "pedestrian_summary.txt")
    if r.returncode != 0 or not os.path.exists(summ):
        print(f"TrackEval FAILED rc={r.returncode}"); sys.exit(1)
    l = [x.split() for x in open(summ) if x.strip()]; d = dict(zip(l[0], l[1]))
    print(f"summary: {summ}")
    print("RESULT " + "  ".join(f"{k}={d[k]}" for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FP",
                                                       "CLR_FN", "CLR_Pr", "CLR_Re")))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print("\n".join(JOBS))
    else:
        main()
