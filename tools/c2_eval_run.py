"""
tools/c2_eval_run.py — C2 three-sequence evaluation, one sequence x one C2 setting per call.

Usage:  python -u tools/c2_eval_run.py <seq> <on|off>

Phase 5 checkpoint_4 (never trained on IOTD), configs/train_iotd_real.yaml (same as the baseline Phase 5 eval),
det/track threshold 0.5, miss tolerance 30, use_c1 off.
  off : tools/run_inference_mot.py without --c2-adapter-checkpoint (USE_C2 False in the config)
  on  : same tool, with --c2-adapter-checkpoint results/phase6_c2_interim/c2_adapter_epoch_4.pth,
        run through tools/c2_eval_infer_tau.py so the per-frame tau is recorded.
seq_02 + off: the existing baseline tracker output results/eval_iotd_val/phase5/seq_02_30sec_a.txt is reused
(copied, never re-run, never modified); it is only re-scored here.
Scoring: tools/run_trackeval.py with results/c2_eval/gt_eval (class-1 eval copies), DO_PREPROC=False.
Outputs: results/c2_eval/<seq>_c2<on|off>/. Refuses to overwrite.
"""
import os, sys, shutil, subprocess, time, hashlib

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
CKPT    = os.path.join(ROOT, "results", "phase5_interim_baseline", "checkpoint_4.pth")
CONFIG  = os.path.join(ROOT, "configs", "train_iotd_real.yaml")
ADAPTER = os.path.join(ROOT, "results", "phase6_c2_interim", "c2_adapter_epoch_4.pth")
OUT     = os.path.join(ROOT, "results", "c2_eval")
EXISTING_SEQ02_OFF = os.path.join(ROOT, "results", "eval_iotd_val", "phase5", "seq_02_30sec_a.txt")

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
    # ignore this process and its parent (tools/c2_eval_batch.py, a driver that runs no model itself)
    own = (f'"{os.getpid()}"', f'"{os.getppid()}"')
    others = [l for l in procs.splitlines() if l.startswith('"python.exe"') and not any(o in l for o in own)]
    print(f"GPU: {smi}; other python.exe processes (excluding self and parent driver): {len(others)}")
    for l in others:
        print("   ", l)
    return len(others) == 0

def main():
    seq, c2 = sys.argv[1], sys.argv[2]
    assert c2 in ("on", "off")
    out = os.path.join(OUT, f"{seq}_c2{c2}")
    if os.path.exists(out) and os.listdir(out):
        print(f"REFUSE: {out} exists and is not empty"); sys.exit(2)
    os.makedirs(out, exist_ok=True)
    trk = os.path.join(out, f"{seq}.txt")

    if seq == "seq_02_30sec_a" and c2 == "off" and os.path.exists(EXISTING_SEQ02_OFF):
        shutil.copy2(EXISTING_SEQ02_OFF, trk)
        h = hashlib.sha256(open(EXISTING_SEQ02_OFF, "rb").read()).hexdigest()
        print(f"reused existing tracker output (not re-run): {EXISTING_SEQ02_OFF} sha256={h}")
        with open(os.path.join(out, "SOURCE.txt"), "w") as f:
            f.write(f"copied from {EXISTING_SEQ02_OFF}\nsha256 {h}\n")
    else:
        if not preflight():
            print("REFUSE: another python process is running"); sys.exit(3)
        split = os.path.join(OUT, "splits", seq)
        script = "c2_eval_infer_tau.py" if c2 == "on" else "run_inference_mot.py"
        cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", script), "--checkpoint", CKPT,
               "--split-dirs", split, "--output-dir", out, "--config", CONFIG,
               "--label", f"PHASE5_{seq}_C2{c2.upper()}", "--det-thresh", "0.5",
               "--track-thresh", "0.5", "--miss-tol", "30"]
        if c2 == "on":
            cmd += ["--c2-adapter-checkpoint", ADAPTER]
        print("CMD:", " ".join(cmd)); t0 = time.time()
        with open(os.path.join(out, "inference.log"), "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=os.path.join(ROOT, "MeMOTR"), env=env(), stdout=f,
                               stderr=subprocess.STDOUT, timeout=1800)
        if r.returncode != 0 or not os.path.exists(trk):
            print(f"inference FAILED rc={r.returncode} (see {out}/inference.log)"); sys.exit(1)
        print(f"inference done in {time.time()-t0:.0f}s")
        for l in open(os.path.join(out, "inference.log"), encoding="utf-8", errors="replace"):
            if "C2 adapter" in l or "tau over" in l or "Loading C2" in l:
                print("  log:", l.rstrip())
    print(f"tracker rows: {sum(1 for _ in open(trk))}")

    label = f"C2EVAL_{seq}_C2{c2.upper()}"
    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_trackeval.py"), "--tracker-dir", out,
           "--gt-dirs", os.path.join(OUT, "gt_eval"), "--output-dir", os.path.join(out, "trackeval"),
           "--label", label, "--num-cores", "1"]
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
