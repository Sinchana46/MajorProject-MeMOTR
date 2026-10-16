"""
tools/after_tuning_seed_run.py — corrected-filter inference on the seed1 / seed2 checkpoint_3 files (no training),
so the run-to-run noise band can be re-measured under the corrected output behaviour.

    python -u tools/after_tuning_seed_run.py <1234|2024> <S_A|S_D>

Identical to tools/after_tuning_sweep_run.py except the checkpoint. Seed 42 (Arm 1) is the corrected sweep itself.
Outputs: results_after_tuning/seed_check/seed<seed>_<run>/. Refuses to overwrite.
"""
import os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import after_tuning_sweep_run as s

CKPTS = {"1234": os.path.join(s.ROOT, "results", "tuning_stage1", "seed1_r2", "checkpoint_3.pth"),
         "2024": os.path.join(s.ROOT, "results", "train_iotd_real_seed2", "checkpoint_3.pth")}

def main():
    seed, run_id = sys.argv[1], sys.argv[2]
    det, trk, miss = s.SETTINGS[run_id]
    ckpt = CKPTS[seed]
    out = os.path.join(s.OUT_ROOT, "seed_check", f"seed{seed}_{run_id}")
    if os.path.exists(out) and os.listdir(out):
        print(f"REFUSE: {out} exists and is not empty"); sys.exit(2)
    print(f"seed {seed} {run_id}: det={det} track={trk} miss_tol={miss} filters=ON (config) ckpt={ckpt}")
    if not s.preflight():
        print("REFUSE: another python process is running"); sys.exit(3)
    os.makedirs(out, exist_ok=True)
    cmd = [s.PYTHON, "-u", os.path.join(s.ROOT, "tools", "run_inference_mot_filtered.py"),
           "--checkpoint", ckpt, "--split-dirs", s.SPLIT, "--output-dir", out, "--config", s.CONFIG,
           "--label", f"SEED{seed}_CKPT3_{run_id}_FILTERED", "--det-thresh", str(det), "--track-thresh", str(trk),
           "--miss-tol", str(miss)]
    print("CMD:", " ".join(cmd)); t0 = time.time()
    with open(os.path.join(out, "inference.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=os.path.join(s.ROOT, "MeMOTR"), env=s.env(), stdout=f,
                           stderr=subprocess.STDOUT, timeout=1800)
    trk_txt = os.path.join(out, "seq_02_30sec_a.txt")
    if r.returncode != 0 or not os.path.exists(trk_txt):
        print(f"inference FAILED rc={r.returncode}"); sys.exit(1)
    for l in open(os.path.join(out, "inference.log"), encoding="utf-8", errors="replace"):
        if "output filter" in l.lower():
            print("  log:", l.rstrip())
    print(f"inference done in {time.time()-t0:.0f}s")
    label = f"SEED{seed}_{run_id}"
    cmd = [s.PYTHON, "-u", os.path.join(s.ROOT, "tools", "run_trackeval.py"), "--tracker-dir", out,
           "--gt-dirs", s.GT_EVAL, "--output-dir", os.path.join(out, "trackeval"), "--label", label,
           "--num-cores", "1"]
    with open(os.path.join(out, "trackeval.log"), "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=s.ROOT, env=s.env(), stdout=f, stderr=subprocess.STDOUT, timeout=300)
    summ = os.path.join(out, "trackeval", "trackeval_output", label, "pedestrian_summary.txt")
    if r.returncode != 0 or not os.path.exists(summ):
        print(f"TrackEval FAILED rc={r.returncode}"); sys.exit(1)
    l = [x.split() for x in open(summ) if x.strip()]; d = dict(zip(l[0], l[1]))
    print(f"summary: {summ}")
    print("  ".join(f"{k}={d[k]}" for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN",
                                            "CLR_Re", "CLR_Pr")))

if __name__ == "__main__":
    main()
