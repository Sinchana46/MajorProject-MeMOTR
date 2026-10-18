import subprocess, sys, os

DATA_DIR = "data/IOTD_val"
EVAL_GT  = "results/eval_iotd_val/gt_eval"
PYTHON   = sys.executable

RUNS = [
    # Priority 1 (equal-step: 10112)
    ("arm1", 3, "results/train_iotd_real/checkpoint_3.pth",
               "results/eval_iotd_val/arm1_ckpt3",
               "configs/train_iotd_real.yaml", "ARM1_CKPT3"),
    ("arm2", 0, "results/train_iotd_real_aug/checkpoint_0.pth",
               "results/eval_iotd_val/arm2_ckpt0",
               "configs/train_iotd_real_aug.yaml", "ARM2_CKPT0"),
    # Priority 2
    ("arm1", 0, "results/train_iotd_real/checkpoint_0.pth",
               "results/eval_iotd_val/arm1_ckpt0",
               "configs/train_iotd_real.yaml", "ARM1_CKPT0"),
    ("arm1", 1, "results/train_iotd_real/checkpoint_1.pth",
               "results/eval_iotd_val/arm1_ckpt1",
               "configs/train_iotd_real.yaml", "ARM1_CKPT1"),
    ("arm1", 2, "results/train_iotd_real/checkpoint_2.pth",
               "results/eval_iotd_val/arm1_ckpt2",
               "configs/train_iotd_real.yaml", "ARM1_CKPT2"),
    ("arm2", 1, "results/train_iotd_real_aug/checkpoint_1.pth",
               "results/eval_iotd_val/arm2_ckpt1",
               "configs/train_iotd_real_aug.yaml", "ARM2_CKPT1"),
    ("arm2", 2, "results/train_iotd_real_aug/checkpoint_2.pth",
               "results/eval_iotd_val/arm2_ckpt2",
               "configs/train_iotd_real_aug.yaml", "ARM2_CKPT2"),
    ("arm2", 3, "results/train_iotd_real_aug/checkpoint_3.pth",
               "results/eval_iotd_val/arm2_ckpt3",
               "configs/train_iotd_real_aug.yaml", "ARM2_CKPT3"),
]

for arm, n, ckpt, out_dir, cfg, label in RUNS:
    out_file = os.path.join(out_dir, "seq_02_30sec_a.txt")
    if os.path.exists(out_file):
        with open(out_file) as f:
            rows = sum(1 for l in f if l.strip())
        print(f"[{label}] Already exists ({rows} rows) - SKIP")
        continue
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PYTHON, "-u", "tools/run_inference_mot.py",
           "--checkpoint", ckpt, "--split-dirs", DATA_DIR,
           "--output-dir", out_dir, "--config", cfg,
           "--label", label, "--det-thresh", "0.5",
           "--track-thresh", "0.5", "--miss-tol", "30"]
    print(f"\n[{label}] Inference...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    det_line = [l for l in out.split("\n") if "detections" in l or "Done" in l or "Error" in l or "Traceback" in l]
    print(f"  {det_line[-3:]}")
    if os.path.exists(out_file):
        with open(out_file) as f:
            rows = sum(1 for l in f if l.strip())
        print(f"  -> {rows} rows")
    else:
        print(f"  ERROR: output not found")

# Now run TrackEval on all new checkpoints
print("\n=== TRACKEVAL FOR ALL PART A CHECKPOINTS ===")
te_results = {}
for arm, n, ckpt, tracker_dir, cfg, label in RUNS:
    te_out = os.path.join("results/eval_iotd_val", f"trackeval_{label.lower()}")
    cmd = [PYTHON, "-u", "tools/run_trackeval.py",
           "--tracker-dir", tracker_dir,
           "--gt-dirs", EVAL_GT,
           "--output-dir", te_out,
           "--label", label, "--num-cores", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # Find summary
    import os as _os
    for root, dirs, files in _os.walk(te_out):
        for fn in files:
            if "pedestrian_summary" in fn.lower():
                with open(_os.path.join(root, fn)) as f:
                    lines = f.readlines()
                vals = dict(zip(lines[0].split(), lines[1].split()))
                steps = (n+1) * (2528 if arm=="arm1" else 10112)
                te_results[label] = {**vals, "steps": steps, "arm": arm, "n": n}
                print(f"  [{label}] steps={steps:,} HOTA={vals.get('HOTA','?')} "
                      f"MOTA={vals.get('MOTA','?')} FP={vals.get('CLR_FP','?')} "
                      f"FN={vals.get('CLR_FN','?')} Recall={vals.get('CLR_Re','?')} "
                      f"Prec={vals.get('CLR_Pr','?')}")

# Print full ordered table
print("\n=== PART A TABLE (ordered by steps) ===")
all_rows = []
# Add phase5 (0 IOTD steps)
all_rows.append({"label":"phase5","steps":0,"arm":"phase5","HOTA":"50.51","DetA":"43.88",
                 "AssA":"58.49","MOTA":"29.10","IDF1":"68.84","IDSW":"1",
                 "CLR_FP":"2117","CLR_FN":"847","CLR_Re":"79.747","CLR_Pr":"61.17"})
for label, vals in sorted(te_results.items(), key=lambda x: x[1]["steps"]):
    all_rows.append({**vals, "label": label})
# Also add arm1_ckpt4 and arm2_ckpt4
for label, steps, h, da, aa, mo, i1, sw, fp, fn, re, pr in [
    ("arm1_ckpt4",12640,"49.58","45.70","54.04","24.94","69.66","7","2855","277","93.376","57.766"),
    ("arm2_ckpt4",50560,"42.84","38.32","48.01","-6.58","63.72","6","4266","185","95.576","48.372"),
]:
    all_rows.append({"label":label,"steps":steps,"HOTA":h,"DetA":da,"AssA":aa,
                     "MOTA":mo,"IDF1":i1,"IDSW":sw,"CLR_FP":fp,"CLR_FN":fn,
                     "CLR_Re":re,"CLR_Pr":pr})

all_rows.sort(key=lambda x: int(x.get("steps",0)))
print(f"{'Label':<16}{'Steps':>8}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'MOTA':>8}{'IDF1':>7}{'IDSW':>5}{'FP':>7}{'FN':>6}{'Rec':>8}{'Prec':>8}")
print("-"*100)
for r in all_rows:
    print(f"  {r['label']:<14}{int(r.get('steps',0)):>8,}{r.get('HOTA','?'):>7}{r.get('DetA','?'):>7}"
          f"{r.get('AssA','?'):>7}{r.get('MOTA','?'):>8}{r.get('IDF1','?'):>7}"
          f"{r.get('IDSW','?'):>5}{r.get('CLR_FP','?'):>7}{r.get('CLR_FN','?'):>6}"
          f"{float(r.get('CLR_Re','0')):>8.2f}{float(r.get('CLR_Pr','0')):>8.2f}")

print("\nPart A DONE")
