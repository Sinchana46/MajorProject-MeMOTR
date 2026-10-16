"""
tools/sweep_collect.py — collect the 7 inference-sweep settings into results/inference_sweep/sweep_results.csv
(baseline-registry columns + det_thresh, track_thresh, miss_tol), print the table, and apply the decision rule:

  better than S_A  <=>  HOTA > Arm1 seed mean + 2 * seed SD at checkpoint_3   AND   IDSW <= IDSW(S_A)
  and a setting is only recommended if it also improves HOTA by >= 1.0 over S_A.

Seed mean / SD at checkpoint_3 (10,112 steps) come from Arm 1 (registry) + seed1 + seed2
(results/tuning_stage1/stage1_results.csv), sample SD (n=3).
"""
import os, csv, hashlib, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "inference_sweep")
REG = os.path.join(ROOT, "results", "pre_tuning_baseline_v1", "data", "results_registry.csv")
STAGE1 = os.path.join(ROOT, "results", "tuning_stage1", "stage1_results.csv")
GT_SHA = "17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472"
SETTINGS = {"S_A": (0.5, 0.5, 30), "S_B": (0.6, 0.6, 30), "S_C": (0.7, 0.7, 30), "S_D": (0.5, 0.5, 20),
            "S_E": (0.5, 0.5, 40), "S_F": (0.6, 0.6, 20), "S_G": (0.7, 0.7, 20)}
COLS = ["run", "checkpoint", "steps", "epochs_completed", "seed", "config", "scoring", "threshold", "GT_sha256",
        "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "TP", "FP", "FN", "recall", "precision",
        "summary_path", "summary_sha256", "det_thresh", "track_thresh", "miss_tol"]

def summary(path):
    l = [x.split() for x in open(path) if x.strip()]
    return dict(zip(l[0], l[1]))

def main():
    arm1 = [r for r in csv.DictReader(open(REG)) if r["run"] == "Arm1" and r["checkpoint"] == "ckpt_3"
            and r["threshold"] == "0.5"][0]
    seeds = [float(arm1["HOTA"])] + [float(r["HOTA"]) for r in csv.DictReader(open(STAGE1))
                                     if r["run"] in ("seed1", "seed2") and r["checkpoint"] == "ckpt_3"]
    mean, sd = st.mean(seeds), st.stdev(seeds)
    bar = mean + 2 * sd
    print(f"Arm1 ckpt_3 seed HOTA values {seeds}: mean={mean:.3f} SD={sd:.3f} -> bar (mean+2SD)={bar:.3f}")
    print(f"Registry Arm1 ckpt_3: HOTA={arm1['HOTA']} IDSW={arm1['IDSW']} FP={arm1['FP']} FN={arm1['FN']}")

    rows = []
    for rid, (det, trk, miss) in SETTINGS.items():
        p = os.path.join(OUT, rid, "trackeval", "trackeval_output", f"SWEEP_{rid}", "pedestrian_summary.txt")
        if not os.path.exists(p):
            print(f"{rid}: MISSING summary {p}"); continue
        d = summary(p)
        rows.append({"run": f"Arm1_sweep_{rid}", "checkpoint": "ckpt_3", "steps": 10112, "epochs_completed": 4,
                     "seed": 42, "config": "configs/train_iotd_real.yaml", "scoring": "S0", "threshold": det,
                     "GT_sha256": GT_SHA, "HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]),
                     "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]), "IDF1": float(d["IDF1"]),
                     "IDSW": int(float(d["IDSW"])), "TP": int(float(d["CLR_TP"])), "FP": int(float(d["CLR_FP"])),
                     "FN": int(float(d["CLR_FN"])), "recall": float(d["CLR_Re"]), "precision": float(d["CLR_Pr"]),
                     "summary_path": os.path.relpath(p, ROOT).replace(os.sep, "/"),
                     "summary_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                     "det_thresh": det, "track_thresh": trk, "miss_tol": miss})
    dst = os.path.join(OUT, "sweep_results.csv")
    if os.path.exists(dst):
        print(f"REFUSE to overwrite {dst}")
    else:
        with open(dst, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        print(f"written {dst} ({len(rows)} rows)")

    sa = next(r for r in rows if r["run"].endswith("S_A"))
    print(f"\n{'Run':4s} {'det':>4s} {'trk':>4s} {'miss':>4s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s} {'MOTA':>6s} "
          f"{'IDF1':>6s} {'IDSW':>4s} {'FP':>5s} {'FN':>5s} {'Rec':>6s} {'Prec':>6s} | {'dHOTA':>6s} {'>bar':>5s} "
          f"{'IDSW<=SA':>8s} {'+1pt':>5s} verdict")
    for r in rows:
        rid = r["run"][-3:]
        above = r["HOTA"] > bar; idsw_ok = r["IDSW"] <= sa["IDSW"]; plus1 = r["HOTA"] - sa["HOTA"] >= 1.0
        if rid == "S_A":
            verdict = "reference"
        elif above and idsw_ok and plus1:
            verdict = "PASS"
        elif above and idsw_ok:
            verdict = "passes bar but <1 pt over S_A: not recommended"
        else:
            verdict = "FAIL"
        r["_verdict"] = verdict
        print(f"{rid:4s} {r['det_thresh']:4.1f} {r['track_thresh']:4.1f} {r['miss_tol']:4d} {r['HOTA']:6.2f} "
              f"{r['DetA']:6.2f} {r['AssA']:6.2f} {r['MOTA']:6.2f} {r['IDF1']:6.2f} {r['IDSW']:4d} {r['FP']:5d} "
              f"{r['FN']:5d} {r['recall']:6.2f} {r['precision']:6.2f} | {r['HOTA']-sa['HOTA']:+6.2f} "
              f"{str(above):>5s} {str(idsw_ok):>8s} {str(plus1):>5s} {verdict}")
    passing = [r for r in rows if r["_verdict"] == "PASS"]
    if passing:
        best = max(passing, key=lambda r: r["HOTA"])
        print(f"\nRecommended Arm 1 setting: {best['run'][-3:]} (det {best['det_thresh']}, track "
              f"{best['track_thresh']}, miss_tol {best['miss_tol']})")
    else:
        print("\nNo setting passes the decision rule: S_A stays (det 0.5, track 0.5, miss_tol 30).")

if __name__ == "__main__":
    main()
