"""
tools/final_review_verify.py — read-only verification for the final review (E002).
1. Stage 1: for seed1, seed2, assoc_lr0, lr_half — five checkpoints exist, five S0 evaluation rows exist, every row's
   metrics match its TrackEval summary file; plus Arm 1 seed 42 from the frozen registry.
   Per-checkpoint mean / SD / range over seeds 42, 1234, 2024; best checkpoint per run (highest HOTA at 0.5).
2. Earlier corrected figures: corrected sweep and seed-check CSV values vs their TrackEval summaries.
"""
import os, csv, json, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE1 = os.path.join(ROOT, "results", "tuning_stage1")
REG = os.path.join(ROOT, "archive", "pre_tuning_baseline_v1", "data", "results_registry_v1.csv")
CK_DIRS = {"seed1": os.path.join(STAGE1, "seed1_r2"), "seed2": os.path.join(ROOT, "results", "train_iotd_real_seed2"),
           "assoc_lr0": os.path.join(STAGE1, "assoc_lr0"), "lr_half": os.path.join(STAGE1, "lr_half")}
M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]

def summ(p):
    l = [x.split() for x in open(p) if x.strip()]; d = dict(zip(l[0], l[1]))
    return {"HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]), "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]),
            "IDF1": float(d["IDF1"]), "IDSW": float(d["IDSW"]), "FP": float(d["CLR_FP"]), "FN": float(d["CLR_FN"]),
            "recall": float(d["CLR_Re"]), "precision": float(d["CLR_Pr"])}

def check(rows, label):
    bad = 0
    for r in rows:
        sp = r.get("summary_path") or r.get("source_path")
        p = sp if os.path.isabs(sp) else os.path.join(ROOT, sp)
        if not os.path.exists(p):
            print(f"  {label} {r.get('run', r.get('seed'))} {r.get('checkpoint', r.get('setting'))}: SUMMARY MISSING {p}")
            bad += 1; continue
        s = summ(p)
        for m in M:
            if abs(float(r[m]) - s[m]) > 0.011:
                print(f"  MISMATCH {label} {r} {m}: csv {r[m]} summary {s[m]}"); bad += 1
    return bad

def main():
    print("== 1. Stage 1 completion ==")
    rows = list(csv.DictReader(open(os.path.join(STAGE1, "stage1_results.csv"))))
    status = json.load(open(os.path.join(STAGE1, "status.json")))
    for run, d in CK_DIRS.items():
        ck = [f"checkpoint_{i}.pth" for i in range(5) if os.path.exists(os.path.join(d, f"checkpoint_{i}.pth"))]
        ev = [r for r in rows if r["run"] == run and r["scoring"] == "S0"]
        bad = check(ev, run)
        st_ = status["jobs"].get(run, {})
        print(f"{run:9s} dir={os.path.relpath(d, ROOT)} checkpoints={len(ck)}/5 eval_rows={len(ev)}/5 "
              f"summary_mismatches={bad} status.json={st_.get('status')} train_h={st_.get('train_wall_h')} "
              f"backbone_max={st_.get('param_diff', {}).get('backbone_max')} "
              f"qu_max={st_.get('param_diff', {}).get('query_updater_max')}")
    reg = list(csv.DictReader(open(REG)))
    arm1 = {r["checkpoint"]: r for r in reg if r["run"] == "Arm1" and float(r["det_thresh"]) == 0.5
            and r["sequence"] == "seq_02_30sec_a" and int(r["miss_tol"]) == 30}
    print(f"Arm1 seed42 rows in frozen registry: {len(arm1)}/5; summary mismatches: {check(list(arm1.values()), 'Arm1')}")

    print("\n== Per-checkpoint seed spread (old unfiltered output, S0, threshold 0.5) ==")
    by = {("42", r["checkpoint"]): r for r in arm1.values()}
    for r in rows:
        if r["run"] in ("seed1", "seed2"):
            by[("1234" if r["run"] == "seed1" else "2024", r["checkpoint"])] = r
    for i in range(5):
        c = f"ckpt_{i}"
        v = [float(by[(s, c)]["HOTA"]) for s in ("42", "1234", "2024")]
        print(f"  {c} ({2528*(i+1):,} steps) HOTA 42/1234/2024 = {v}  mean {st.mean(v):.2f} SD {st.stdev(v):.2f} "
              f"range {max(v)-min(v):.2f}")
    print("\n== Best checkpoint per run (highest HOTA of 5, threshold 0.5, old unfiltered output, S0) ==")
    for run in ("Arm1", "seed1", "seed2", "assoc_lr0", "lr_half"):
        src = list(arm1.values()) if run == "Arm1" else [r for r in rows if r["run"] == run]
        b = max(src, key=lambda r: float(r["HOTA"]))
        print(f"  {run:9s} {b['checkpoint']} HOTA {float(b['HOTA']):.2f} IDSW {int(float(b['IDSW']))}")
    for run in ("Arm2",):
        src = [r for r in reg if r["run"] == run and float(r["det_thresh"]) == 0.5 and int(r["miss_tol"]) == 30]
        b = max(src, key=lambda r: float(r["HOTA"]))
        print(f"  {run:9s} {b['checkpoint']} HOTA {float(b['HOTA']):.2f} (frozen registry; {len(src)} rows)")

    print("\n== 2. Earlier corrected figures vs TrackEval summaries ==")
    cs = list(csv.DictReader(open(os.path.join(ROOT, "results_after_tuning", "corrected_sweep_results.csv"))))
    print(f"corrected_sweep_results.csv rows={len(cs)} mismatches={check(cs, 'sweep')}")
    for r in cs:
        print(f"  {r['run']:20s} HOTA {float(r['HOTA']):.2f} IDSW {r['IDSW']} FP {r['FP']} FN {r['FN']}")
    sc = list(csv.DictReader(open(os.path.join(ROOT, "results_after_tuning", "seed_check_results.csv"))))
    print(f"seed_check_results.csv rows={len(sc)} mismatches={check(sc, 'seed')}")
    v = {(r["seed"], r["setting"]): float(r["HOTA"]) for r in sc}
    for s in ("42", "1234", "2024"):
        print(f"  seed {s}: S_A {v[(s,'S_A')]:.2f} S_D {v[(s,'S_D')]:.2f} diff {v[(s,'S_D')]-v[(s,'S_A')]:+.2f}")
    a = [v[(s, "S_A")] for s in ("42", "1234", "2024")]; d = [v[(s, "S_D")] for s in ("42", "1234", "2024")]
    print(f"  mean S_A {st.mean(a):.2f} ± {st.stdev(a):.2f}; S_D {st.mean(d):.2f} ± {st.stdev(d):.2f}; "
          f"mean diff {st.mean([y-x for x, y in zip(a, d)]):+.2f}")

if __name__ == "__main__":
    main()
