"""
tools/tune_report.py — Tuning Stage 1 report numbers (read-only)

Reads results/tuning_stage1/stage1_results.csv, status.json and the frozen baseline registry
(results/pre_tuning_baseline_v1/data/results_registry.csv), re-hashes the original IOTD files,
and prints every table used in results/tuning_stage1/stage1_report.md. Also writes the
computed tables to results/tuning_stage1/stage1_report_tables.json.
"""
import os, sys, csv, json, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_queue as q

ROOT = q.ROOT
REG = os.path.join(ROOT, "results", "pre_tuning_baseline_v1", "data", "results_registry.csv")
METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]
CKPTS = [f"ckpt_{i}" for i in range(5)]
STEPS = {f"ckpt_{i}": 2528 * (i + 1) for i in range(5)}

def load():
    rows = {}
    with open(REG, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scoring"] == "S0" and r["threshold"] in ("0.5", "0.50") and r["run"] in ("Arm1", "Phase5"):
                rows[(r["run"], r["checkpoint"])] = r
    with open(q.RESULTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["run"], r["checkpoint"])] = r
    return rows

def num(r, k):
    return float(r[k])

def rehash():
    ok, msg = q.verify_hashes()
    raw = json.load(open(q.HASH_FILE, encoding="utf-8"))
    known = {p for p, _ in q._flatten_hash_entries(raw)}
    added = []
    for seq in raw["sequences"]:
        base = os.path.join(ROOT, "data", "IOTD", seq)
        for dp, _, fs in os.walk(base):
            for fn in fs:
                rel = os.path.relpath(os.path.join(dp, fn), ROOT).replace(os.sep, "/")
                if rel not in known:
                    added.append(rel)
    return ok, msg, len(known), added

def main():
    rows = load()
    out = {}
    ok, msg, n, added = rehash()
    print(f"REHASH: {msg}; added={len(added)} {added[:5]}")
    out["rehash"] = {"ok": ok, "msg": msg, "files": n, "added": added}

    # seed spread
    print("\n## Seed spread (Arm1 seed42, seed1 1234, seed2 2024)")
    spread = {}
    for c in CKPTS:
        spread[c] = {}
        print(f"\n{c} ({STEPS[c]} steps)")
        print(f"{'metric':6s} {'Arm1':>8s} {'seed1':>8s} {'seed2':>8s} {'mean':>8s} {'sd':>7s} {'range':>7s}")
        for m in METRICS:
            v = [num(rows[(r, c)], m) for r in ("Arm1", "seed1", "seed2")]
            mean, sd = st.mean(v), st.stdev(v)
            spread[c][m] = {"values": v, "mean": mean, "sd": sd, "range": max(v) - min(v)}
            print(f"{m:6s} {v[0]:8.2f} {v[1]:8.2f} {v[2]:8.2f} {mean:8.2f} {sd:7.2f} {max(v)-min(v):7.2f}")
    out["spread"] = spread

    # variants vs Arm1 and seed spread
    for var in ("assoc_lr0", "lr_half"):
        print(f"\n## {var} vs Arm1 (paired seed 42)")
        out[var] = {}
        for c in CKPTS:
            print(f"\n{c}")
            print(f"{'metric':6s} {'var':>8s} {'Arm1':>8s} {'diff':>8s} {'diff/sd':>8s} {'seedmean':>9s} {'z_mean':>7s}")
            out[var][c] = {}
            for m in METRICS:
                v = num(rows[(var, c)], m); a = num(rows[("Arm1", c)], m)
                sd = spread[c][m]["sd"]; mu = spread[c][m]["mean"]
                z = (v - a) / sd if sd else float("nan")
                zm = (v - mu) / sd if sd else float("nan")
                out[var][c][m] = {"value": v, "arm1": a, "diff": v - a, "diff_sd": z, "z_vs_seedmean": zm}
                print(f"{m:6s} {v:8.2f} {a:8.2f} {v-a:8.2f} {z:8.2f} {mu:9.2f} {zm:7.2f}")

    # best checkpoint per run (highest HOTA at 0.5)
    print("\n## Best checkpoint per run (highest HOTA)")
    best = {}
    for run in ("Arm1", "seed1", "seed2", "assoc_lr0", "lr_half"):
        c = max(CKPTS, key=lambda c: num(rows[(run, c)], "HOTA"))
        r = rows[(run, c)]
        best[run] = c
        print(f"{run:9s} {c} steps={STEPS[c]} " + " ".join(f"{m}={num(r,m):.2f}" for m in METRICS)
              + f" recall={num(r,'recall'):.2f} prec={num(r,'precision'):.2f}")
    out["best"] = best

    # decision rule
    print("\n## Decision rule: HOTA > seed mean + 2 sd at >1 checkpoint, without worse AssA or IDSW")
    dec = {}
    for var in ("assoc_lr0", "lr_half"):
        hits = []
        for c in CKPTS:
            h = num(rows[(var, c)], "HOTA"); s = spread[c]["HOTA"]
            if h > s["mean"] + 2 * s["sd"]:
                worse_assa = num(rows[(var, c)], "AssA") < num(rows[("Arm1", c)], "AssA")
                worse_idsw = num(rows[(var, c)], "IDSW") > num(rows[("Arm1", c)], "IDSW")
                hits.append((c, h, s["mean"] + 2 * s["sd"], worse_assa, worse_idsw))
        clean = [x for x in hits if not x[3] and not x[4]]
        verdict = "BETTER" if len(clean) > 1 else "NOT BETTER (default stays)"
        dec[var] = {"hits": hits, "clean_hits": len(clean), "verdict": verdict}
        print(f"{var}: hits={hits} clean={len(clean)} -> {verdict}")
        for c in CKPTS:
            s = spread[c]["HOTA"]
            print(f"   {c}: HOTA {num(rows[(var,c)],'HOTA'):.2f} threshold {s['mean']+2*s['sd']:.2f}")
    out["decision"] = dec

    # FP/FN/precision/recall trends
    print("\n## FP / FN / precision / recall trends")
    for run in ("Arm1", "seed1", "seed2", "assoc_lr0", "lr_half"):
        for m in ("FP", "FN", "precision", "recall"):
            print(f"{run:9s} {m:9s} " + " -> ".join(f"{num(rows[(run,c)],m):.1f}" for c in CKPTS))

    p5 = rows[("Phase5", "ckpt_4")]
    print("\nPhase5 ckpt4: " + " ".join(f"{m}={num(p5,m):.2f}" for m in METRICS + ["recall", "precision"]))

    status = json.load(open(q.STATUS_JSON, encoding="utf-8"))
    print("\n## Jobs")
    for j, v in status["jobs"].items():
        print(j, v["status"], v.get("train_start"), v.get("eval_start"), v.get("end"), v.get("train_wall_h"),
              v.get("error"), json.dumps(v.get("param_diff")))
    with open(os.path.join(q.STAGE_DIR, "stage1_report_tables.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
