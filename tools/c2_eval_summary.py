"""
tools/c2_eval_summary.py — collect the 6 C2 evaluation runs (3 sequences x C2 off/on) into
results/c2_eval/c2_results.csv, print the results table and the per-sequence tau statistics.
"""
import os, csv, hashlib, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "c2_eval")
SEQS = [("seq_01_1min", "night"), ("seq_02_30sec_a", "night"), ("seq_03_30sec_b", "dusk")]
COLS = ["sequence", "lighting", "c2", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "TP", "FP", "FN", "recall",
        "precision", "tau_mean", "tau_min", "tau_max", "tau_sd", "brightness_mean", "frames",
        "checkpoint", "GT_sha256_eval_copy", "summary_path", "summary_sha256"]

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main():
    rows = []
    for seq, light in SEQS:
        for c2 in ("off", "on"):
            d_ = os.path.join(OUT, f"{seq}_c2{c2}")
            p = os.path.join(d_, "trackeval", "trackeval_output", f"C2EVAL_{seq}_C2{c2.upper()}", "pedestrian_summary.txt")
            if not os.path.exists(p):
                print(f"MISSING {p}"); continue
            l = [x.split() for x in open(p) if x.strip()]; d = dict(zip(l[0], l[1]))
            r = {"sequence": seq, "lighting": light, "c2": c2, "HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]),
                 "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]), "IDF1": float(d["IDF1"]),
                 "IDSW": int(float(d["IDSW"])), "TP": int(float(d["CLR_TP"])), "FP": int(float(d["CLR_FP"])),
                 "FN": int(float(d["CLR_FN"])), "recall": float(d["CLR_Re"]), "precision": float(d["CLR_Pr"]),
                 "checkpoint": "results/phase5_interim_baseline/checkpoint_4.pth",
                 "GT_sha256_eval_copy": sha(os.path.join(OUT, "gt_eval", seq, "gt", "gt.txt")),
                 "summary_path": os.path.relpath(p, ROOT).replace(os.sep, "/"), "summary_sha256": sha(p)}
            tp = os.path.join(d_, "tau_per_frame.csv")
            if c2 == "on" and os.path.exists(tp):
                t = list(csv.DictReader(open(tp)))
                taus = [float(x["tau"]) for x in t]; br = [float(x["brightness"]) for x in t if x["brightness"]]
                r.update(tau_mean=st.mean(taus), tau_min=min(taus), tau_max=max(taus), tau_sd=st.pstdev(taus),
                         brightness_mean=st.mean(br) if br else "", frames=len(t))
            else:
                r.update(tau_mean=0.5, tau_min=0.5, tau_max=0.5, tau_sd=0.0, brightness_mean="", frames="")
            rows.append(r)
    dst = os.path.join(OUT, "c2_results.csv")
    if os.path.exists(dst):
        print(f"REFUSE to overwrite {dst}")
    else:
        with open(dst, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        print(f"written {dst} ({len(rows)} rows)")
    print(f"\n| Sequence | Lighting | C2 | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall | Precision |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['sequence']} | {r['lighting']} | {r['c2']} | {r['HOTA']:.2f} | {r['DetA']:.2f} | {r['AssA']:.2f} | "
              f"{r['MOTA']:.2f} | {r['IDF1']:.2f} | {r['IDSW']} | {r['FP']} | {r['FN']} | {r['recall']:.2f} | "
              f"{r['precision']:.2f} |")
    print("\nTau (C2 on), per sequence over all frames:")
    for r in rows:
        if r["c2"] == "on":
            print(f"  {r['sequence']} ({r['lighting']}): mean={r['tau_mean']:.4f} min={r['tau_min']:.4f} "
                  f"max={r['tau_max']:.4f} sd={r['tau_sd']:.4f}  mean brightness (SceneAnalyzer, 0-1)="
                  f"{r['brightness_mean']:.4f}  frames={r['frames']}")
    print("\nHOTA change with C2 (on - off):")
    for seq, _ in SEQS:
        a = [r for r in rows if r["sequence"] == seq]
        if len(a) == 2:
            off, on = a
            print(f"  {seq}: dHOTA={on['HOTA']-off['HOTA']:+.2f}  dIDSW={on['IDSW']-off['IDSW']:+d}  "
                  f"dFP={on['FP']-off['FP']:+d}  dFN={on['FN']-off['FN']:+d}  dRecall={on['recall']-off['recall']:+.2f}")

if __name__ == "__main__":
    main()
