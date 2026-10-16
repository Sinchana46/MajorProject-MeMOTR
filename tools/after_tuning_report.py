"""
tools/after_tuning_report.py — build the final outputs of the corrected inference verification (read-only on inputs):
  results_after_tuning/corrected_sweep_results.csv
  results_after_tuning/before_after_comparison.csv
  results_after_tuning/seed_check_results.csv
  results_after_tuning/final_results_dashboard.png
and print every number used in results_after_tuning/final_results_report.md. Refuses to overwrite.
"""
import os, sys, csv, hashlib, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results_after_tuning")
BEFORE = os.path.join(ROOT, "results", "inference_sweep", "sweep_results.csv")
STAGE1 = os.path.join(ROOT, "results", "tuning_stage1", "stage1_results.csv")
REG = os.path.join(ROOT, "archive", "pre_tuning_baseline_v1", "data", "results_registry_v1.csv")
GT_SHA = "17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472"
SETTINGS = {"S_A": (0.5, 0.5, 30), "S_B": (0.6, 0.6, 30), "S_C": (0.7, 0.7, 30), "S_D": (0.5, 0.5, 20),
            "S_E": (0.5, 0.5, 40), "S_F": (0.6, 0.6, 20), "S_G": (0.7, 0.7, 20)}
M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "precision", "recall"]
OI = {"grey": "#999999", "blue": "#0072B2", "vermilion": "#D55E00", "sky": "#56B4E9", "green": "#009E73",
      "orange": "#E69F00"}

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def summ(p):
    l = [x.split() for x in open(p) if x.strip()]; d = dict(zip(l[0], l[1]))
    return {"HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]), "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]),
            "IDF1": float(d["IDF1"]), "IDSW": int(float(d["IDSW"])), "TP": int(float(d["CLR_TP"])),
            "FP": int(float(d["CLR_FP"])), "FN": int(float(d["CLR_FN"])), "recall": float(d["CLR_Re"]),
            "precision": float(d["CLR_Pr"])}

def filter_stats(log):
    for l in open(log, encoding="utf-8", errors="replace"):
        if "output filter (result_score_thresh" in l:
            return eval(l.split("): ", 1)[1].strip())          # dict literal printed by the runner
    return {}

def write_csv(path, rows, cols):
    if os.path.exists(path):
        print(f"REFUSE to overwrite {path}"); return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"written {path} ({len(rows)} rows)")

def main():
    # corrected sweep
    after = {}
    for rid, (det, trk, miss) in SETTINGS.items():
        d = os.path.join(OUT, "sweep", rid)
        p = os.path.join(d, "trackeval", "trackeval_output", f"AFTER_{rid}", "pedestrian_summary.txt")
        fs = filter_stats(os.path.join(d, "inference.log"))
        after[rid] = dict(run=f"Arm1_corrected_{rid}", checkpoint="ckpt_3", steps=10112, epochs_completed=4, seed=42,
                          config="configs/train_iotd_real.yaml", scoring="S0", threshold=det, GT_sha256=GT_SHA,
                          **summ(p), summary_path=os.path.relpath(p, ROOT).replace(os.sep, "/"),
                          summary_sha256=sha(p), det_thresh=det, track_thresh=trk, miss_tol=miss,
                          result_score_thresh=0.5, area_thresh=100, rows_before_filter=fs.get("candidates"),
                          removed_by_score=fs.get("removed_score"), removed_by_area=fs.get("removed_area"),
                          rows_written=fs.get("kept"), min_kept_score=round(fs.get("min_kept_score", float("nan")), 5),
                          inference_tool="tools/run_inference_mot_filtered.py")
    cols = ["run", "checkpoint", "steps", "epochs_completed", "seed", "config", "scoring", "threshold", "GT_sha256",
            "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "TP", "FP", "FN", "recall", "precision", "summary_path",
            "summary_sha256", "det_thresh", "track_thresh", "miss_tol", "result_score_thresh", "area_thresh",
            "rows_before_filter", "removed_by_score", "removed_by_area", "rows_written", "min_kept_score",
            "inference_tool"]
    write_csv(os.path.join(OUT, "corrected_sweep_results.csv"), list(after.values()), cols)

    before = {r["run"][-3:]: {k: float(r[k]) for k in M} for r in csv.DictReader(open(BEFORE))}
    ba = []
    for rid in SETTINGS:
        row = {"setting": rid, "det_thresh": SETTINGS[rid][0], "track_thresh": SETTINGS[rid][1],
               "miss_tol": SETTINGS[rid][2]}
        for m in M:
            row[f"{m}_before"] = before[rid][m]; row[f"{m}_after"] = after[rid][m]
            row[f"{m}_delta"] = round(after[rid][m] - before[rid][m], 3)
        ba.append(row)
    write_csv(os.path.join(OUT, "before_after_comparison.csv"), ba,
              ["setting", "det_thresh", "track_thresh", "miss_tol"] + [f"{m}_{s}" for m in M
                                                                       for s in ("before", "after", "delta")])

    # corrected seed check (S_A and S_D, seeds 42 / 1234 / 2024 at checkpoint_3)
    seeds = {"42": {}, "1234": {}, "2024": {}}
    for rid in ("S_A", "S_D"):
        seeds["42"][rid] = after[rid]
        for sd_ in ("1234", "2024"):
            p = os.path.join(OUT, "seed_check", f"seed{sd_}_{rid}", "trackeval", "trackeval_output",
                             f"SEED{sd_}_{rid}", "pedestrian_summary.txt")
            seeds[sd_][rid] = dict(summ(p), summary_path=os.path.relpath(p, ROOT).replace(os.sep, "/"),
                                   summary_sha256=sha(p))
    srows = []
    for sd_ in seeds:
        for rid in ("S_A", "S_D"):
            r = seeds[sd_][rid]
            srows.append({"seed": sd_, "setting": rid, **{m: r[m] for m in M}, "summary_path": r["summary_path"],
                          "summary_sha256": r["summary_sha256"]})
    write_csv(os.path.join(OUT, "seed_check_results.csv"), srows,
              ["seed", "setting"] + M + ["summary_path", "summary_sha256"])

    old = {"42": 50.47}
    for r in csv.DictReader(open(STAGE1)):
        if r["checkpoint"] == "ckpt_3" and r["run"] in ("seed1", "seed2"):
            old["1234" if r["run"] == "seed1" else "2024"] = float(r["HOTA"])
    print("\n=== Seed band at checkpoint_3 (HOTA) ===")
    band = {}
    for label, vals in (("ORIGINAL output (unfiltered), S_A", [old[s] for s in ("42", "1234", "2024")]),
                        ("CORRECTED output, S_A", [seeds[s]["S_A"]["HOTA"] for s in ("42", "1234", "2024")]),
                        ("CORRECTED output, S_D", [seeds[s]["S_D"]["HOTA"] for s in ("42", "1234", "2024")])):
        mu, sdv = st.mean(vals), st.stdev(vals)
        band[label] = (mu, sdv)
        print(f"{label:38s} seeds 42/1234/2024 = {vals}  mean={mu:.3f} SD={sdv:.3f} mean+2SD={mu+2*sdv:.3f}")
    print("\nPer-seed paired difference S_D - S_A (corrected):")
    for m in ("HOTA", "DetA", "AssA", "IDSW", "FP", "FN"):
        diffs = [seeds[s]["S_D"][m] - seeds[s]["S_A"][m] for s in ("42", "1234", "2024")]
        print(f"  {m:5s} " + "  ".join(f"seed {s}: {d:+.2f}" for s, d in zip(("42", "1234", "2024"), diffs)) +
              f"   mean {st.mean(diffs):+.2f}")
    print("\nPer-seed S_A HOTA change from the filter (corrected - original):",
          {s: round(seeds[s]["S_A"]["HOTA"] - old[s], 2) for s in old})

    mu, sdv = band["CORRECTED output, S_A"]; bar = mu + 2 * sdv
    sa = after["S_A"]
    print(f"\n=== Corrected sweep, decision rule with corrected band (S_A seeds: mean {mu:.2f}, SD {sdv:.2f}, "
          f"bar {bar:.2f}) ===")
    print(f"{'Run':4s} {'det':>4s} {'miss':>4s} " + " ".join(f"{m:>9s}" for m in M) + "   >bar IDSW<=SA +1pt verdict")
    verdict = {}
    for rid, r in after.items():
        above = r["HOTA"] > bar; idok = r["IDSW"] <= sa["IDSW"]; p1 = r["HOTA"] - sa["HOTA"] >= 1.0
        v = "reference" if rid == "S_A" else ("PASS" if above and idok and p1 else "FAIL")
        verdict[rid] = v
        print(f"{rid:4s} {r['det_thresh']:4.1f} {r['miss_tol']:4d} " +
              " ".join(f"{r[m]:9.2f}" if m not in ("IDSW", "FP", "FN") else f"{r[m]:9d}" for m in M) +
              f"   {str(above):5s} {str(idok):8s} {str(p1):5s} {v}")

    # dashboard
    ids = list(SETTINGS)
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.28,
                          left=0.05, right=0.98, top=0.93, bottom=0.04)
    fig.suptitle("Final inference verification — Arm 1 ckpt_3, seq_02_30sec_a, output filter as upstream MeMOTR "
                 "(RESULT_SCORE_THRESH 0.5, area > 100 px)", fontsize=12)
    x = np.arange(len(ids)); w = 0.38
    def bars(ax, m, title, fmt="{:.2f}"):
        b1 = ax.bar(x - w / 2, [before[i][m] for i in ids], w, color=OI["grey"], label="before (unfiltered)")
        b2 = ax.bar(x + w / 2, [after[i][m] for i in ids], w, color=OI["blue"], label="after (filtered)")
        for bb in (b1, b2):
            for b_ in bb:
                ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), fmt.format(b_.get_height()),
                        ha="center", va="bottom", fontsize=6.5, rotation=90 if m in ("FP", "FN") else 0)
        ax.set_xticks(x); ax.set_xticklabels([f"{i}\n{SETTINGS[i][0]}/{SETTINGS[i][2]}" for i in ids], fontsize=8)
        ax.set_title(title, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[0, 0]); bars(ax, "HOTA", "A  HOTA before vs after"); ax.set_ylim(40, 72)
    ax.axhline(bar, color="black", ls="-.", lw=0.8, label=f"corrected seed mean+2SD ({bar:.2f})")
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    ax = fig.add_subplot(gs[0, 1]); bars(ax, "IDSW", "B  ID switches", "{:.0f}")
    ax = fig.add_subplot(gs[0, 2]); bars(ax, "MOTA", "C  MOTA")
    ax = fig.add_subplot(gs[1, 0]); bars(ax, "FP", "D  False positives", "{:.0f}")
    ax = fig.add_subplot(gs[1, 1]); bars(ax, "FN", "E  False negatives", "{:.0f}")
    ax = fig.add_subplot(gs[1, 2])
    for s_, c in (("42", OI["blue"]), ("1234", OI["sky"]), ("2024", OI["green"])):
        ax.plot([0, 1], [seeds[s_]["S_A"]["HOTA"], seeds[s_]["S_D"]["HOTA"]], "-o", color=c, label=f"seed {s_}")
        for k, rid in enumerate(("S_A", "S_D")):
            ax.annotate(f"{seeds[s_][rid]['HOTA']:.2f}", (k, seeds[s_][rid]["HOTA"]), xytext=(6 if k else -6, 0),
                        textcoords="offset points", ha="left" if k else "right", va="center", fontsize=7, color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["S_A (0.5/30)", "S_D (0.5/20)"]); ax.set_xlim(-0.5, 1.5)
    ax.set_title("F  Corrected HOTA, three training seeds (ckpt_3)", fontsize=10); ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    hdr = ["Setting", "det/trk", "miss"] + ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Prec %",
                                           "Rec %"] + ["Rule (corrected band)"]
    cells = [[i, f"{SETTINGS[i][0]}", f"{SETTINGS[i][2]}"] +
             [f"{after[i][m]:.2f}" if m not in ("IDSW", "FP", "FN") else f"{after[i][m]:d}" for m in M] + [verdict[i]]
             for i in ids]
    tb = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.35)
    for (r_, c_), cl in tb.get_celld().items():
        cl.set_linewidth(0.3)
        if r_ == 0:
            cl.set_text_props(fontweight="bold"); cl.set_facecolor("#eeeeee")
    ax.set_title("G  Corrected results (filtered output). Single validation sequence; not statistically established.",
                 fontsize=10)
    dst = os.path.join(OUT, "final_results_dashboard.png")
    if os.path.exists(dst):
        print(f"REFUSE to overwrite {dst}")
    else:
        fig.savefig(dst, dpi=150); print(f"written {dst}")

if __name__ == "__main__":
    main()
