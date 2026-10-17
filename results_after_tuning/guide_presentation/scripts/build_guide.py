"""
build_guide.py — guide presentation package (before vs after tuning). Read-only on all inputs; writes only under
results_after_tuning/guide_presentation/ and refuses to overwrite. Every number is read from a file and checked
against its TrackEval summary; the expected values from the brief are compared and mismatches printed.

    python -u results_after_tuning/guide_presentation/scripts/build_guide.py
"""
import os, sys, csv, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(OUT, "..", ".."))
P = lambda *a: os.path.join(ROOT, *a)
SRC = {"model": P("results_after_tuning", "final_review", "corrected_model_comparison.csv"),
       "c2": P("results_after_tuning", "final_review", "corrected_c2_comparison.csv"),
       "seed": P("results_after_tuning", "seed_check_results.csv"),
       "seed_table": P("results", "tuning_closeout", "tables", "table_seed_spread.csv"),
       "hist": P("archive", "pre_tuning_baseline_v1", "data", "results_registry_v1.csv")}
M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "recall", "precision"]
INT = {"IDSW", "FP", "FN"}
OI = {"grey": "#999999", "blue": "#0072B2", "vermilion": "#D55E00", "teal": "#009E73", "pink": "#CC79A7",
      "orange": "#E69F00", "sky": "#56B4E9"}
COL = {"Phase5": OI["grey"], "Arm1": OI["blue"], "Arm2": OI["vermilion"], "assoc_lr0": OI["teal"],
       "lr_half": OI["pink"]}
FOOT = ("Corrected protocol: upstream output filter score > 0.5, area > 100 px.\n"
        "Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run, seed 42. Not statistically established.")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
EXPECTED = [  # from the brief; compared, never used
    ("Phase5", 4, "HIST", dict(HOTA=50.51, DetA=43.88, AssA=58.49, MOTA=29.10, IDF1=68.84, IDSW=1, FP=2117, FN=847)),
    ("Arm1", 3, "HIST", dict(HOTA=50.47, DetA=46.96, AssA=54.51, MOTA=29.63, IDF1=70.23, IDSW=9, FP=2629, FN=305)),
    ("Arm2", 1, "HIST", dict(HOTA=48.17, DetA=45.54, AssA=51.14, MOTA=29.84, IDF1=68.64, IDSW=12, FP=2432, FN=490)),
    ("Phase5", 4, "CORR", dict(HOTA=57.74, FP=940, FN=1097)),
    ("Arm1", 3, "CORR", dict(HOTA=64.12, DetA=60.54, AssA=68.21, MOTA=68.15, IDF1=84.64, IDSW=3, FP=841, FN=488)),
    ("Arm2", 1, "CORR", dict(HOTA=61.60, FP=600, FN=758)),
]

def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))

def summ(p):
    l = [x.split() for x in open(p) if x.strip()]; d = dict(zip(l[0], l[1]))
    return {"HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]), "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]),
            "IDF1": float(d["IDF1"]), "IDSW": float(d["IDSW"]), "FP": float(d["CLR_FP"]), "FN": float(d["CLR_FN"]),
            "recall": float(d["CLR_Re"]), "precision": float(d["CLR_Pr"])}

def num(r):
    return {k: (int(float(r[k])) if k in INT else float(r[k])) for k in M}

def load():
    bad = n = 0
    def chk(rows, key):
        nonlocal bad, n
        for r in rows:
            p = r[key]; p = p if os.path.isabs(p) else P(p); s = summ(p)
            for k in M:
                n += 1
                if abs(float(r[k]) - s[k]) > 0.006:
                    bad += 1; print("MISMATCH vs summary", p, k, r[k], s[k])
    model, c2, seed, hist = read(SRC["model"]), read(SRC["c2"]), read(SRC["seed"]), read(SRC["hist"])
    chk(model, "summary_path"); chk(c2, "summary_path"); chk(seed, "summary_path")
    hsel = [r for r in hist if float(r["det_thresh"]) == 0.5 and r["sequence"] == "seq_02_30sec_a" and r["c2"] == "off"
            and int(r["miss_tol"]) == 30 and r["run"] in ("Phase5", "Arm1", "Arm2")]
    chk(hsel, "source_path")
    print(f"VERIFY: {n} values checked against TrackEval summaries, {bad} mismatches")
    if bad:
        sys.exit(1)
    corr = {(r["model"], int(r["checkpoint"].split("_")[1].split(".")[0])): num(r) for r in model}
    histd = {(r["run"], int(r["checkpoint"].split("_")[1])): num(r) for r in hsel}
    seeds = {r["seed"]: num(r) for r in seed if r["setting"] == "S_A"}
    hv = [seeds[s]["HOTA"] for s in ("42", "1234", "2024")]
    mean, sd = st.mean(hv), st.stdev(hv); bar = mean + 2 * sd
    t = read(SRC["seed_table"]); tb = [r for r in t if r["Seed"].startswith("noise bar")][0]
    print(f"noise bar: computed {bar:.3f}; results/tuning_closeout table_seed_spread {float(tb['HOTA']):.3f} "
          f"{'OK' if abs(bar - float(tb['HOTA'])) < 0.006 else 'MISMATCH'}")
    print("\n== expected values from the brief vs files ==")
    for run, ck, prot, exp in EXPECTED:
        src = histd if prot == "HIST" else corr
        got = src[(run, ck)]
        for k, v in exp.items():
            ok = abs(got[k] - v) <= (0.006 if k not in INT else 0)
            if not ok:
                print(f"  EXPECTED-MISMATCH {run} ckpt_{ck} {prot} {k}: brief {v}  file {got[k]}  -> file value used")
    print("  (all other expected values match)")
    return corr, histd, seeds, mean, sd, bar, c2

FOOT_NARROW = ("Corrected protocol: upstream output filter\nscore > 0.5, area > 100 px.\n"
               "Val: seq_02_30sec_a (744 frames, 50 tracks, night).\nSingle run, seed 42. Not statistically established.")

def save(fig, name, footer=FOOT, pdf=True):
    for ext in (("png", "pdf") if pdf else ("png",)):
        p = os.path.join(OUT, "figures", f"{name}.{ext}")
        if os.path.exists(p):
            print(f"REFUSE to overwrite {p}"); sys.exit(2)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    if footer:
        fig.text(0.01, 0.01, footer, fontsize=8, ha="left", va="bottom", color="#444444")
    for ext in (("png", "pdf") if pdf else ("png",)):
        fig.savefig(os.path.join(OUT, "figures", f"{name}.{ext}"), dpi=300 if pdf else 150)
    plt.close(fig); print(f"figure {name}")

STEPS = lambda m, ck: 0 if m == "Phase5" else (10112 if m == "Arm2" else 2528) * (ck + 1)

def figures(corr, histd, seeds, mean, sd, bar, c2):
    best = {m: max([k for k in corr if k[0] == m], key=lambda k: corr[k]["HOTA"]) for m in
            ("Phase5", "Arm1", "Arm2", "assoc_lr0", "lr_half")}
    p5 = corr[("Phase5", 4)]; a1 = corr[("Arm1", 3)]; a2 = corr[best["Arm2"]]
    # fig01 HOTA journey
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(p5["HOTA"], color=COL["Phase5"], ls="--", lw=1.3)
    ax.plot(0, p5["HOTA"], "o", ms=11, color=COL["Phase5"], mec="black", label=f"Phase 5 baseline ({p5['HOTA']:.2f})")
    xs = [STEPS("Arm1", i) for i in range(5)]; ys = [corr[("Arm1", i)]["HOTA"] for i in range(5)]
    ax.plot(xs, ys, "-o", color=COL["Arm1"], lw=2, ms=6, label="Arm 1 (real IOTD), ckpt_0–4")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9,
                    color=COL["Arm1"])
    ax.plot(STEPS("Arm1", 3), a1["HOTA"], "*", ms=22, color=COL["Arm1"], mec="black", zorder=5)
    x2 = [STEPS("Arm2", i) for i in range(5)]; y2 = [corr[("Arm2", i)]["HOTA"] for i in range(5)]
    ax.plot(x2[:2], y2[:2], "-o", color=COL["Arm2"], lw=2, ms=6, label="Arm 2 (real + augmented)")
    ax.plot(x2[1:], y2[1:], "--o", color=COL["Arm2"], lw=1.3, ms=4, alpha=0.7, label="Arm 2 after best (not chosen)")
    for x, y in zip(x2, y2):
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9,
                    color=COL["Arm2"])
    ax.annotate(f"Chosen: Arm 1 ckpt_3 = {a1['HOTA']:.2f}\n{a1['HOTA'] - p5['HOTA']:+.1f} HOTA over baseline",
                xy=(STEPS("Arm1", 3), a1["HOTA"]), xytext=(14500, 66.6), fontsize=12, fontweight="bold",
                color=COL["Arm1"], arrowprops=dict(arrowstyle="->", color=COL["Arm1"], lw=1.2))
    ax.text(0.99, 0.03, "All values use the corrected protocol.\nEarlier numbers (archive/) used a different\n"
            "protocol and are not directly comparable.", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#fff8e1", edgecolor="#bbbbbb"))
    ax.set_xlabel("Optimizer steps (Arm 1: 2,528 per epoch; Arm 2: 10,112 per epoch)")
    ax.set_ylabel("HOTA"); ax.set_ylim(49, 69); ax.set_xlim(-2500, 53000)
    sec = ax.secondary_xaxis("top", functions=(lambda s: s / 2528, lambda e: e * 2528))
    sec.set_xlabel("Arm 1 epochs (2,528 steps each)", fontsize=9); sec.set_xticks(range(0, 6))
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.22)
    save(fig, "fig01_hota_journey")
    # fig02 before and after
    fig, ax = plt.subplots(figsize=(3.4, 4.6)); x = np.arange(2)
    for i, m in enumerate(("Phase5", "Arm1")):
        ck = 4 if m == "Phase5" else 3
        h, c = histd[(m, ck)]["HOTA"], corr[(m, ck)]["HOTA"]
        ax.bar(i - 0.2, h, 0.38, color=COL[m], alpha=0.35, hatch="//", edgecolor=COL[m])
        ax.bar(i + 0.2, c, 0.38, color=COL[m])
        ax.text(i - 0.2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["Phase 5\nckpt_4", "Arm 1\nckpt_3"]); ax.set_ylabel("HOTA"); ax.set_ylim(0, 80)
    ax.legend(handles=[Patch(facecolor="#cccccc", hatch="//", edgecolor="#777777", label="HISTORICAL"),
                       Patch(color="#555555", label="CORRECTED")], frameon=False, fontsize=9, loc="upper left")
    fig.text(0.5, 0.265, "HISTORICAL bars use an earlier evaluation\nprotocol (unfiltered). CORRECTED bars use the\n"
             "upstream output filter. These are the same\nmodels; the protocol changed.", ha="center", va="top",
             fontsize=8.5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.48)
    save(fig, "fig02_before_after_corrected", FOOT_NARROW)
    # fig03 FP / FN
    cand = [("Phase5", 4, "Phase 5\nckpt_4"), ("Arm1", 3, "Arm 1\nckpt_3"), ("Arm2", best["Arm2"][1], "Arm 2\nckpt_1")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0))
    for ax, k in zip(axes, ("FP", "FN")):
        v = [corr[(m, ck)][k] for m, ck, _ in cand]
        b = ax.bar(range(3), v, color=[COL[m] for m, _, _ in cand])
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:,}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, _, l in cand])
        ax.set_title("False positives (FP)" if k == "FP" else "False negatives (FN)")
        ax.set_ylim(0, max(v) * 1.2)
    dfp = 100 * (a1["FP"] - p5["FP"]) / p5["FP"]; dfn = 100 * (a1["FN"] - p5["FN"]) / p5["FN"]
    print(f"fig03 percentages: FP {p5['FP']} -> {a1['FP']} = {dfp:+.1f}%; FN {p5['FN']} -> {a1['FN']} = {dfn:+.1f}%")
    fig.text(0.5, 0.2, f"Arm 1 reduces FP by {-dfp:.1f}% and FN by {-dfn:.1f}% vs baseline "
             f"(FP {p5['FP']:,} → {a1['FP']:,}; FN {p5['FN']:,} → {a1['FN']:,}).", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.36, wspace=0.3)
    save(fig, "fig03_fp_fn_comparison")
    # fig04 identity
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 4.4), gridspec_kw={"width_ratios": [2, 1.2]})
    ax = axes[0]; w = 0.27
    for j, (m, ck, lab) in enumerate(cand):
        vals = [corr[(m, ck)]["IDF1"], corr[(m, ck)]["AssA"]]
        b = ax.bar(np.arange(2) + (j - 1) * w, vals, w, color=COL[m], label=lab.replace("\n", " "))
        for bb, vv in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=7,
                    rotation=90)
    ax.set_xticks(range(2)); ax.set_xticklabels(["IDF1", "AssA"]); ax.set_ylim(0, 100); ax.set_ylabel("%")
    ax = axes[1]
    v = [corr[(m, ck)]["IDSW"] for m, ck, _ in cand]
    b = ax.bar(range(3), v, color=[COL[m] for m, _, _ in cand])
    for bb, vv in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(["P5", "A1", "A2"]); ax.set_ylabel("IDSW (count)")
    ax.set_ylim(0, max(v) + 2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=1, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.255, " / ".join(f"{l.split(chr(10))[0]}: {corr[(m, ck)]['IDSW']} IDSW" for m, ck, l in cand),
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.8, bottom=0.4, wspace=0.55)
    save(fig, "fig04_identity_metrics", FOOT_NARROW)
    # fig05 what was tried
    rows = [("Arm 1 ckpt_3 ★ CHOSEN", a1["HOTA"], COL["Arm1"], ""),
            (f"Arm 1 seed mean (n=3)", mean, OI["sky"], "3 seeds; sets the noise bar"),
            (f"Arm 2 ckpt_{best['Arm2'][1]}", a2["HOTA"], COL["Arm2"], "lower than real-only at every checkpoint"),
            (f"lr_half ckpt_{best['lr_half'][1]}", corr[best["lr_half"]]["HOTA"], COL["lr_half"],
             "single run, below noise bar, unstable after epoch 1"),
            (f"assoc_lr0 ckpt_{best['assoc_lr0'][1]}", corr[best["assoc_lr0"]]["HOTA"], COL["assoc_lr0"],
             "below noise bar; AssA not improved"),
            ("Phase 5 ckpt_4 BASELINE", p5["HOTA"], COL["Phase5"], "baseline (no IOTD data)")]
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.72)
    for yi, (lab, v, c, why) in zip(y, rows):
        ax.text(v + 0.15, yi, f"{v:.2f}", va="center", fontsize=10, fontweight="bold" if "CHOSEN" in lab else None)
        if why:
            ax.text(50.25, yi, why, va="center", fontsize=8.5, color="white", style="italic", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(p5["HOTA"], color=COL["Phase5"], ls="--", lw=1.2)
    ax.axvline(bar, color="black", ls=":", lw=1.2)
    ax.text(bar, len(rows) - 0.45, f"noise bar\n{bar:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(p5["HOTA"], len(rows) - 0.45, f"Phase 5\n{p5['HOTA']:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(50, 68.5); ax.set_ylim(-0.7, len(rows) + 0.35); ax.set_xlabel("Best corrected HOTA")
    fig.subplots_adjust(left=0.3, right=0.97, top=0.92, bottom=0.22)
    save(fig, "fig05_what_was_tried")
    # fig06 C2
    seqs = (("seq_01_1min", "night"), ("seq_02_30sec_a", "night"), ("seq_03_30sec_b", "dusk"))
    fig, ax = plt.subplots(figsize=(3.4, 4.6)); x = np.arange(3)
    for j, onoff in enumerate(("off", "on")):
        rs = [[r for r in c2 if r["sequence"] == s and r["c2"] == onoff][0] for s, _ in seqs]
        v = [float(r["HOTA"]) for r in rs]
        b = ax.bar(x + (j - 0.5) * 0.38, v, 0.38, color=OI["grey"] if onoff == "off" else OI["orange"],
                   label=f"C2 {onoff}")
        for bb, vv, r in zip(b, v, rs):
            lab = f"{vv:.1f}" + (f"\nτ {float(r['tau_mean']):.2f}" if onoff == "on" else "")
            ax.text(bb.get_x() + bb.get_width() / 2, vv, lab, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"seq_{s.split('_')[1]}\n{l}" for s, l in seqs]); ax.set_ylabel("HOTA")
    ax.set_ylim(0, 75); ax.legend(frameon=False, fontsize=9, loc="upper right")
    d = {s: float([r for r in c2 if r["sequence"] == s and r["c2"] == "on"][0]["HOTA"]) -
         float([r for r in c2 if r["sequence"] == s and r["c2"] == "off"][0]["HOTA"]) for s, _ in seqs}
    within = abs(d["seq_02_30sec_a"]) <= sd
    note = ("HOTA change on − off:\n" + ", ".join(f"seq_{s.split('_')[1]} {v:+.2f}" for s, v in d.items()) + ".\n" +
            (f"Difference within seed noise band\n(±{sd:.2f} HOTA)." if within else "Difference exceeds seed SD.") +
            "\nPhase 5 ckpt_4; C2 adapter\nnot trained on IOTD.")
    fig.text(0.5, 0.315, note, ha="center", va="top", fontsize=8)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.53)
    print(f"fig06: dHOTA {d}; seq_02 within ±SD {sd:.2f}: {within}")
    save(fig, "fig06_c2_result", "Corrected protocol: upstream output filter\nscore > 0.5, area > 100 px.\n"
                                 "Phase 5 ckpt_4, all three IOTD sequences.\nSingle run. Not statistically established.")
    # dashboard
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("MeMOTR on IOTD — Tuning Results Summary", fontsize=30, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "For guide review. All figures use the corrected evaluation protocol unless labelled "
             "HISTORICAL.", ha="center", fontsize=17, style="italic")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1], hspace=0.06, wspace=0.05, left=0.01, right=0.99, top=0.93,
                          bottom=0.01)
    pos = {"fig01_hota_journey": gs[0, 0:3], "fig05_what_was_tried": gs[0, 3:6], "fig02_before_after_corrected": gs[1, 0],
           "fig03_fp_fn_comparison": gs[1, 1:3], "fig04_identity_metrics": gs[1, 3], "fig06_c2_result": gs[1, 4:6]}
    for k, (n, g) in enumerate(pos.items()):
        ax = fig.add_subplot(g); ax.imshow(mpimg.imread(os.path.join(OUT, "figures", n + ".png"))); ax.axis("off")
        ax.set_title(n.replace("_", " "), fontsize=13, loc="left")
    save(fig, "dashboard_for_guide", None, pdf=False)
    return best

# ── tables ─────────────────────────────────────────────────────────────────────
def fmt(v):
    if v is None or v == "":
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 1000 else str(v)
    return f"{v:.2f}"

def table(name, header, rows, best=None, footer=""):
    base = os.path.join(OUT, "tables", name)
    for ext in (".csv", ".tex", ".md", ".png"):
        if os.path.exists(base + ext):
            print(f"REFUSE to overwrite {base + ext}"); sys.exit(2)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (round(v, 3) if isinstance(v, float) else v) for v in r])
    bv = {}
    for ci, how in (best or {}).items():
        c = [r[ci] for r in rows if isinstance(r[ci], (int, float))]
        if c:
            bv[ci] = max(c) if how == "max" else min(c)
    isb = lambda r, ci: ci in bv and isinstance(r[ci], (int, float)) and abs(r[ci] - bv[ci]) < 1e-9
    esc = lambda s: (s.replace("_", r"\_").replace("%", r"\%").replace("★", r"$\star$").replace(">", r"$>$")
                     .replace("<", r"$<$").replace("—", "--").replace("±", r"$\pm$").replace("×", r"$\times$"))
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"**{fmt(v)}**" if isb(r, i) else fmt(v) for i, v in enumerate(r)) + " |\n")
        f.write(f"\n*{footer}*\n")
    with open(base + ".tex", "w", encoding="utf-8") as f:
        f.write("% requires \\usepackage{booktabs}\n\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}\n\\toprule\n")
        f.write(" & ".join(esc(h) for h in header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(rf"\textbf{{{esc(fmt(v))}}}" if isb(r, i) else esc(fmt(v)) for i, v in enumerate(r))
                    + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\par\\smallskip\\footnotesize\\emph{{{esc(footer)}}}\n\\end{{table}}\n")
    nc, nr = len(header), len(rows)
    fig, ax = plt.subplots(figsize=(min(max(6, 1.25 * nc), 18), 0.5 * (nr + 1) + 0.9)); ax.axis("off")
    tb = ax.table(cellText=[[fmt(v) for v in r] for r in rows], colLabels=header, loc="upper center",
                  cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 1.6)
    for (ri, ci), c in tb.get_celld().items():
        c.set_linewidth(0.4)
        if ri == 0:
            c.set_text_props(fontweight="bold"); c.set_facecolor("#e8eef5")
        elif isb(rows[ri - 1], ci):
            c.set_text_props(fontweight="bold")
    tb.auto_set_column_width(list(range(nc)))
    fig.text(0.01, 0.02, footer, fontsize=10, ha="left", va="bottom", style="italic", wrap=True)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"table {name}: {nr} rows")

def tables(corr, histd, seeds, mean, sd, bar, c2, best):
    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "recall", "precision"]
    H = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Recall %", "Precision %"]
    rows = []
    for m, lab in (("Phase5", "Phase 5"), ("Arm1", "Arm 1"), ("Arm2", "Arm 2"), ("assoc_lr0", "assoc_lr0"),
                   ("lr_half", "lr_half")):
        k = (m, 4) if m == "Phase5" else ((m, 3) if m == "Arm1" else best[m]); v = corr[k]
        rows.append([lab, f"ckpt_{k[1]}" + (" ★" if m == "Arm1" else ""), STEPS(m, k[1])] + [v[c] for c in cols])
    table("table01_main_results", ["Run", "Checkpoint", "Steps"] + H, rows,
          {i + 3: ("min" if c in INT else "max") for i, c in enumerate(cols)},
          "Corrected protocol. Val: seq_02_30sec_a. Best HOTA marked ★. ★ = chosen model (Arm 1 ckpt_3); lr_half ckpt_0 "
          "is 0.14 higher in a single, unstable run inside the seed noise.")
    rows = []
    for m, ck, lab in (("Phase5", 4, "Phase 5 ckpt_4"), ("Arm1", 3, "Arm 1 ckpt_3")):
        for prot, src in (("HISTORICAL", histd), ("CORRECTED", corr)):
            v = src[(m, ck)]
            rows.append([lab, prot] + [v[c] for c in ("HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN")])
    table("table02_before_after", ["Run", "Protocol", "HOTA", "MOTA", "IDF1", "IDSW", "FP", "FN"], rows, None,
          "HISTORICAL = unfiltered (archive/pre_tuning_baseline_v1/). CORRECTED = upstream output filter applied. "
          "Same models, different evaluation tool.")
    c2d = {s: float([r for r in c2 if r["sequence"] == s and r["c2"] == "on"][0]["HOTA"]) -
           float([r for r in c2 if r["sequence"] == s and r["c2"] == "off"][0]["HOTA"])
           for s in ("seq_01_1min", "seq_02_30sec_a", "seq_03_30sec_b")}
    c2on = float([r for r in c2 if r["sequence"] == "seq_02_30sec_a" and r["c2"] == "on"][0]["HOTA"])
    rows = [["Phase 5 (baseline)", "no IOTD data (start point)", corr[("Phase5", 4)]["HOTA"], "reference", "baseline"],
            ["Arm 1 (real only)", "—", corr[("Arm1", 3)]["HOTA"], "reference (default)", "CHOSEN (ckpt_3)"],
            ["Arm 2 (augmented)", "+7,590 augmented frames", corr[best["Arm2"]]["HOTA"], "NO",
             "discarded: lower at every checkpoint"],
            ["assoc_lr0", "query-updater LR = 0", corr[best["assoc_lr0"]]["HOTA"], "NO", "discarded: below noise bar"],
            ["lr_half", "all LRs × 0.5", corr[best["lr_half"]]["HOTA"], "NO",
             "discarded: single run below noise bar, unstable"],
            ["C2 adapter", "scene-conditioned detection threshold (Phase 5 base)", c2on, "NO",
             "no gain: " + ", ".join(f"{v:+.2f}" for v in c2d.values()) + " HOTA on seq_01/02/03"]]
    table("table03_what_was_tried", ["Configuration", "Change from Arm 1", "Best corrected HOTA", "Passed decision rule",
                                     "Outcome"], rows, None,
          f"Decision rule: HOTA > seed mean + 2 SD (= {bar:.2f}) at more than one checkpoint and IDSW does not increase. "
          f"Under the corrected protocol the seed band exists at ckpt_3 only, so each best value is compared with "
          f"{bar:.2f}. C2 HOTA is Phase 5 ckpt_4 with C2 on, seq_02.")

def main():
    if os.path.exists(os.path.join(OUT, "figures")) or os.path.exists(os.path.join(OUT, "tables")):
        print("REFUSE: figures/ or tables/ already exists"); sys.exit(2)
    corr, histd, seeds, mean, sd, bar, c2 = load()
    best = figures(corr, histd, seeds, mean, sd, bar, c2)
    tables(corr, histd, seeds, mean, sd, bar, c2, best)

if __name__ == "__main__":
    main()
