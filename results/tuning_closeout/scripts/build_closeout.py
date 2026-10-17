"""
build_closeout.py — Hyperparameter-tuning close-out: verify every corrected number against its TrackEval summary,
then write the paper tables (csv/tex/md) and figures (png 300 dpi + pdf) under results/tuning_closeout/.

    python -u results/tuning_closeout/scripts/build_closeout.py [--out <dir>]

Read-only on all inputs. Refuses to overwrite existing outputs. Sources (all corrected protocol unless marked):
  results_after_tuning/final_review/corrected_model_comparison.csv   (21 checkpoints, seq_02)
  results_after_tuning/final_review/corrected_c2_comparison.csv      (3 sequences x C2 off/on)
  results_after_tuning/corrected_sweep_results.csv                   (7 inference settings)
  results_after_tuning/seed_check_results.csv                        (seeds 42/1234/2024, ckpt_3)
  results_after_tuning/before_after_comparison.csv                   (sweep, unfiltered vs filtered)
  archive/pre_tuning_baseline_v1/data/results_registry_v1.csv        (HISTORICAL unfiltered; read only)
"""
import os, sys, csv, argparse, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P = lambda *a: os.path.join(ROOT, *a)
SRC = {"model": P("results_after_tuning", "final_review", "corrected_model_comparison.csv"),
       "c2": P("results_after_tuning", "final_review", "corrected_c2_comparison.csv"),
       "sweep": P("results_after_tuning", "corrected_sweep_results.csv"),
       "seed": P("results_after_tuning", "seed_check_results.csv"),
       "before_after": P("results_after_tuning", "before_after_comparison.csv"),
       "hist": P("archive", "pre_tuning_baseline_v1", "data", "results_registry_v1.csv")}
M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "recall", "precision"]
LOWER = {"IDSW", "FP", "FN"}
OI = {"grey": "#999999", "blue": "#0072B2", "vermilion": "#D55E00", "teal": "#009E73", "pink": "#CC79A7",
      "sky": "#56B4E9", "orange": "#E69F00", "yellow": "#F0E442"}
COL = {"Phase5": OI["grey"], "Arm1": OI["blue"], "Arm2": OI["vermilion"], "assoc_lr0": OI["teal"],
       "lr_half": OI["pink"]}
NAME = {"Phase5": "Phase 5", "Arm1": "Arm 1", "Arm2": "Arm 2", "assoc_lr0": "assoc_lr0", "lr_half": "lr_half"}
FOOT_FIG = ("Val: seq_02_30sec_a (744 frames, 50 tracks, night). Corrected protocol. Single run, seed 42 unless stated. "
            "Not statistically established.")
FOOT_TEX = (r"Val: seq\_02\_30sec\_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. "
            r"Corrected protocol: upstream output filter score $>$ 0.5, area $>$ 100 px.")
W1, W2 = 3.4, 7.0
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))

def summ(p):
    l = [x.split() for x in open(p) if x.strip()]; d = dict(zip(l[0], l[1]))
    return {"HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]), "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]),
            "IDF1": float(d["IDF1"]), "IDSW": float(d["IDSW"]), "FP": float(d["CLR_FP"]), "FN": float(d["CLR_FN"]),
            "recall": float(d["CLR_Re"]), "precision": float(d["CLR_Pr"])}

def num(r):
    return {k: (int(float(r[k])) if k in LOWER else float(r[k])) for k in M}

# ── 1. load and verify ─────────────────────────────────────────────────────────
def verify_and_load():
    bad = 0; n = 0
    def chk(rows, label, path_key="summary_path"):
        nonlocal bad, n
        for r in rows:
            sp = r[path_key]; sp = sp if os.path.isabs(sp) else P(sp)
            s = summ(sp)
            for k in M:
                n += 1
                if abs(float(r[k]) - s[k]) > 0.006:
                    bad += 1; print(f"MISMATCH {label} {r.get('model', r.get('run', r.get('sequence')))} {k}: "
                                    f"csv {r[k]} summary {s[k]}")
    model = read(SRC["model"]); chk(model, "model")
    c2 = read(SRC["c2"]); chk(c2, "c2")
    sweep = read(SRC["sweep"]); chk(sweep, "sweep")
    seed = read(SRC["seed"]); chk(seed, "seed")
    hist = read(SRC["hist"]); chk([r for r in hist], "hist", "source_path")
    ba = read(SRC["before_after"])
    for r in ba:          # before/after must agree with the verified sweep CSVs
        sw = [x for x in sweep if x["run"].endswith(r["setting"])][0]
        for k in M:
            n += 1
            if abs(float(r[f"{k}_after"]) - float(sw[k])) > 0.006:
                bad += 1; print(f"MISMATCH before_after {r['setting']} {k}")
    print(f"VERIFY: {n} values checked against TrackEval summaries / verified CSVs, {bad} mismatches")
    if bad:
        sys.exit(1)
    return model, c2, sweep, seed, hist, ba

# ── table writer ───────────────────────────────────────────────────────────────
def fmt(v, h=""):
    if v is None or v == "":
        return "–"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 1000 else str(v)
    return f"{v:+.2f}" if h.startswith("Δ") else f"{v:.2f}"

def write_table(out, name, header, rows, best=None, bold_rows=None, caption=""):
    base = os.path.join(out, "tables", name)
    for ext in (".csv", ".tex", ".md"):
        if os.path.exists(base + ext):
            print(f"REFUSE to overwrite {base + ext}"); sys.exit(2)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (round(v, 3) if isinstance(v, float) else v) for v in r])
    bv = {}
    for ci, how in (best or {}).items():
        cand = [rows[i][ci] for i in (bold_rows if bold_rows is not None else range(len(rows)))
                if isinstance(rows[i][ci], (int, float))]
        if cand:
            bv[ci] = max(cand) if how == "max" else min(cand)
    def cell(ri, ci, tex):
        v = rows[ri][ci]; s = fmt(v, header[ci])
        isb = ci in bv and isinstance(v, (int, float)) and abs(v - bv[ci]) < 1e-9 and \
            (bold_rows is None or ri in bold_rows)
        if tex:
            s = (s.replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$").replace("★", r"$\star$")
                 .replace("–", "--").replace("Δ", r"$\Delta$").replace("×", r"$\times$").replace("≤", r"$\leq$")
                 .replace(">", r"$>$").replace("<", r"$<$"))
            return rf"\textbf{{{s}}}" if isb else s
        return f"**{s}**" if isb else s
    with open(base + ".md", "w", encoding="utf-8") as f:
        if caption:
            f.write(f"{caption}\n\n")
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for ri in range(len(rows)):
            f.write("| " + " | ".join(cell(ri, ci, False) for ci in range(len(header))) + " |\n")
        f.write("\n*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. "
                "Corrected protocol: upstream output filter score > 0.5, area > 100 px.*\n")
    th = lambda h: (h.replace("_", r"\_").replace("%", r"\%").replace("Δ", r"$\Delta$").replace("★", r"$\star$")
                    .replace("±", r"$\pm$").replace("×", r"$\times$"))
    with open(base + ".tex", "w", encoding="utf-8") as f:
        f.write("% requires \\usepackage{booktabs}\n\\begin{table}[t]\n\\centering\n\\small\n")
        if caption:
            f.write(f"\\caption{{{th(caption)}}}\n")
        f.write("\\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}\n\\toprule\n")
        f.write(" & ".join(th(h) for h in header) + " \\\\\n\\midrule\n")
        for ri in range(len(rows)):
            f.write(" & ".join(cell(ri, ci, True) for ci in range(len(header))) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\par\\smallskip\\footnotesize\\emph{{{FOOT_TEX}}}\n\\end{{table}}\n")
    print(f"table {name}: {len(rows)} rows")

# ── 2. tables ──────────────────────────────────────────────────────────────────
def steps(model, ck):
    return 0 if model == "Phase5" else (10112 if model == "Arm2" else 2528) * (ck + 1)

def tables(out, model, c2, sweep, seed, hist):
    get = lambda m, ck: [r for r in model if r["model"] == m and r["checkpoint"] == f"checkpoint_{ck}.pth"][0]
    best = {m: max([r for r in model if r["model"] == m], key=lambda r: float(r["HOTA"]))
            for m in ("Phase5", "Arm1", "Arm2", "assoc_lr0", "lr_half")}
    bc = lambda r: int(r["checkpoint"].split("_")[1].split(".")[0])
    H = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Recall %", "Precision %"]
    BEST = {i + 3: ("min" if k in LOWER else "max") for i, k in enumerate(M)}
    # all runs
    rows = []
    for m, cks in (("Phase5", [4]), ("Arm1", range(5)), ("Arm2", range(5)), ("assoc_lr0", [bc(best["assoc_lr0"])]),
                   ("lr_half", [bc(best["lr_half"])])):
        for ck in cks:
            r = get(m, ck); star = " ★" if r is best[m] else ""
            v = num(r); rows.append([NAME[m], f"ckpt_{ck}{star}", steps(m, ck)] + [v[k] for k in M])
    write_table(out, "table_corrected_all_runs", ["Run", "Checkpoint", "Steps"] + H, rows, BEST,
                caption="Corrected protocol: every Phase 5, Arm 1 and Arm 2 checkpoint, and the best checkpoint of "
                        "assoc_lr0 and lr_half. ★ = highest HOTA of the run.")
    # candidates
    p5 = num(get("Phase5", 4)); rows = []; br = []
    for m in ("Phase5", "Arm1", "Arm2"):
        r = best[m]; v = num(r); star = "" if m == "Phase5" else " ★"
        br.append(len(rows)); rows.append([NAME[m], f"ckpt_{bc(r)}{star}"] + [v[k] for k in M])
        if m != "Phase5":
            rows.append([f"Δ vs Phase 5", ""] + [round(v[k] - p5[k], 3) if k not in LOWER else v[k] - p5[k]
                                                 for k in M])
    write_table(out, "table_corrected_candidates", ["Model", "Checkpoint"] + H, rows,
                {i + 2: ("min" if k in LOWER else "max") for i, k in enumerate(M)}, bold_rows=br,
                caption="Corrected protocol: candidate checkpoints and difference to Phase 5 "
                        "(positive IDSW/FP/FN = more errors).")
    # seed spread
    sd = {r["seed"]: num(r) for r in seed if r["setting"] == "S_A"}
    M8 = M[:8]; rows = []
    for s in ("42", "1234", "2024"):
        rows.append([f"seed {s}" + (" (Arm 1)" if s == "42" else ""), "ckpt_3"] + [sd[s][k] for k in M8])
    mean = {k: st.mean(sd[s][k] for s in sd) for k in M8}; sdev = {k: st.stdev(sd[s][k] for s in sd) for k in M8}
    rows.append(["mean ± SD", "ckpt_3"] + [f"{mean[k]:.2f} ± {sdev[k]:.2f}" for k in M8])
    rows.append(["noise bar (mean + 2×SD)", "ckpt_3", round(mean["HOTA"] + 2 * sdev["HOTA"], 3)] + [None] * 7)
    write_table(out, "table_seed_spread", ["Seed", "Checkpoint"] + H[:8], rows,
                {i + 2: ("min" if k in LOWER else "max") for i, k in enumerate(M8)}, bold_rows=[0, 1, 2],
                caption="Corrected protocol: Arm 1 configuration, three training seeds at checkpoint_3 (n = 3; rough "
                        "estimate). Corrected seed values exist only for checkpoint_3.")
    bar = mean["HOTA"] + 2 * sdev["HOTA"]
    # tuning variants
    rows = [["Phase 5", "ckpt_4"] + [p5[k] for k in M] + ["reference", "pre-IOTD baseline"],
            ["Arm 1 (seed mean, n=3)", "ckpt_3"] + [round(mean[k], 3) for k in M8] +
            [round(st.mean(sd[s]["recall"] for s in sd), 3), round(st.mean(sd[s]["precision"] for s in sd), 3)] +
            ["reference", "default configuration"]]
    for m, why in (("assoc_lr0", "best {:.2f} < bar {:.2f}; collapses at ckpt_1 (IDSW 21)"),
                   ("lr_half", "best {:.2f} < bar {:.2f}; single run; collapses at ckpt_1 (IDSW 21)")):
        r = best[m]; v = num(r)
        rows.append([m, f"ckpt_{bc(r)}"] + [v[k] for k in M] + ["NO", why.format(v["HOTA"], bar)])
    write_table(out, "table_tuning_variants", ["Run", "Checkpoint"] + H + ["Passed rule", "Reason"], rows,
                {i + 2: ("min" if k in LOWER else "max") for i, k in enumerate(M)},
                caption=f"Corrected protocol: training variants against the Arm 1 seed noise bar "
                        f"(mean + 2×SD at ckpt_3 = {bar:.2f}).")
    # sweep
    sa = [r for r in sweep if r["run"].endswith("S_A")][0]; rows = []
    for r in sweep:
        rid = r["run"][-3:]; v = num(r)
        ok = v["HOTA"] > bar and v["IDSW"] <= int(float(sa["IDSW"])) and v["HOTA"] - float(sa["HOTA"]) >= 1.0
        dec = "reference (retained)" if rid == "S_A" else ("PASS" if ok else "FAIL")
        rows.append([rid, f"{float(r['det_thresh'])}/{float(r['track_thresh'])}/{int(float(r['miss_tol']))}",
                     v["HOTA"], v["FP"], v["FN"], v["IDSW"], v["recall"], v["precision"], dec])
    write_table(out, "table_inference_sweep", ["Setting", "det/track/miss", "HOTA", "FP", "FN", "IDSW", "Recall %",
                                               "Precision %", "Decision rule"], rows,
                {2: "max", 3: "min", 4: "min", 5: "min", 6: "max", 7: "max"},
                caption=f"Corrected protocol: inference settings on Arm 1 ckpt_3. Rule: HOTA $>$ {bar:.2f}, "
                        f"IDSW ≤ S_A and ≥ 1 HOTA over S_A.")
    # C2
    rows = []
    for s, l in (("seq_01_1min", "night"), ("seq_02_30sec_a", "night"), ("seq_03_30sec_b", "dusk")):
        a = {r["c2"]: r for r in c2 if r["sequence"] == s}
        for onoff in ("off", "on"):
            v = num(a[onoff])
            rows.append([s, l, onoff] + [v[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")] +
                        [float(a[onoff]["tau_mean"])])
        d = {k: num(a["on"])[k] - num(a["off"])[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")}
        rows.append([s, l, "Δ on − off"] + [round(d[k], 3) if k != "IDSW" else d[k]
                                           for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")] + [None])
    write_table(out, "table_c2_results", ["Sequence", "Light", "C2", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW",
                                          "mean τ"], rows, None,
                caption="Corrected protocol: Phase 5 ckpt_4 with C2 off (fixed τ = 0.5) and on (adapter epoch 4) on "
                        "all three IOTD sequences; seq_01 and seq_03 are IOTD training sequences.")
    # protocol comparison
    hrow = lambda run, ck: [r for r in hist if r["run"] == run and r["checkpoint"] == f"ckpt_{ck}"
                            and float(r["det_thresh"]) == 0.5 and r["sequence"] == "seq_02_30sec_a"
                            and r["c2"] == "off" and int(r["miss_tol"]) == 30][0]
    rows = []
    for m, ck in (("Phase5", 4), ("Arm1", 3)):
        h = num(hrow(m, ck)); c = num(get(m, ck))
        rows.append([NAME[m], f"ckpt_{ck}", "HISTORICAL (unfiltered)"] + [h[k] for k in M])
        rows.append([NAME[m], f"ckpt_{ck}", "CORRECTED (filtered)"] + [c[k] for k in M])
    write_table(out, "table_protocol_comparison", ["Model", "Checkpoint", "Protocol"] + H, rows, None,
                caption="The same checkpoints scored under two DIFFERENT protocols. The difference is caused by the "
                        "output filter, not by any change to the model; it is not an improvement.")
    return best, bar, mean, sdev, sd, get

# ── 3. figures ─────────────────────────────────────────────────────────────────
def save(fig, out, name, footer=FOOT_FIG):
    for ext in ("png", "pdf"):
        p = os.path.join(out, "figures", f"{name}.{ext}")
        if os.path.exists(p):
            print(f"REFUSE to overwrite {p}"); sys.exit(2)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.text(0.01, 0.005, footer, fontsize=6, ha="left", va="bottom", color="#444444", wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=300)
    plt.close(fig); print(f"figure {name}")

def figures(out, model, c2, sweep, seed, hist, best, bar, mean, sdev, sd, get):
    bc = lambda r: int(r["checkpoint"].split("_")[1].split(".")[0])
    p5 = float(get("Phase5", 4)["HOTA"])
    # HOTA vs steps
    fig, ax = plt.subplots(figsize=(W2, 3.4))
    ax.axhline(p5, color=COL["Phase5"], ls="--", lw=1.2, label=f"Phase 5 ckpt_4 ({p5:.2f})")
    for m in ("Arm1", "Arm2", "assoc_lr0", "lr_half"):
        x = [steps(m, i) for i in range(5)]; y = [float(get(m, i)["HOTA"]) for i in range(5)]
        ax.plot(x, y, "-o", color=COL[m], ms=3.5, lw=1.3, label=NAME[m])
        b = bc(best[m]); ax.plot(steps(m, b), float(best[m]["HOTA"]), "*", ms=13, color=COL[m], mec="black",
                                 mew=0.6, zorder=5)
        if m in ("Arm1", "Arm2"):
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 5), textcoords="offset points", ha="center",
                            fontsize=7, color=COL[m])
    for drop, m, txt, ha, y in ((10112, "Arm1", "LR drop (Arm 1) \nafter 10,112 steps ", "right", 67.6),
                                (40448, "Arm2", "LR drop (Arm 2) \nafter 40,448 steps ", "right", 67.6)):
        ax.axvline(drop, color=COL[m], ls=":", lw=1)
        ax.text(drop, y, txt, color=COL[m], fontsize=7, ha=ha, va="top")
    a1 = float(get("Arm1", 3)["HOTA"]); a2 = float(get("Arm2", 0)["HOTA"])
    ax.plot([10112, 10112], [a2, a1], color="black", ls="--", lw=0.7)
    ax.annotate(f"equal steps (10,112): Arm 1 ckpt_3 − Arm 2 ckpt_0 = {a1 - a2:+.2f} HOTA", xy=(10112, 55.0),
                xytext=(23000, 64.5), textcoords="data", fontsize=7, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
    ax.plot([], [], "*", color="grey", mec="black", ms=10, ls="none", label="best checkpoint of run")
    ax.set_xlabel("Optimizer steps"); ax.set_ylabel("HOTA (corrected)"); ax.set_ylim(46, 68)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.2)
    save(fig, out, "fig_hota_vs_steps_corrected")
    # seed spread
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    xs = [steps("Arm1", i) for i in range(5)]
    ax.plot(xs, [float(get("Arm1", i)["HOTA"]) for i in range(5)], "-o", lw=0.9, ms=3, color=OI["blue"],
            label="seed 42 (Arm 1), all checkpoints")
    for s, c in (("1234", OI["sky"]), ("2024", OI["teal"])):
        ax.plot(10112, sd[s]["HOTA"], "o", ms=6, color=c, label=f"seed {s} (ckpt_3 only)")
    m_, s_ = mean["HOTA"], sdev["HOTA"]
    ax.fill_between([10112 - 900, 10112 + 900], m_ - s_, m_ + s_, color=OI["blue"], alpha=0.15,
                    label="mean ± 1 SD at ckpt_3 (n = 3)")
    ax.plot([10112 - 900, 10112 + 900], [m_, m_], color=OI["blue"], lw=3, label=f"seed mean {m_:.2f}")
    ax.axhline(m_ + 2 * s_, color="black", ls=":", lw=1, label=f"noise bar mean + 2 SD = {m_ + 2 * s_:.2f}")
    ax.axhline(p5, color=COL["Phase5"], ls="--", lw=1.2, label=f"Phase 5 ({p5:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}\nckpt_{i}" for i, x in enumerate(xs)])
    ax.set_xlabel("Optimizer steps (Arm 1 configuration)"); ax.set_ylabel("HOTA (corrected)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.27))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.42)
    save(fig, out, "fig_seed_spread", FOOT_FIG.replace("Single run, seed 42 unless stated",
                                                        "Seeds 1234 and 2024 were re-scored at ckpt_3 only"))
    # protocol comparison
    hrow = lambda run, ck: [r for r in hist if r["run"] == run and r["checkpoint"] == f"ckpt_{ck}"
                            and float(r["det_thresh"]) == 0.5 and r["sequence"] == "seq_02_30sec_a"
                            and r["c2"] == "off" and int(r["miss_tol"]) == 30][0]
    fig, ax = plt.subplots(figsize=(W1, 2.9)); x = np.arange(2)
    for k, (lab, vals, kw) in enumerate((
            ("HISTORICAL (unfiltered)", [float(hrow("Phase5", 4)["HOTA"]), float(hrow("Arm1", 3)["HOTA"])],
             dict(fill=False, hatch="///")),
            ("CORRECTED (filtered)", [p5, float(get("Arm1", 3)["HOTA"])], dict()))):
        for i, (m, v) in enumerate(zip(("Phase5", "Arm1"), vals)):
            b = ax.bar(i + (k - 0.5) * 0.38, v, 0.38, edgecolor=COL[m], color=COL[m], lw=1.2, **kw)
            ax.text(i + (k - 0.5) * 0.38, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(["Phase 5 ckpt_4", "Arm 1 ckpt_3"]); ax.set_ylabel("HOTA")
    ax.set_ylim(0, 75)
    ax.legend(handles=[Patch(fill=False, hatch="///", edgecolor="black", label="HISTORICAL (unfiltered)"),
                       Patch(color="black", label="CORRECTED (filtered)")], frameon=False, fontsize=7,
              loc="upper left")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.22)
    fig.text(0.5, 0.09, "Different protocols — not a model improvement.", ha="center", fontsize=7)
    save(fig, out, "fig_protocol_comparison")
    # inference sweep
    ids = [r["run"][-3:] for r in sweep]; sa = [r for r in sweep if r["run"].endswith("S_A")][0]
    fig, axes = plt.subplots(2, 1, figsize=(W1, 4.4), sharex=True)
    for ax, k in zip(axes, ("HOTA", "FP")):
        vals = [float(r[k]) for r in sweep]
        bars = ax.bar(ids, vals, color=[OI["orange"] if i == "S_A" else COL["Arm1"] for i in ids])
        ax.axhline(float(get("Phase5", 4)[k]), color=COL["Phase5"], ls="--", lw=1, label="Phase 5")
        ax.axhline(float(sa[k]), color=OI["orange"], ls=":", lw=1, label="S_A (retained)")
        if k == "HOTA":
            ax.axhline(bar, color="black", ls="-.", lw=0.8, label=f"noise bar {bar:.2f}"); ax.set_ylim(50, 70)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}" if k == "HOTA" else f"{v:.0f}", ha="center",
                    va="bottom", fontsize=6.5)
        ax.set_ylabel(k)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    axes[1].set_xticks(range(len(ids)))
    axes[1].set_xticklabels([f"{i}\n{float(r['det_thresh'])}/{int(float(r['miss_tol']))}" for i, r in zip(ids, sweep)],
                            fontsize=7)
    axes[1].set_xlabel("Setting (det=track threshold / miss tolerance)")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.2, hspace=0.12)
    save(fig, out, "fig_inference_sweep", FOOT_FIG.replace("Single run, seed 42 unless stated",
                                                            "Arm 1 ckpt_3, single run each"))
    # C2
    seqs = (("seq_01_1min", "night"), ("seq_02_30sec_a", "night"), ("seq_03_30sec_b", "dusk"))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0)); x = np.arange(3)
    for ax, k in zip(axes, ("HOTA", "IDSW")):
        for j, onoff in enumerate(("off", "on")):
            vals = [float([r for r in c2 if r["sequence"] == s and r["c2"] == onoff][0][k]) for s, _ in seqs]
            b = ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=OI["grey"] if onoff == "off" else OI["orange"],
                       label=f"C2 {onoff}")
            for bb, v in zip(b, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}" if k == "HOTA" else f"{v:.0f}", ha="center",
                        va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"{s.split('_')[1]}\n({l})" for s, l in seqs]); ax.set_ylabel(k)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    taus = []
    for s, _ in seqs:
        r = [r for r in c2 if r["sequence"] == s and r["c2"] == "on"][0]
        taus.append(f"seq_{s.split('_')[1]} τ {float(r['tau_min']):.3f}–{float(r['tau_max']):.3f} "
                    f"(mean {float(r['tau_mean']):.3f})")
    axes[0].legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.115, "C2 on (adapter epoch 4): " + ";  ".join(taus) + ".\nC2 off: τ = 0.5 fixed.", ha="center",
             va="center", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.3, wspace=0.25)
    save(fig, out, "fig_c2_comparison", FOOT_FIG.replace("Val: seq_02_30sec_a (744 frames, 50 tracks, night). ",
                                                          "Phase 5 ckpt_4 on all three IOTD sequences. "))
    # FP / FN
    order = ("Phase5", "Arm1", "Arm2", "assoc_lr0", "lr_half")
    fig, ax = plt.subplots(figsize=(W1, 2.9)); x = np.arange(len(order))
    for j, (k, hatch) in enumerate((("FP", ""), ("FN", "..."))):
        vals = [float(best[m][k]) for m in order]
        b = ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=[COL[m] for m in order], hatch=hatch,
                   edgecolor="white" if not hatch else "black", lw=0.4)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([f"{NAME[m]}\nckpt_{bc(best[m])}" for m in order], fontsize=6.5)
    ax.set_ylabel("Count"); ax.set_ylim(0, 1400)
    ax.legend(handles=[Patch(color="#777777", label="FP (solid)"),
                       Patch(facecolor="#bbbbbb", hatch="...", edgecolor="black", label="FN (dotted)")],
              frameon=False, fontsize=7, loc="upper right")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.25)
    save(fig, out, "fig_fp_fn_corrected")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.dirname(HERE))
    out = os.path.abspath(ap.parse_args().out)
    model, c2, sweep, seed, hist, ba = verify_and_load()
    best, bar, mean, sdev, sd, get = tables(out, model, c2, sweep, seed, hist)
    figures(out, model, c2, sweep, seed, hist, best, bar, mean, sdev, sd, get)
    # numbers for the narrative
    print("\n== numbers for narrative ==")
    hv = read(P("results_after_tuning", "final_review", "historical_vs_corrected.csv"))
    g = [float(r["HOTA_corrected"]) - float(r["HOTA_historical_unfiltered"]) for r in hv]
    gs = [float(r["HOTA_after"]) - float(r["HOTA_before"]) for r in ba]
    print(f"filter HOTA gain: 21 model checkpoints {min(g):+.2f} to {max(g):+.2f}; 7 sweep settings {min(gs):+.2f} to "
          f"{max(gs):+.2f}")
    print(f"seed ckpt_3 HOTA: {[sd[s]['HOTA'] for s in sd]} mean {mean['HOTA']:.3f} SD {sdev['HOTA']:.3f} bar {bar:.3f}")
    a5 = float(get("Phase5", 4)["AssA"])
    for m in ("Arm1", "assoc_lr0", "lr_half", "Arm2"):
        a = [float(get(m, i)["AssA"]) for i in range(5)]
        print(f"AssA {m}: {a}  vs Phase 5 {a5}: diff range {min(a)-a5:+.2f} .. {max(a)-a5:+.2f}")
    for i in range(5):
        print(f"ckpt_{i}: Arm1 {float(get('Arm1', i)['HOTA']):.2f} Arm2 {float(get('Arm2', i)['HOTA']):.2f}  "
              f"Arm2 recall {float(get('Arm2', i)['recall']):.2f} prec {float(get('Arm2', i)['precision']):.2f} "
              f"FP {get('Arm2', i)['FP']}  | lr_half {float(get('lr_half', i)['HOTA']):.2f} FN {get('lr_half', i)['FN']}"
              f"  Arm1 FN {get('Arm1', i)['FN']}  assoc {float(get('assoc_lr0', i)['HOTA']):.2f}")

if __name__ == "__main__":
    main()
