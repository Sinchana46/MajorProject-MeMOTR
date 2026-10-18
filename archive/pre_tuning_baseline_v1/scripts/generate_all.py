"""
generate_all.py — regenerate every figure, table and data file of archive/pre_tuning_baseline_v1 from the saved
evaluation files of the project (TrackEval pedestrian_summary.txt files, tracker outputs, GT, CSV registries).

    python archive/pre_tuning_baseline_v1/scripts/generate_all.py [--out <dir>]

--out defaults to the archive folder itself. Once the archive is frozen (read-only) regenerate into another folder,
e.g. --out D:/tmp/regen, and compare with scripts/verify_archive.py or a diff tool.

Every metric is re-read from its TrackEval summary file and checked against the source CSV it came from
(tolerance 0.01); any mismatch aborts.
"""
from __future__ import annotations
import os, sys, csv, json, argparse, hashlib, statistics as st
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ARCHIVE, "..", ".."))

# ── Sources ────────────────────────────────────────────────────────────────────
SRC = {
    "baseline_registry": "results/pre_tuning_baseline_v1/data/results_registry.csv",
    "stage1": "results/tuning_stage1/stage1_results.csv",
    "stage1_status": "results/tuning_stage1/status.json",
    "sweep": "results/inference_sweep/sweep_results.csv",
    "c2": "results/c2_eval/c2_results.csv",
    "seq_profile": "data/IOTD/seq_profile.csv",
    "gt_seq02": "data/IOTD_val/seq_02_30sec_a/gt/gt.txt",
    "gt_seq02_eval": "results/eval_iotd_val/gt_eval/seq_02_30sec_a/gt/gt.txt",
    "trk_phase5": "results/eval_iotd_val/phase5/seq_02_30sec_a.txt",
    "trk_arm1_ckpt3": "results/eval_iotd_val/arm1_ckpt3/seq_02_30sec_a.txt",
    "trk_arm2_ckpt1": "results/eval_iotd_val/arm2_ckpt1/seq_02_30sec_a.txt",
}
GT_FILES = {"seq_01_1min": "data/IOTD_train_real/seq_01_1min/gt/gt.txt",
            "seq_02_30sec_a": "data/IOTD_val/seq_02_30sec_a/gt/gt.txt",
            "seq_03_30sec_b": "data/IOTD_train_real/seq_03_30sec_b/gt/gt.txt"}
SEQINFO = {"seq_01_1min": "data/IOTD_train_real/seq_01_1min/seqinfo.ini",
           "seq_02_30sec_a": "data/IOTD_val/seq_02_30sec_a/seqinfo.ini",
           "seq_03_30sec_b": "data/IOTD_train_real/seq_03_30sec_b/seqinfo.ini"}

# ── Style ──────────────────────────────────────────────────────────────────────
OI = {"grey": "#999999", "blue": "#0072B2", "vermilion": "#D55E00", "sky": "#56B4E9", "green": "#009E73",
      "orange": "#E69F00", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
C_P5, C_A1, C_A2 = OI["grey"], OI["blue"], OI["vermilion"]
C_SEED = {"42": OI["blue"], "1234": OI["sky"], "2024": OI["green"]}
C_C2 = {"off": OI["grey"], "on": OI["orange"]}
FOOTER = ("Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run, seed 42, threshold 0.5 unless stated. "
          "Not statistically established.")
W1, W2 = 3.4, 7.0
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "font.family": "DejaVu Sans",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "recall", "precision"]
LOWER_BETTER = {"IDSW", "FP", "FN"}
STEPS = [2528, 5056, 7584, 10112, 12640]


def p(rel):
    return os.path.join(ROOT, rel)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

def read_csv(rel):
    with open(p(rel), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_summary(path):
    lines = [l.split() for l in open(path, encoding="utf-8") if l.strip()]
    d = dict(zip(lines[0], lines[1]))
    return {"HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]), "AssA": float(d["AssA"]),
            "MOTA": float(d["MOTA"]), "IDF1": float(d["IDF1"]), "IDSW": int(float(d["IDSW"])),
            "TP": int(float(d["CLR_TP"])), "FP": int(float(d["CLR_FP"])), "FN": int(float(d["CLR_FN"])),
            "recall": float(d["CLR_Re"]), "precision": float(d["CLR_Pr"])}


# ── 1. Registry (every evaluated run, re-read from its summary file) ───────────
REG_COLS = ["run", "checkpoint", "steps", "epochs_completed", "seed", "config", "sequence", "scoring",
            "threshold", "det_thresh", "track_thresh", "miss_tol", "c2", "GT_sha256",
            "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "TP", "FP", "FN", "recall", "precision",
            "source_path", "source_sha256", "registry_source"]

def build_registry():
    rows, problems = [], []
    def add(base, summary_rel, seq, det, trk, miss, c2, gt_sha, src_name):
        sp = p(summary_rel)
        m = read_summary(sp)
        for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"):
            if k in base and base[k] not in ("", None) and abs(float(base[k]) - m[k]) > 0.011:
                problems.append(f"{base.get('run')} {base.get('checkpoint')} {k}: csv={base[k]} summary={m[k]}")
        r = {k: base.get(k, "") for k in ("run", "checkpoint", "steps", "epochs_completed", "seed", "config")}
        r.update(sequence=seq, scoring="S0", threshold=det, det_thresh=det, track_thresh=trk, miss_tol=miss, c2=c2,
                 GT_sha256=gt_sha, source_path=summary_rel.replace(os.sep, "/"), source_sha256=sha(sp),
                 registry_source=src_name, **m)
        rows.append(r)
    gt02 = sha(p(GT_FILES["seq_02_30sec_a"]))
    for b in read_csv(SRC["baseline_registry"]):
        t = float(b["threshold"])
        if b["run"] == "Phase5":           # Phase 5 never saw IOTD: 0 IOTD steps for every row
            b = dict(b, steps="0", epochs_completed="0")
        add(b, b["summary_path"], "seq_02_30sec_a", t, t, 30, "off", b["GT_sha256"], SRC["baseline_registry"])
    for b in read_csv(SRC["stage1"]):
        add(dict(b, run=b["run"]), b["summary_path"], "seq_02_30sec_a", 0.5, 0.5, 30, "off", b["GT_sha256"],
            SRC["stage1"])
    for b in read_csv(SRC["sweep"]):
        add(b, b["summary_path"], "seq_02_30sec_a", float(b["det_thresh"]), float(b["track_thresh"]),
            int(b["miss_tol"]), "off", b["GT_sha256"], SRC["sweep"])
    for b in read_csv(SRC["c2"]):
        base = dict(b, run=f"Phase5_C2{b['c2']}", checkpoint="ckpt_4", steps="0", epochs_completed="0", seed="42",
                    config="configs/train_iotd_real.yaml")
        add(base, b["summary_path"], b["sequence"], 0.5, 0.5, 30, b["c2"], sha(p(GT_FILES[b["sequence"]])),
            SRC["c2"])
    if problems:
        raise SystemExit("REGISTRY MISMATCH:\n" + "\n".join(problems))
    assert all(r["GT_sha256"] == gt02 for r in rows if r["sequence"] == "seq_02_30sec_a")
    return rows

def reg_get(reg, run, ckpt, det=0.5, seq="seq_02_30sec_a", c2="off", miss=30):
    m = [r for r in reg if r["run"] == run and r["checkpoint"] == ckpt and float(r["det_thresh"]) == det
         and r["sequence"] == seq and r["c2"] == c2 and int(r["miss_tol"]) == miss]
    assert len(m) == 1, (run, ckpt, det, seq, c2, len(m))
    return m[0]


# ── 2. Detection outcomes (own per-frame Hungarian matching, IoU >= 0.5) ───────
def load_boxes(path, gt=False):
    d = defaultdict(list)
    for line in open(path):
        q = line.strip().split(",")
        if len(q) < 6:
            continue
        if gt and len(q) >= 7 and float(q[6]) == 0:     # TrackEval drops zero-marked GT rows
            continue
        f = int(float(q[0])); x, y, w, h = map(float, q[2:6])
        d[f].append((x, y, x + w, y + h))
    return d

def iou_matrix(a, b):
    a = np.asarray(a)[:, None, :]; b = np.asarray(b)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-12)

def detection_outcomes(reg):
    gt = load_boxes(p(SRC["gt_seq02_eval"]), gt=True)
    out = []
    for run, ckpt, key in (("Phase5", "ckpt_4", "trk_phase5"), ("Arm1", "ckpt_3", "trk_arm1_ckpt3"),
                           ("Arm2", "ckpt_1", "trk_arm2_ckpt1")):
        trk = load_boxes(p(SRC[key]))
        tp = fp = fn = 0
        for f in sorted(set(gt) | set(trk)):
            g, t = gt.get(f, []), trk.get(f, [])
            if g and t:
                iou = iou_matrix(g, t)
                cost = np.where(iou >= 0.5, 1.0 - iou, 1e6)
                ri, ci = linear_sum_assignment(cost)
                m = int(sum(1 for i, j in zip(ri, ci) if iou[i, j] >= 0.5))
            else:
                m = 0
            tp += m; fp += len(t) - m; fn += len(g) - m
        te = reg_get(reg, run, ckpt)
        out.append({"run": run, "checkpoint": ckpt, "TP_own": tp, "FP_own": fp, "FN_own": fn,
                    "recall_own": round(100 * tp / (tp + fn), 3), "precision_own": round(100 * tp / (tp + fp), 3),
                    "TP_trackeval": te["TP"], "FP_trackeval": te["FP"], "FN_trackeval": te["FN"],
                    "delta_TP": tp - te["TP"], "delta_FP": fp - te["FP"], "delta_FN": fn - te["FN"],
                    "tracker_file": SRC[key], "tracker_sha256": sha(p(SRC[key])),
                    "note": "Per-frame Hungarian matching, IoU>=0.5, GT rows with col7==0 dropped; "
                            "TrackEval CLEAR also prefers continuing IDs across frames"})
    return out


# ── 3. Figures ─────────────────────────────────────────────────────────────────
def footer(fig, y=0.005):
    fig.text(0.01, y, FOOTER, fontsize=6, ha="left", va="bottom", color="#444444", wrap=True)

def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=300)
    plt.close(fig)

def series(reg, run, metric):
    return [reg_get(reg, run, f"ckpt_{i}")[metric] for i in range(5)]

def steps_of(run):
    return [s * (4 if run == "Arm2" else 1) for s in STEPS]

def fig01(reg, out):
    fig, ax = plt.subplots(figsize=(W2, 3.3))
    p5 = reg_get(reg, "Phase5", "ckpt_4")["HOTA"]
    ax.axhline(p5, color=C_P5, ls="--", lw=1.2, label=f"Phase 5 (HOTA {p5:.2f})")
    for run, c, best, drop in (("Arm1", C_A1, 3, 10112), ("Arm2", C_A2, 1, 40448)):
        x, y = steps_of(run), series(reg, run, "HOTA")
        ax.plot(x, y, "-o", color=c, ms=4, lw=1.4, label=run.replace("Arm", "Arm "))
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=8, color=c)
        ax.plot(x[best], y[best], marker="*", ms=14, color=c, mec="black", mew=0.6, zorder=5)
        ax.axvline(drop, color=c, ls=":", lw=1)
        if run == "Arm1":
            ax.text(drop, 38.4, f"LR drop (Arm 1) \nafter {drop:,} steps ", color=c, fontsize=8,
                    va="bottom", ha="right")
        else:
            ax.text(drop, 53.6, f"LR drop (Arm 2) \nafter {drop:,} steps ", color=c, fontsize=8,
                    va="top", ha="right")
    a1, a2 = reg_get(reg, "Arm1", "ckpt_3")["HOTA"], reg_get(reg, "Arm2", "ckpt_0")["HOTA"]
    ax.plot([10112, 10112], [a2, a1], color="black", ls="--", lw=0.7)
    ax.text(9700, 42.6, f"equal steps (10,112):\nArm 1 ckpt_3 − Arm 2 ckpt_0\n= {a1 - a2:+.2f} HOTA",
            fontsize=8, va="center", ha="right")
    ax.set_xlabel("Optimizer steps"); ax.set_ylabel("HOTA")
    ax.set_ylim(38, 54)
    ax.plot([], [], "*", color="grey", mec="black", ms=10, ls="none", label="best checkpoint (highest HOTA)")
    ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(0.97, 0.0))
    ax.set_xlim(-6000, 53000)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.2)
    footer(fig); save(fig, out, "fig01_hota_vs_steps")

def fig_two(reg, out, metrics, ylabels, name):
    fig, axes = plt.subplots(2, 1, figsize=(W1, 4.4), sharex=True)
    for ax, m, yl in zip(axes, metrics, ylabels):
        p5 = reg_get(reg, "Phase5", "ckpt_4")[m]
        ax.axhline(p5, color=C_P5, ls="--", lw=1.1, label="Phase 5")
        for run, c in (("Arm1", C_A1), ("Arm2", C_A2)):
            ax.plot(steps_of(run), series(reg, run, m), "-o", color=c, ms=3, lw=1.2,
                    label=run.replace("Arm", "Arm "))
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False, loc="best")
    axes[1].set_xlabel("Optimizer steps")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.98, bottom=0.17, hspace=0.12)
    footer(fig); save(fig, out, name)

def fig04(det, out):
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.9))
    for ax, d, c, name in zip(axes, det, (C_P5, C_A1, C_A2),
                              ("Phase 5 ckpt_4", "Arm 1 ckpt_3", "Arm 2 ckpt_1")):
        cells = [((0, 1), "TP", d["TP_own"]), ((1, 1), "FN", d["FN_own"]), ((0, 0), "FP", d["FP_own"])]
        for (cx, cy), lab, v in cells:
            ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=c, alpha=0.25 if lab == "TP" else 0.12,
                                   edgecolor="black", lw=0.8))
            ax.text(cx + 0.5, cy + 0.5, f"{lab}\n{v:,}", ha="center", va="center", fontsize=9)
        ax.add_patch(Rectangle((1, 0), 1, 1, facecolor="#dddddd", edgecolor="black", lw=0.8, hatch="//"))
        ax.text(1.5, 0.5, "TN: not defined\n(single-class\ndetection)", ha="center", va="center", fontsize=8)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.axis("off")
        ax.text(1, 2.08, name, ha="center", va="bottom", fontsize=8, color=c, fontweight="bold")
        ax.text(1, -0.08, f"Precision {d['precision_own']:.2f}%   Recall {d['recall_own']:.2f}%",
                ha="center", va="top", fontsize=8)
    fig.text(0.5, 0.09, "Top row: ground-truth boxes, matched (TP) or missed (FN). Bottom left: predictions with no "
             "matching ground truth (FP).\nPer-frame Hungarian matching, IoU ≥ 0.5.", ha="center", va="center",
             fontsize=7)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.25, wspace=0.15)
    footer(fig); save(fig, out, "fig04_detection_outcomes")

CANDS = (("Phase5", "ckpt_4", C_P5, "Phase 5 ckpt_4"), ("Arm1", "ckpt_3", C_A1, "Arm 1 ckpt_3"),
         ("Arm2", "ckpt_1", C_A2, "Arm 2 ckpt_1"))

def fig05(reg, out):
    ms = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
    neg = any(reg_get(reg, r, c)["MOTA"] < 0 for r, c, _, _ in CANDS)
    panels = [["HOTA", "DetA", "AssA", "IDF1"], ["MOTA"]] if neg else [ms]
    fig, axes = plt.subplots(1, len(panels), figsize=(W2, 2.9), squeeze=False,
                             gridspec_kw={"width_ratios": [len(x) for x in panels]})
    for ax, pm in zip(axes[0], panels):
        x = np.arange(len(pm)); w = 0.26
        for k, (run, ck, c, lab) in enumerate(CANDS):
            vals = [reg_get(reg, run, ck)[m] for m in pm]
            b = ax.bar(x + (k - 1) * w, vals, w, color=c, label=lab)
            for bi, v in zip(b, vals):
                ax.text(bi.get_x() + bi.get_width() / 2, v + (0.8 if v >= 0 else -0.8), f"{v:.2f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(pm); ax.set_ylabel("Score (%)")
        ax.set_ylim(min(0, min(reg_get(reg, r, c)[m] for r, c, _, _ in CANDS for m in pm) - 5), 85)
        ax.axhline(0, color="black", lw=0.5)
    axes[0][0].legend(frameon=False, ncol=3, loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.18)
    footer(fig); save(fig, out, "fig05_candidate_comparison")

def fig06(reg, out):
    fig, axes = plt.subplots(3, 1, figsize=(W1, 4.8), sharex=True)
    labs = [c[3] for c in CANDS]
    for ax, m in zip(axes, ("IDSW", "AssA", "IDF1")):
        vals = [reg_get(reg, r, c)[m] for r, c, _, _ in CANDS]
        best = int(np.argmin(vals)) if m == "IDSW" else int(np.argmax(vals))
        bars = ax.bar(labs, vals, color=[c[2] for c in CANDS], edgecolor=["black" if i == best else "none"
                                                                          for i in range(3)], linewidth=1.5)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:g}" if m == "IDSW" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(m + (" (lowest outlined)" if m == "IDSW" else " (highest outlined)"), fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.98, bottom=0.14, hspace=0.15)
    footer(fig); save(fig, out, "fig06_identity_metrics")

def seed_table(reg):
    runs = (("42", "Arm1"), ("1234", "seed1"), ("2024", "seed2"))
    t = {}
    for i, s in enumerate(STEPS):
        t[s] = {}
        for m in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"):
            v = [reg_get(reg, run, f"ckpt_{i}")[m] for _, run in runs]
            t[s][m] = {"values": dict(zip([x for x, _ in runs], v)), "mean": st.mean(v), "sd": st.stdev(v),
                       "range": max(v) - min(v)}
    return t

def fig07(reg, seeds, out):
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    mean = [seeds[s]["HOTA"]["mean"] for s in STEPS]; sd = [seeds[s]["HOTA"]["sd"] for s in STEPS]
    ax.fill_between(STEPS, np.array(mean) - sd, np.array(mean) + sd, color=OI["blue"], alpha=0.12,
                    label="mean ± 1 SD (n = 3)")
    for seed in ("42", "1234", "2024"):
        ax.plot(STEPS, [seeds[s]["HOTA"]["values"][seed] for s in STEPS], "-o", lw=0.9, ms=3,
                color=C_SEED[seed], label=f"seed {seed}" + (" (Arm 1)" if seed == "42" else ""))
    ax.plot(STEPS, mean, "-", lw=2.6, color="black", label="seed mean")
    p5 = reg_get(reg, "Phase5", "ckpt_4")["HOTA"]
    ax.axhline(p5, color=C_P5, ls="--", lw=1.2, label=f"Phase 5 ({p5:.2f})")
    lo = min(min(seeds[s]["HOTA"]["values"].values()) for s in STEPS) - 1.2
    for s, m_, d_ in zip(STEPS, mean, sd):
        ax.text(s, lo + 0.2, f"{m_:.2f} ± {d_:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(lo, max(p5, max(np.array(mean) + sd)) + 0.6)
    ax.set_yticks(np.arange(np.ceil(lo + 1.0), np.floor(p5) + 1))
    ax.set_xlim(STEPS[0] - 1200, STEPS[-1] + 1200)
    ax.set_xticks(STEPS); ax.set_xticklabels([f"{s:,}" for s in STEPS])
    ax.set_xlabel("Optimizer steps (Arm 1 configuration)"); ax.set_ylabel("HOTA")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.36)
    fig.text(0.01, 0.005, FOOTER.replace("Single run, seed 42", "Seeds 42, 1234, 2024"), fontsize=6, ha="left",
             va="bottom", color="#444444")
    save(fig, out, "fig07_seed_spread")

def sweep_rows(reg):
    rs = [r for r in reg if r["registry_source"] == SRC["sweep"]]
    return sorted(rs, key=lambda r: r["run"])

def sweep_decision(reg, seeds):
    bar = seeds[10112]["HOTA"]["mean"] + 2 * seeds[10112]["HOTA"]["sd"]
    rs = sweep_rows(reg); sa = [r for r in rs if r["run"].endswith("S_A")][0]
    res = {}
    for r in rs:
        rid = r["run"][-3:]
        ok = r["HOTA"] > bar and r["IDSW"] <= sa["IDSW"] and r["HOTA"] - sa["HOTA"] >= 1.0
        res[rid] = "reference" if rid == "S_A" else ("PASS" if ok else "FAIL")
    return bar, res

def fig08(reg, seeds, out):
    rs = sweep_rows(reg); bar, dec = sweep_decision(reg, seeds)
    ids = [r["run"][-3:] for r in rs]
    p5 = reg_get(reg, "Phase5", "ckpt_4"); sa = [r for r in rs if r["run"].endswith("S_A")][0]
    passing = [i for i in ids if dec[i] == "PASS"]
    best = max(passing, key=lambda i: [r for r in rs if r["run"].endswith(i)][0]["HOTA"]) if passing else None
    fig, axes = plt.subplots(3, 1, figsize=(W1, 5.2), sharex=True)
    for ax, m in zip(axes, ("HOTA", "FP", "FN")):
        vals = [r[m] for r in rs]
        cols = [OI["orange"] if i == best else C_A1 for i in ids]
        bars = ax.bar(ids, vals, color=cols)
        ax.axhline(p5[m], color=C_P5, ls="--", lw=1, label="Phase 5")
        ax.axhline(sa[m], color=C_A1, ls=":", lw=1, label="S_A (0.5/0.5/30)")
        if m == "HOTA":
            ax.axhline(bar, color="black", ls="-.", lw=0.8, label="seed mean + 2 SD")
            ax.set_ylim(min(vals) - 4, max(max(vals), bar, p5[m]) + 3)
        for b_, v in zip(bars, vals):
            if m == "HOTA":
                ax.text(b_.get_x() + b_.get_width() / 2, v - 0.4, f"{v:.2f}", ha="center", va="top",
                        fontsize=6.5, color="white", fontweight="bold")
            else:
                ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:g}", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylabel(m)
    h, l = axes[0].get_legend_handles_labels()
    h.append(Rectangle((0, 0), 1, 1, color=OI["orange"])); l.append("passes decision rule")
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.55, 1.0))
    axes[2].set_xticks(range(len(ids)))
    axes[2].set_xticklabels([f"{i}\n{r['det_thresh']}/{r['miss_tol']}" for i, r in zip(ids, rs)], fontsize=7)
    axes[2].set_xlabel("Setting (det=track threshold / miss tolerance)")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.16, hspace=0.12)
    fig.text(0.01, 0.005, FOOTER.replace("threshold 0.5 unless stated", "Arm 1 ckpt_3, thresholds as labelled"),
             fontsize=6, ha="left", va="bottom", color="#444444", wrap=True)
    save(fig, out, "fig08_sweep_results")

C2SEQ = (("seq_01_1min", "night"), ("seq_02_30sec_a", "night"), ("seq_03_30sec_b", "dusk"))

def c2_rows():
    return read_csv(SRC["c2"])

def fig09(reg, out):
    c2 = c2_rows()
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.0))
    x = np.arange(3); w = 0.36
    for ax, m, lab in zip(axes, ("HOTA", "IDSW", "recall"), ("HOTA", "IDSW", "Recall (%)")):
        for k, onoff in enumerate(("off", "on")):
            vals = [reg_get(reg, f"Phase5_C2{onoff}", "ckpt_4", seq=s, c2=onoff)[m] for s, _ in C2SEQ]
            b = ax.bar(x + (k - 0.5) * w, vals, w, color=C_C2[onoff], label=f"C2 {onoff}")
            for bi, v in zip(b, vals):
                ax.text(bi.get_x() + bi.get_width() / 2, v, f"{v:g}" if m == "IDSW" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{s.split('_')[1]}\n({l})" for s, l in C2SEQ])
        ax.set_ylabel(lab); ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    taus = []
    for s, l in C2SEQ:
        r = [c for c in c2 if c["sequence"] == s and c["c2"] == "on"][0]
        taus.append(f"{s.split('_')[1]}: τ {float(r['tau_min']):.3f}–{float(r['tau_max']):.3f} "
                    f"(mean {float(r['tau_mean']):.3f})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.115, "C2 on (adapter epoch 4), τ per frame — " + ";  ".join(taus) +
             ".\nC2 off: τ = 0.5 fixed.", ha="center", va="center", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.31, wspace=0.35)
    fig.text(0.01, 0.005, "Phase 5 ckpt_4 (never trained on IOTD) on all three IOTD sequences; seq_01 and seq_03 are "
             "IOTD training sequences.\nSingle run, det/track 0.5, miss tolerance 30. Not statistically established.",
             fontsize=6, ha="left", va="bottom", color="#444444")
    save(fig, out, "fig09_c2_comparison")


# ── 4. Tables ──────────────────────────────────────────────────────────────────
def fmt(v, m=None):
    if v is None or v == "":
        return "–"
    if isinstance(v, str):
        return v
    if m in ("IDSW", "FP", "FN", "TP", "frames", "GT rows", "tracks") or isinstance(v, int):
        return f"{int(v):,}" if abs(v) >= 1000 else f"{int(v)}"
    return f"{v:.2f}"

def write_table(out, name, header, rows, bold_cols=None, note="", bold_rows=None):
    """rows: list of lists (raw values). bold_cols: {col_index: 'max'|'min'} -> best value bold in tex/md."""
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    base = os.path.join(out, "tables", name)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (round(v, 3) if isinstance(v, float) else v) for v in r])
    best = {}
    for ci, how in (bold_cols or {}).items():
        cand = rows if bold_rows is None else [rows[i] for i in bold_rows]
        vals = [r[ci] for r in cand if isinstance(r[ci], (int, float))]
        if vals:
            best[ci] = max(vals) if how == "max" else min(vals)
    def cell(ri, ci, tex):
        v = rows[ri][ci]; s = fmt(v, header[ci])
        if tex:
            s = s.replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$").replace("–", "--")
        if ci in best and isinstance(v, (int, float)) and abs(v - best[ci]) < 1e-9 and                 (bold_rows is None or ri in bold_rows):
            return rf"\textbf{{{s}}}" if tex else f"**{s}**"
        return s
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for ri in range(len(rows)):
            f.write("| " + " | ".join(cell(ri, ci, False) for ci in range(len(header))) + " |\n")
        if note:
            f.write(f"\n{note}\n")
    with open(base + ".tex", "w", encoding="utf-8") as f:
        f.write("% requires \\usepackage{booktabs}\n\\begin{tabular}{" + "l" * 1 + "r" * (len(header) - 1) + "}\n")
        f.write("\\toprule\n" + " & ".join(h.replace("_", r"\_").replace("%", r"\%") for h in header)
                + " \\\\\n\\midrule\n")
        for ri in range(len(rows)):
            f.write(" & ".join(cell(ri, ci, True) for ci in range(len(header))) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        if note:
            f.write(f"% {note}\n")
    # PNG image of the table
    ncol = len(header); nrow = len(rows)
    width = min(max(3.4, 0.62 * ncol + 0.6), 11.0)
    fig, ax = plt.subplots(figsize=(width, 0.24 * (nrow + 1) + (0.35 if note else 0.15)))
    ax.axis("off")
    txt = [[fmt(rows[ri][ci], header[ci]) for ci in range(ncol)] for ri in range(nrow)]
    tb = ax.table(cellText=txt, colLabels=header, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
    for (r, c), cl in tb.get_celld().items():
        cl.set_linewidth(0.3)
        if r == 0:
            cl.set_text_props(fontweight="bold"); cl.set_facecolor("#eeeeee")
        elif c in best and isinstance(rows[r - 1][c], (int, float)) and abs(rows[r - 1][c] - best[c]) < 1e-9                 and (bold_rows is None or (r - 1) in bold_rows):
            cl.set_text_props(fontweight="bold")
    tb.auto_set_column_width(list(range(ncol)))
    if note:
        fig.text(0.01, 0.01, note, fontsize=6, ha="left", va="bottom", wrap=True)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def tables(reg, seeds, det, out):
    M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "recall", "precision"]
    H = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "Recall %", "Prec %"]
    best_m = {i: ("min" if m in LOWER_BETTER else "max") for i, m in enumerate(M)}
    # table01
    rows = []
    for run, ck in [("Phase5", "ckpt_4")] + [("Arm1", f"ckpt_{i}") for i in range(5)] + [("Arm2", f"ckpt_{i}")
                                                                                         for i in range(5)]:
        r = reg_get(reg, run, ck)
        steps = int(r["steps"])
        sm = seeds.get(steps, {}).get("HOTA") if run == "Arm1" else None
        rows.append([run, ck, steps] + [r[m] for m in M] +
                    [sm["mean"] if sm else None, sm["sd"] if sm else None])
    write_table(out, "table01_all_checkpoints", ["Run", "Ckpt", "Steps"] + H + ["HOTA seed mean", "HOTA seed SD"],
                rows, {i + 3: v for i, v in best_m.items()},
                "Seed mean/SD: Arm 1 configuration, seeds 42/1234/2024 at the same step count (n = 3). "
                "Phase 5 has 0 IOTD steps.")
    # table02
    p5 = reg_get(reg, "Phase5", "ckpt_4"); rows = []
    for run, ck, _, lab in CANDS:
        r = reg_get(reg, run, ck)
        rows.append([lab] + [r[m] for m in M])
        if run != "Phase5":
            rows.append([f"  Δ vs Phase 5"] + [r[m] - p5[m] for m in M])
    write_table(out, "table02_selected_candidates", ["Model"] + H, rows, {i + 1: v for i, v in best_m.items()},
                "Δ rows: candidate minus Phase 5 (positive FP/FN/IDSW = more errors). Bold: best of the three models.",
                bold_rows=[0, 1, 3])
    # bold best among the 3 model rows only -> rewrite with explicit bold columns on model rows
    # table03
    rows = []
    for s in STEPS:
        for seed in ("42", "1234", "2024"):
            rows.append([f"{s:,}", f"seed {seed}"] + [seeds[s][m]["values"][seed] for m in M[:8]])
        rows.append([f"{s:,}", "mean"] + [seeds[s][m]["mean"] for m in M[:8]])
        rows.append([f"{s:,}", "SD"] + [seeds[s][m]["sd"] for m in M[:8]])
        rows.append([f"{s:,}", "range"] + [seeds[s][m]["range"] for m in M[:8]])
    write_table(out, "table03_seed_spread", ["Steps", "Row"] + H[:8], rows,
                {i + 2: v for i, v in best_m.items() if i < 8},
                "Arm 1 configuration; seed 42 = Arm 1. SD = sample SD over three runs (rough estimate only). "
                "Bold: best single run.", bold_rows=[k for k in range(len(rows)) if rows[k][1].startswith("seed")])
    # table04
    bar, dec = sweep_decision(reg, seeds); rows = []
    rows.append(["Phase 5 ckpt_4", 0.5, 0.5, 30, p5["HOTA"], p5["FP"], p5["FN"], p5["IDSW"], p5["recall"],
                 p5["precision"], "reference"])
    for r in sweep_rows(reg):
        rid = r["run"][-3:]
        rows.append([f"Arm 1 ckpt_3 {rid}", r["det_thresh"], r["track_thresh"], r["miss_tol"], r["HOTA"], r["FP"],
                     r["FN"], r["IDSW"], r["recall"], r["precision"], dec[rid]])
    write_table(out, "table04_sweep_results", ["Setting", "det", "track", "miss", "HOTA", "FP", "FN", "IDSW",
                                               "Recall %", "Prec %", "Rule"], rows,
                {4: "max", 5: "min", 6: "min", 7: "min", 8: "max", 9: "max"},
                f"Rule: HOTA > Arm 1 seed mean + 2 SD at ckpt_3 ({bar:.2f}), IDSW ≤ S_A, and ≥ 1 HOTA over S_A.")
    # table05
    c2 = c2_rows(); rows = []
    for s, l in C2SEQ:
        for onoff in ("off", "on"):
            r = reg_get(reg, f"Phase5_C2{onoff}", "ckpt_4", seq=s, c2=onoff)
            src = [c for c in c2 if c["sequence"] == s and c["c2"] == onoff][0]
            rows.append([s, l, onoff] + [r[m] for m in M] + [float(src["tau_mean"])])
    write_table(out, "table05_c2_results", ["Sequence", "Light", "C2"] + H + ["mean τ"], rows,
                {i + 3: v for i, v in best_m.items()},
                "Phase 5 ckpt_4, det/track 0.5, miss 30. C2 off: τ fixed 0.5. seq_01 and seq_03 are IOTD training "
                "sequences (Phase 5 never trained on them). Bold: best across all six rows.")
    # table06
    prof = {r["sequence"]: r for r in read_csv(SRC["seq_profile"])}
    role = {"seq_01_1min": "train", "seq_02_30sec_a": "val", "seq_03_30sec_b": "train"}
    rows = []; tot = defaultdict(int)
    for s in ("seq_01_1min", "seq_02_30sec_a", "seq_03_30sec_b"):
        g = [l.strip().split(",") for l in open(p(GT_FILES[s])) if l.strip()]
        frames = len({int(x[0]) for x in g}); tracks = len({int(x[1]) for x in g})
        cls = defaultdict(int)
        for x in g:
            cls[x[7]] += 1
        import configparser
        cp = configparser.ConfigParser(); cp.read(p(SEQINFO[s]))
        sq = cp["Sequence"]
        assert int(prof[s]["gt_rows"]) == len(g) and int(prof[s]["n_tracks"]) == tracks
        row = [s, role[s], int(sq["seqLength"]), len(g), tracks, cls["1"], cls["2"], cls["3"], cls["4"], cls["5"],
               float(prof[s]["mean_brightness"]), prof[s]["lighting"], f"{sq['imWidth']}×{sq['imHeight']}",
               int(sq["frameRate"])]
        rows.append(row)
        for i, k in ((2, "frames"), (3, "gt"), (5, "c1"), (6, "c2"), (7, "c3"), (8, "c4"), (9, "c5")):
            tot[k] += row[i]
    rows.append(["Total (real, annotated)", "", tot["frames"], tot["gt"], None, tot["c1"], tot["c2"], tot["c3"],
                 tot["c4"], tot["c5"], None, "", "", None])
    write_table(out, "table06_dataset_summary", ["Sequence", "Role", "Frames", "GT rows", "Tracks", "car",
                                                 "two-wheeler", "auto", "bus", "truck", "Mean luma", "Lighting",
                                                 "Resolution", "fps"], rows, None,
                f"Real annotated frames: {tot['frames']:,}; real training frames: "
                f"{rows[0][2] + rows[2][2]:,} (seq_01 + seq_03). Augmented frames are training samples, not "
                "additional annotated data. Mean luma from data/IOTD/seq_profile.csv (0-255).")
    # table07
    best_mean = max(STEPS, key=lambda s: seeds[s]["HOTA"]["mean"])
    rows = [["Phase 5", "ckpt_4", p5["HOTA"], p5["DetA"], p5["AssA"], p5["MOTA"], p5["IDF1"], "reference"],
            [f"Arm 1 (seed mean, n=3)", f"{best_mean:,} steps"] +
            [seeds[best_mean][m]["mean"] for m in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")] + ["reference"]]
    status = json.load(open(p(SRC["stage1_status"])))
    for v in ("assoc_lr0", "lr_half"):
        b = max((reg_get(reg, v, f"ckpt_{i}") for i in range(5)), key=lambda r: r["HOTA"])
        rows.append([v, b["checkpoint"], b["HOTA"], b["DetA"], b["AssA"], b["MOTA"], b["IDF1"],
                     "No (above bar at 1 of 5 checkpoints)"])
    write_table(out, "table07_tuning_variants", ["Run", "Best ckpt", "HOTA", "DetA", "AssA", "MOTA", "IDF1",
                                                 "Rule passed"], rows,
                {2: "max", 3: "max", 4: "max", 5: "max", 6: "max"},
                "Rule (Stage 1): HOTA > Arm 1 seed mean + 2 SD at > 1 checkpoint without worse AssA or IDSW. "
                "Arm 1 row: checkpoint with the highest seed-mean HOTA.")
    return bar, dec


# ── 5. Captions and dashboard ──────────────────────────────────────────────────
def captions(reg, seeds, det, out):
    p5 = reg_get(reg, "Phase5", "ckpt_4"); a1 = reg_get(reg, "Arm1", "ckpt_3"); a2 = reg_get(reg, "Arm2", "ckpt_1")
    a20 = reg_get(reg, "Arm2", "ckpt_0")
    bar, dec = sweep_decision(reg, seeds)
    c2 = {(c["sequence"], c["c2"]): c for c in c2_rows()}
    sw = {r["run"][-3:]: r for r in sweep_rows(reg)}
    best_sw = max(sw.values(), key=lambda r: r["HOTA"])
    L = ["# Figure captions (measured facts only)", ""]
    L.append(f"**Figure 1.** HOTA against optimizer steps for Arm 1 (real IOTD frames, 2,528 steps/epoch) and Arm 2 "
             f"(augmented, 10,112 steps/epoch). Dashed line: Phase 5 (HOTA {p5['HOTA']:.2f}). Stars: highest-HOTA "
             f"checkpoint per arm (Arm 1 ckpt_3, {a1['HOTA']:.2f}; Arm 2 ckpt_1, {a2['HOTA']:.2f}). Dotted lines: "
             f"learning-rate drop. At 10,112 steps Arm 1 ckpt_3 exceeds Arm 2 ckpt_0 ({a20['HOTA']:.2f}) by "
             f"{a1['HOTA'] - a20['HOTA']:.2f} HOTA.")
    L.append("")
    L.append(f"**Figure 2.** DetA (top) and AssA (bottom) against optimizer steps. Phase 5: DetA {p5['DetA']:.2f}, "
             f"AssA {p5['AssA']:.2f}. No Arm 1 or Arm 2 checkpoint reaches the Phase 5 AssA; the highest is "
             f"{max(series(reg, 'Arm1', 'AssA') + series(reg, 'Arm2', 'AssA')):.2f}.")
    L.append("")
    L.append(f"**Figure 3.** False positives (top) and false negatives (bottom) against optimizer steps. Phase 5: "
             f"FP {p5['FP']:,}, FN {p5['FN']:,}. Arm 1 FN falls from {series(reg, 'Arm1', 'FN')[0]} to "
             f"{series(reg, 'Arm1', 'FN')[-1]}; Arm 2 FP rises from {series(reg, 'Arm2', 'FP')[0]:,} to "
             f"{series(reg, 'Arm2', 'FP')[-1]:,}.")
    L.append("")
    L.append("**Figure 4.** Detection outcomes from per-frame Hungarian matching (IoU ≥ 0.5) for " + "; ".join(
        f"{d['run']} {d['checkpoint']}: TP {d['TP_own']:,}, FP {d['FP_own']:,}, FN {d['FN_own']:,}, precision "
        f"{d['precision_own']:.2f}%, recall {d['recall_own']:.2f}%" for d in det) +
        ". True negatives are not defined for single-class detection.")
    L.append("")
    L.append(f"**Figure 5.** HOTA, DetA, AssA, MOTA and IDF1 of the three candidates. HOTA: Phase 5 {p5['HOTA']:.2f}, "
             f"Arm 1 ckpt_3 {a1['HOTA']:.2f}, Arm 2 ckpt_1 {a2['HOTA']:.2f}.")
    L.append("")
    L.append(f"**Figure 6.** Identity metrics of the three candidates. IDSW: {p5['IDSW']}, {a1['IDSW']}, "
             f"{a2['IDSW']}; AssA: {p5['AssA']:.2f}, {a1['AssA']:.2f}, {a2['AssA']:.2f}; IDF1: {p5['IDF1']:.2f}, "
             f"{a1['IDF1']:.2f}, {a2['IDF1']:.2f} (Phase 5, Arm 1 ckpt_3, Arm 2 ckpt_1).")
    L.append("")
    sm = seeds
    L.append("**Figure 7.** HOTA of three training runs of the Arm 1 configuration (seeds 42, 1234, 2024) against "
             "optimizer steps; thick line: mean; band: ±1 sample SD (n = 3). Mean ± SD per checkpoint: " + ", ".join(
                 f"{sm[s]['HOTA']['mean']:.2f} ± {sm[s]['HOTA']['sd']:.2f}" for s in STEPS) +
             f". Range at the same step count up to {max(sm[s]['HOTA']['range'] for s in STEPS):.2f} HOTA.")
    L.append("")
    L.append(f"**Figure 8.** Inference-time settings for Arm 1 ckpt_3 (det/track threshold and miss tolerance). "
             f"HOTA ranges from {min(r['HOTA'] for r in sw.values()):.2f} to {best_sw['HOTA']:.2f} "
             f"({best_sw['run'][-3:]}); S_A (0.5/0.5/30) {sw['S_A']['HOTA']:.2f}. Dash-dot line: seed mean + 2 SD "
             f"at ckpt_3 ({bar:.2f}). Settings passing the decision rule: "
             f"{', '.join(k for k, v in dec.items() if v == 'PASS') or 'none'}.")
    L.append("")
    L.append("**Figure 9.** Phase 5 ckpt_4 with C2 off (fixed τ = 0.5) and on (adapter epoch 4) on the three IOTD "
             "sequences. HOTA off → on: " + "; ".join(
                 f"{s} {float(c2[(s, 'off')]['HOTA']):.2f} → {float(c2[(s, 'on')]['HOTA']):.2f}" for s, _ in C2SEQ) +
             ". Mean τ with C2 on: " + "; ".join(f"{s} {float(c2[(s, 'on')]['tau_mean']):.3f}" for s, _ in C2SEQ)
             + ".")
    L.append("")
    with open(os.path.join(out, "figure_captions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

def dashboard(reg, seeds, det, out):
    import matplotlib.image as mpimg
    names = ["fig01_hota_vs_steps", "fig02_deta_assa_vs_steps", "fig03_fp_fn_vs_steps", "fig04_detection_outcomes",
             "fig05_candidate_comparison", "fig06_identity_metrics", "fig07_seed_spread", "fig08_sweep_results",
             "fig09_c2_comparison"]
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.42], hspace=0.08, wspace=0.04,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)
    for k, n in enumerate(names):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.imshow(mpimg.imread(os.path.join(out, "figures", n + ".png"))); ax.axis("off")
        ax.text(0.0, 1.0, "ABCDEFGHI"[k], transform=ax.transAxes, fontsize=26, fontweight="bold", va="top")
    a1 = reg_get(reg, "Arm1", "ckpt_3"); a2 = reg_get(reg, "Arm2", "ckpt_1"); p5 = reg_get(reg, "Phase5", "ckpt_4")
    bar, dec = sweep_decision(reg, seeds)
    passing = [k for k, v in dec.items() if v == "PASS"]
    sw = {r["run"][-3:]: r for r in sweep_rows(reg)}
    rec = max(passing, key=lambda k: sw[k]["HOTA"]) if passing else "S_A"
    c2 = {(c["sequence"], c["c2"]): c for c in c2_rows()}
    c2txt = ", ".join(f"{s.split('_')[1]} {float(c2[(s, 'on')]['HOTA']) - float(c2[(s, 'off')]['HOTA']):+.2f}"
                      for s, _ in C2SEQ)
    txt = (f"STATUS (pre-tuning baseline v1, generated from saved evaluation files)\n"
           f"Best checkpoint per arm (highest HOTA at 0.5): Phase 5 ckpt_4 {p5['HOTA']:.2f} | Arm 1 ckpt_3 "
           f"{a1['HOTA']:.2f} | Arm 2 ckpt_1 {a2['HOTA']:.2f}\n"
           f"Seed spread (Arm 1 config, n=3): HOTA SD {min(seeds[s]['HOTA']['sd'] for s in STEPS):.2f}–"
           f"{max(seeds[s]['HOTA']['sd'] for s in STEPS):.2f}; ckpt_3 mean {seeds[10112]['HOTA']['mean']:.2f}, "
           f"bar (mean+2SD) {bar:.2f}\n"
           f"Recommended inference setting (Arm 1): {rec} (det {sw[rec]['det_thresh']}, track "
           f"{sw[rec]['track_thresh']}, miss {sw[rec]['miss_tol']}); settings passing the rule: "
           f"{', '.join(passing) or 'none'}\n"
           f"C2 (Phase 5 base, adapter epoch 4) HOTA change on − off: {c2txt}\n"
           f"Open items: single validation sequence; 4 medium-confidence GT gaps in seq_02 unverified; C1 zone "
           f"boundaries uncalibrated; C2 adapter trained on UA-DETRAC/VisDrone, not IOTD; no S1 (corrected GT) "
           f"scoring;\n            run_inference_mot.py writes tracks inside their miss window and has no result-score "
           f"filter (upstream submit_engine.py has one), which is why miss tolerance moves FP so much.")
    ax = fig.add_subplot(gs[3, :]); ax.axis("off")
    ax.text(0.01, 0.95, txt, va="top", ha="left", fontsize=15, family="DejaVu Sans",
            bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#888888"))
    fig.savefig(os.path.join(out, "figures", "dashboard.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ARCHIVE)
    out = os.path.abspath(ap.parse_args().out)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    reg = build_registry()
    with open(os.path.join(out, "data", "results_registry_v1.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REG_COLS); w.writeheader(); w.writerows(reg)
    print(f"registry: {len(reg)} rows, every metric re-read from its TrackEval summary and matched to its source CSV")
    det = detection_outcomes(reg)
    with open(os.path.join(out, "data", "detection_outcomes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det[0].keys())); w.writeheader(); w.writerows(det)
    for d in det:
        print(f"detection outcomes {d['run']} {d['checkpoint']}: own TP/FP/FN {d['TP_own']}/{d['FP_own']}/"
              f"{d['FN_own']}  TrackEval {d['TP_trackeval']}/{d['FP_trackeval']}/{d['FN_trackeval']}")
    seeds = seed_table(reg)
    fig01(reg, out)
    fig_two(reg, out, ("DetA", "AssA"), ("DetA", "AssA"), "fig02_deta_assa_vs_steps")
    fig_two(reg, out, ("FP", "FN"), ("False positives", "False negatives"), "fig03_fp_fn_vs_steps")
    fig04(det, out); fig05(reg, out); fig06(reg, out); fig07(reg, seeds, out); fig08(reg, seeds, out)
    fig09(reg, out)
    bar, dec = tables(reg, seeds, det, out)
    captions(reg, seeds, det, out)
    dashboard(reg, seeds, det, out)
    print(f"sweep bar {bar:.3f}; decisions {dec}")
    print(f"written to {out}")

if __name__ == "__main__":
    main()
