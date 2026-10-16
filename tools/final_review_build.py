"""
tools/final_review_build.py — build the final-review tables, dashboard and run manifest from saved TrackEval summaries
(read-only on every input; refuses to overwrite). Writes to results_after_tuning/final_review/:
  corrected_model_comparison.csv, historical_vs_corrected.csv, corrected_c2_comparison.csv,
  historical_c2_unfiltered.csv, selected_settings.csv, final_dashboard.png, run_manifest.json
and prints every number used in the Markdown documents.
"""
import os, csv, json, hashlib, datetime, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AT = os.path.join(ROOT, "results_after_tuning"); FR = os.path.join(AT, "final_review")
R = lambda *p: os.path.join(ROOT, *p)
rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/")
REG = R("archive", "pre_tuning_baseline_v1", "data", "results_registry_v1.csv")
STAGE1 = R("results", "tuning_stage1", "stage1_results.csv")
HIST_C2 = R("results", "c2_eval", "c2_results.csv")
GT02 = R("data", "IOTD_val", "seq_02_30sec_a", "gt", "gt.txt")
GT_ORIG = {"seq_01_1min": R("data", "IOTD_train_real", "seq_01_1min", "gt", "gt.txt"),
           "seq_02_30sec_a": GT02, "seq_03_30sec_b": R("data", "IOTD_train_real", "seq_03_30sec_b", "gt", "gt.txt")}
MODELS = {"Phase5": ("results/phase5_interim_baseline", "configs/train_iotd_real.yaml", "Phase 5 baseline (no IOTD)", [4]),
          "Arm1": ("results/train_iotd_real", "configs/train_iotd_real.yaml", "Arm 1: real IOTD frames", range(5)),
          "Arm2": ("results/train_iotd_real_aug", "configs/train_iotd_real_aug.yaml", "Arm 2: real + augmented", range(5)),
          "assoc_lr0": ("results/tuning_stage1/assoc_lr0", "configs/train_iotd_real.yaml",
                        "Stage 1: query-updater LR 0", range(5)),
          "lr_half": ("results/tuning_stage1/lr_half", "configs/train_iotd_real.yaml", "Stage 1: all LRs halved", range(5))}
STEPS = {"Phase5": lambda i: 0, "Arm2": lambda i: 10112 * (i + 1)}
M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "precision", "recall"]
OI = {"grey": "#999999", "blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
      "orange": "#E69F00", "sky": "#56B4E9"}
COL = {"Phase5": OI["grey"], "Arm1": OI["blue"], "Arm2": OI["vermilion"], "assoc_lr0": OI["green"],
       "lr_half": OI["purple"]}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

def summ(p):
    l = [x.split() for x in open(p) if x.strip()]; d = dict(zip(l[0], l[1]))
    return {"HOTA": float(d["HOTA"]), "DetA": float(d["DetA"]), "AssA": float(d["AssA"]), "MOTA": float(d["MOTA"]),
            "IDF1": float(d["IDF1"]), "IDSW": int(float(d["IDSW"])), "TP": int(float(d["CLR_TP"])),
            "FP": int(float(d["CLR_FP"])), "FN": int(float(d["CLR_FN"])), "precision": float(d["CLR_Pr"]),
            "recall": float(d["CLR_Re"])}

def run_summary(jid):
    if jid == "Arm1_ckpt_3":
        return R("results_after_tuning", "sweep", "S_A", "trackeval", "trackeval_output", "AFTER_S_A",
                 "pedestrian_summary.txt"), R("results_after_tuning", "sweep", "S_A"), "reused (corrected sweep S_A)"
    d = os.path.join(FR, "runs", jid)
    return os.path.join(d, "trackeval", "trackeval_output", jid.upper(), "pedestrian_summary.txt"), d, \
        "newly evaluated (final review)"

def write_csv(path, rows, cols):
    if os.path.exists(path):
        print(f"REFUSE to overwrite {path}"); return False
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"written {rel(path)} ({len(rows)} rows)"); return True

def main():
    gt_sha = sha(GT02); manifest_rows = []
    # 1. corrected model comparison
    comp = []; missing = []
    for m, (cdir, cfg, desc, cks) in MODELS.items():
        for i in cks:
            jid = f"{m}_ckpt_{i}"; sp, outdir, how = run_summary(jid)
            base = {"model": m, "description": desc, "checkpoint": f"checkpoint_{i}.pth",
                    "checkpoint_path": f"{cdir}/checkpoint_{i}.pth",
                    "optimizer_steps": STEPS.get(m, lambda i: 2528 * (i + 1))(i), "seed": 42,
                    "training_variant": desc, "inference_runner": "tools/run_inference_mot_filtered.py",
                    "inference_config": cfg, "det_thresh": 0.5, "track_thresh": 0.5, "miss_tol": 30,
                    "score_filter": "RESULT_SCORE_THRESH 0.5 (score > 0.5)", "area_filter": "area > 100 px",
                    "c1": "off", "c2": "off", "sequence": "seq_02_30sec_a", "GT_sha256": gt_sha,
                    "evaluation": "TrackEval HOTA/CLEAR/Identity, DO_PREPROC=False, class-1 eval GT copy",
                    "row_status": how}
            if os.path.exists(sp):
                base.update(summ(sp)); base.update(summary_path=rel(sp), summary_sha256=sha(sp), verified="Verified")
            else:
                base.update(row_status="unavailable (summary missing)", verified="Not available"); missing.append(jid)
            comp.append(base)
            manifest_rows.append({"row": jid, "status": base["row_status"], "summary": base.get("summary_path"),
                                  "summary_sha256": base.get("summary_sha256"), "output_dir": rel(outdir)})
    best = {}
    for m in MODELS:
        rows = [r for r in comp if r["model"] == m and "HOTA" in r]
        if rows:
            b = max(rows, key=lambda r: r["HOTA"]); best[m] = b
    for r in comp:
        r["best_checkpoint_of_model"] = "yes" if best.get(r["model"]) is r else ""
    cols = (["model", "checkpoint", "best_checkpoint_of_model", "optimizer_steps"] + M +
            ["TP", "checkpoint_path", "seed", "training_variant", "inference_runner", "inference_config", "det_thresh",
             "track_thresh", "miss_tol", "score_filter", "area_filter", "c1", "c2", "sequence", "GT_sha256",
             "evaluation", "row_status", "verified", "summary_path", "summary_sha256"])
    write_csv(os.path.join(FR, "corrected_model_comparison.csv"), comp, cols)

    print("\n=== Corrected procedure, every checkpoint (seq_02, det/track 0.5, miss 30, filtered) ===")
    for r in comp:
        if "HOTA" in r:
            print(f"  {r['model']:9s} {r['checkpoint']:16s} " + " ".join(
                f"{k}={r[k]:.2f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in M) +
                  ("  <- best" if r["best_checkpoint_of_model"] else ""))
        else:
            print(f"  {r['model']:9s} {r['checkpoint']:16s} UNAVAILABLE")
    print("\n=== Best checkpoint per model (highest corrected HOTA of its checkpoints) ===")
    for m, b in sorted(best.items(), key=lambda kv: -kv[1]["HOTA"]):
        print(f"  {m:9s} {b['checkpoint']:16s} HOTA {b['HOTA']:.2f} DetA {b['DetA']:.2f} AssA {b['AssA']:.2f} "
              f"MOTA {b['MOTA']:.2f} IDF1 {b['IDF1']:.2f} IDSW {b['IDSW']} FP {b['FP']} FN {b['FN']} "
              f"Prec {b['precision']:.2f} Rec {b['recall']:.2f}")

    # 2. historical vs corrected
    reg = list(csv.DictReader(open(REG))); s1 = list(csv.DictReader(open(STAGE1)))
    def hist(m, i):
        if m in ("Phase5", "Arm1", "Arm2"):
            c = [r for r in reg if r["run"] == m and r["checkpoint"] == f"ckpt_{i}" and float(r["det_thresh"]) == 0.5
                 and r["sequence"] == "seq_02_30sec_a" and r["c2"] == "off" and int(r["miss_tol"]) == 30]
            return (c[0], rel(REG)) if c else (None, None)
        c = [r for r in s1 if r["run"] == m and r["checkpoint"] == f"ckpt_{i}"]
        return (c[0], rel(STAGE1)) if c else (None, None)
    hv = []
    for r in comp:
        i = int(r["checkpoint"].split("_")[1].split(".")[0]); h, src = hist(r["model"], i)
        row = {"model": r["model"], "checkpoint": r["checkpoint"],
               "WARNING": "historical = unfiltered output (old protocol); corrected = filtered output. "
                          "Do not compare across protocols as a model improvement.",
               "historical_source": src, "corrected_source": r.get("summary_path")}
        for k in M:
            row[f"{k}_historical_unfiltered"] = float(h[k]) if h else ""
            row[f"{k}_corrected"] = r.get(k, "")
        hv.append(row)
    write_csv(os.path.join(FR, "historical_vs_corrected.csv"), hv,
              ["model", "checkpoint", "WARNING"] + [f"{k}_{s}" for k in M for s in ("historical_unfiltered",
                                                                                     "corrected")]
              + ["historical_source", "corrected_source"])
    print("\n=== Historical (unfiltered) best vs corrected best ===")
    for m in MODELS:
        hrows = [(i, hist(m, i)[0]) for i in MODELS[m][3]]
        hb = max(hrows, key=lambda t: float(t[1]["HOTA"]))
        print(f"  {m:9s} historical best ckpt_{hb[0]} HOTA {float(hb[1]['HOTA']):.2f} | corrected best "
              f"{best[m]['checkpoint'] if m in best else '-'} HOTA {best[m]['HOTA'] if m in best else float('nan'):.2f}")

    # 3. C2
    c2 = []
    for s, light in (("seq_01_1min", "night"), ("seq_02_30sec_a", "night"), ("seq_03_30sec_b", "dusk")):
        for onoff in ("off", "on"):
            jid = "Phase5_ckpt_4" if (s == "seq_02_30sec_a" and onoff == "off") else f"C2_{s}_{onoff}"
            d = os.path.join(FR, "runs", jid)
            sp = os.path.join(d, "trackeval", "trackeval_output", jid.upper(), "pedestrian_summary.txt")
            row = {"sequence": s, "lighting": light, "c2": onoff, "model": "Phase5 checkpoint_4.pth",
                   "inference_runner": "tools/run_inference_mot_filtered.py" + (" via tools/final_review_c2_tau.py"
                                                                               if onoff == "on" else ""),
                   "adapter": "results/phase6_c2_interim/c2_adapter_epoch_4.pth" if onoff == "on" else "",
                   "det_thresh": "adaptive (0.3-0.7)" if onoff == "on" else 0.5, "track_thresh": 0.5, "miss_tol": 30,
                   "score_filter": "score > 0.5", "area_filter": "area > 100 px", "GT_sha256_original": sha(GT_ORIG[s]),
                   "row_status": "reused (Phase5_ckpt_4 job)" if jid == "Phase5_ckpt_4" else "newly evaluated"}
            if os.path.exists(sp):
                row.update(summ(sp)); row.update(summary_path=rel(sp), verified="Verified")
                tp = os.path.join(d, "tau_per_frame.csv")
                if onoff == "on" and os.path.exists(tp):
                    t = [float(x["tau"]) for x in csv.DictReader(open(tp))]
                    row.update(tau_mean=round(st.mean(t), 4), tau_min=round(min(t), 4), tau_max=round(max(t), 4))
                elif onoff == "off":
                    row.update(tau_mean=0.5, tau_min=0.5, tau_max=0.5)
            else:
                row.update(row_status="pending (summary missing)", verified="Not available")
            c2.append(row)
            manifest_rows.append({"row": f"C2 {s} {onoff}", "status": row["row_status"],
                                  "summary": row.get("summary_path"), "output_dir": rel(d)})
    write_csv(os.path.join(FR, "corrected_c2_comparison.csv"), c2,
              ["sequence", "lighting", "c2"] + M + ["tau_mean", "tau_min", "tau_max", "model", "adapter",
                                                    "inference_runner", "det_thresh", "track_thresh", "miss_tol",
                                                    "score_filter", "area_filter", "GT_sha256_original", "row_status",
                                                    "verified", "summary_path"])
    hc2 = [dict(r, WARNING="HISTORICAL — unfiltered output (old protocol); not comparable with corrected results",
                source=rel(HIST_C2)) for r in csv.DictReader(open(HIST_C2))]
    write_csv(os.path.join(FR, "historical_c2_unfiltered.csv"), hc2,
              ["WARNING", "sequence", "lighting", "c2", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN",
               "precision", "recall", "tau_mean", "tau_min", "tau_max", "source"])
    print("\n=== C2 corrected (Phase 5 ckpt_4, filtered) ===")
    for r in c2:
        if "HOTA" in r:
            print(f"  {r['sequence']:15s} {r['lighting']:5s} C2 {r['c2']:3s} HOTA {r['HOTA']:.2f} DetA {r['DetA']:.2f} "
                  f"AssA {r['AssA']:.2f} MOTA {r['MOTA']:.2f} IDF1 {r['IDF1']:.2f} IDSW {r['IDSW']} FP {r['FP']} "
                  f"FN {r['FN']} Prec {r['precision']:.2f} Rec {r['recall']:.2f} tau {r.get('tau_mean')} "
                  f"[{r.get('tau_min')}, {r.get('tau_max')}]")
        else:
            print(f"  {r['sequence']} C2 {r['c2']}: PENDING")
    for s in ("seq_01_1min", "seq_02_30sec_a", "seq_03_30sec_b"):
        a = [r for r in c2 if r["sequence"] == s and "HOTA" in r]
        if len(a) == 2:
            print(f"  {s}: dHOTA on-off {a[1]['HOTA']-a[0]['HOTA']:+.2f} dIDSW {a[1]['IDSW']-a[0]['IDSW']:+d} "
                  f"dFP {a[1]['FP']-a[0]['FP']:+d} dFN {a[1]['FN']-a[0]['FN']:+d}")

    # 4. selected settings
    sel = [
        {"item": "runner", "value": "tools/run_inference_mot_filtered.py", "type": "implementation (matches upstream)",
         "status": "selected", "evidence": "results_after_tuning/logs/00_equivalence_filters_off.txt; 01_S_A.txt"},
        {"item": "checkpoint (main model)", "value": f"Arm 1 {best['Arm1']['checkpoint']}" if "Arm1" in best else "",
         "type": "model", "status": "see final_review_report.md section 4",
         "evidence": "final_review/corrected_model_comparison.csv"},
        {"item": "detection threshold", "value": 0.5, "type": "tunable", "status": "selected (S_A)",
         "evidence": "results_after_tuning/corrected_sweep_results.csv"},
        {"item": "tracking threshold", "value": 0.5, "type": "tunable", "status": "selected (S_A)",
         "evidence": "results_after_tuning/corrected_sweep_results.csv"},
        {"item": "miss tolerance", "value": 30, "type": "tunable", "status": "selected (S_A); 20 rejected",
         "evidence": "results_after_tuning/seed_check_results.csv"},
        {"item": "output score filter", "value": "RESULT_SCORE_THRESH 0.5 (keep score > 0.5)",
         "type": "fixed by upstream config, not tuned", "status": "applied",
         "evidence": "MeMOTR/submit_engine.py:94,122-125; configs/train_iotd_real.yaml:38"},
        {"item": "box-area filter", "value": "area > 100 px", "type": "fixed by upstream code, not tuned",
         "status": "applied", "evidence": "MeMOTR/submit_engine.py:95,127-131"},
        {"item": "C1 (perspective memory decay)", "value": "disabled", "type": "contribution",
         "status": "implemented; not calibrated, trained or evaluated", "evidence": "modules/position_decay.py:29-31"},
        {"item": "C2 (scene-conditioned threshold)", "value": "disabled", "type": "contribution",
         "status": "pending camera-specific calibration; see corrected_c2_comparison.csv",
         "evidence": "final_review/corrected_c2_comparison.csv"},
    ]
    write_csv(os.path.join(FR, "selected_settings.csv"), sel, ["item", "value", "type", "status", "evidence"])

    # 5. dashboard
    dst = os.path.join(FR, "final_dashboard.png")
    if os.path.exists(dst):
        print(f"REFUSE to overwrite {rel(dst)}")
    else:
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3, left=0.05, right=0.98, top=0.92, bottom=0.04)
        fig.suptitle("MeMOTR on IOTD — final review. CORRECTED procedure (upstream output filter: score > 0.5, area > 100 px)"
                     " unless marked HISTORICAL.\nValidation: seq_02_30sec_a (744 frames, night). Det/track 0.5, miss 30. "
                     "Single sequence; not statistically established.", fontsize=12)
        ax = fig.add_subplot(gs[0, 0:2])
        for m in ("Arm1", "Arm2", "assoc_lr0", "lr_half"):
            rows = [r for r in comp if r["model"] == m and "HOTA" in r]
            ax.plot([int(r["checkpoint"][11]) for r in rows], [r["HOTA"] for r in rows], "-o", color=COL[m],
                    label=MODELS[m][2])
            b = best[m]; ax.plot(int(b["checkpoint"][11]), b["HOTA"], "*", ms=16, color=COL[m], mec="black")
            for r in rows:
                ax.annotate(f"{r['HOTA']:.1f}", (int(r["checkpoint"][11]), r["HOTA"]), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=7, color=COL[m])
        if "Phase5" in best:
            ax.axhline(best["Phase5"]["HOTA"], color=OI["grey"], ls="--", label=f"Phase 5 ckpt_4 ({best['Phase5']['HOTA']:.2f})")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"ckpt_{i}" for i in range(5)]); ax.set_ylabel("HOTA")
        ax.set_title("A  CORRECTED HOTA per checkpoint (star = best of model)", fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
        order = [m for m in MODELS if m in best]; x = np.arange(len(order))
        ax = fig.add_subplot(gs[0, 2])
        for k, (met, c) in enumerate((("HOTA", "#333333"), ("DetA", OI["sky"]), ("AssA", OI["orange"]))):
            v = [best[m][met] for m in order]; b_ = ax.bar(x + (k - 1) * 0.27, v, 0.27, color=c, label=met)
            for bb, vv in zip(b_, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=6.5,
                        rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([f"{m}\n{best[m]['checkpoint'][:12]}" for m in order], fontsize=7)
        ax.set_ylim(0, 85); ax.legend(fontsize=7, frameon=False, ncol=3)
        ax.set_title("B  CORRECTED best checkpoint per model: HOTA / DetA / AssA", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_subplot(gs[1, 0])
        for k, (met, c) in enumerate((("FP", OI["vermilion"]), ("FN", OI["blue"]))):
            v = [best[m][met] for m in order]; b_ = ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=c, label=met)
            for bb, vv in zip(b_, v):
                ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8); ax.legend(fontsize=8, frameon=False)
        ax.set_title("C  CORRECTED best checkpoints: false positives / false negatives", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_subplot(gs[1, 1])
        v = [best[m]["IDSW"] for m in order]; b_ = ax.bar(x, v, color=[COL[m] for m in order])
        for bb, vv in zip(b_, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8); ax.set_title("D  CORRECTED best checkpoints: ID switches",
                                                                               fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_subplot(gs[1, 2])
        hb = []
        for m in order:
            hrows = [hist(m, i)[0] for i in MODELS[m][3]]; hb.append(max(float(r["HOTA"]) for r in hrows))
        cb = [best[m]["HOTA"] for m in order]
        b1 = ax.bar(x - 0.2, hb, 0.4, color="white", edgecolor="black", hatch="//", label="HISTORICAL (unfiltered) best")
        b2 = ax.bar(x + 0.2, cb, 0.4, color=OI["blue"], label="CORRECTED best")
        for bb, vv in list(zip(b1, hb)) + list(zip(b2, cb)):
            ax.text(bb.get_x() + bb.get_width() / 2, vv, f"{vv:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8); ax.set_ylim(0, 80); ax.legend(fontsize=7, frameon=False)
        ax.set_title("E  Protocol change: HISTORICAL vs CORRECTED best HOTA\n(different protocols — not an improvement)",
                     fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_subplot(gs[2, 0])
        seqs = ["seq_01_1min", "seq_02_30sec_a", "seq_03_30sec_b"]; xs = np.arange(3)
        for k, onoff in enumerate(("off", "on")):
            v = [next((r["HOTA"] for r in c2 if r["sequence"] == s and r["c2"] == onoff and "HOTA" in r), np.nan)
                 for s in seqs]
            b_ = ax.bar(xs + (k - 0.5) * 0.38, v, 0.38, color=OI["grey"] if onoff == "off" else OI["orange"],
                        label=f"C2 {onoff}")
            for bb, vv in zip(b_, v):
                ax.text(bb.get_x() + bb.get_width() / 2, 0 if np.isnan(vv) else vv,
                        "pending" if np.isnan(vv) else f"{vv:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(["seq_01\n(night)", "seq_02\n(night)", "seq_03\n(dusk)"])
        ax.set_ylabel("HOTA"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("F  CORRECTED C2 off vs on (Phase 5 ckpt_4)", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_subplot(gs[2, 1:]); ax.axis("off")
        a1 = best.get("Arm1", {}); p5 = best.get("Phase5", {})
        txt = ("SELECTED INFERENCE CONFIGURATION (S_A)\n"
               "  runner tools/run_inference_mot_filtered.py | det 0.5 | track 0.5 | miss tolerance 30\n"
               "  output filter: score > RESULT_SCORE_THRESH 0.5, box area > 100 px (upstream MeMOTR; not a tuned gain)\n"
               "  C1 disabled (implemented, not calibrated/evaluated) | C2 disabled (see panel F)\n\n"
               "WHY S_A: corrected S_A 64.12 vs S_D 63.89 (seed 42); 3 seeds S_A 61.64 ± 2.25 vs S_D 61.57 ± 2.12,\n"
               "  mean S_D − S_A = −0.07. S_F 64.25 is a single run (+0.12, IDSW 6 vs 3) below the bar 66.14.\n\n"
               f"CORRECTED: Arm 1 {a1.get('checkpoint', '?')} HOTA {a1.get('HOTA', float('nan')):.2f} vs Phase 5 "
               f"HOTA {p5.get('HOTA', float('nan')):.2f} (single run each, seed 42).")
        ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=10.5, family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#888888"))
        ax.set_title("G  Selected configuration", fontsize=10, loc="left")
        fig.savefig(dst, dpi=150); print(f"written {rel(dst)}")

    # 6. run manifest
    mp = os.path.join(FR, "run_manifest.json")
    if not os.path.exists(mp):
        man = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "validation_sequence": "seq_02_30sec_a", "GT_original_sha256": {k: sha(v) for k, v in GT_ORIG.items()},
               "GT_eval_copy_seq02": rel(R("results", "eval_iotd_val", "gt_eval", "seq_02_30sec_a", "gt", "gt.txt")),
               "evaluation": "tools/run_trackeval.py (TrackEval HOTA, CLEAR, Identity; DO_PREPROC=False; IoU 0.5)",
               "settings": {"det_thresh": 0.5, "track_thresh": 0.5, "miss_tol": 30, "RESULT_SCORE_THRESH": 0.5,
                            "area_thresh_px": 100, "use_c1": False},
               "scripts": {rel(R("tools", f)): sha(R("tools", f)) for f in
                           ("run_inference_mot.py", "run_inference_mot_filtered.py", "run_trackeval.py",
                            "final_review_run.py", "final_review_batch.py", "final_review_c2_tau.py",
                            "final_review_build.py", "final_review_verify.py", "sweep_hashcheck.py")},
               "models": {m: {"checkpoint_dir": v[0], "inference_config": v[1], "description": v[2]}
                          for m, v in MODELS.items()},
               "c2_adapter": "results/phase6_c2_interim/c2_adapter_epoch_4.pth",
               "rows": manifest_rows,
               "historical_sources": {"registry": rel(REG), "stage1": rel(STAGE1), "c2": rel(HIST_C2)},
               "missing": missing}
        json.dump(man, open(mp, "w", encoding="utf-8"), indent=1); print(f"written {rel(mp)}")

if __name__ == "__main__":
    main()
