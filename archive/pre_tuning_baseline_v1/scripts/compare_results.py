"""
compare_results.py — compare new results with the frozen registry data/results_registry_v1.csv.

    python archive/pre_tuning_baseline_v1/scripts/compare_results.py --new <folder | registry-format CSV>
           [--ref-run Arm1 --ref-ckpt ckpt_3 --ref-thresh 0.5] [--seq seq_02_30sec_a] [--out <csv outside archive>]

--new  either a folder: every pedestrian_summary.txt below it is read (sequence and GT SHA256 are then taken from
       --seq and the archive's GT record, and the user is warned that they cannot be checked from a bare summary),
       or a CSV in registry format (needs run, checkpoint, scoring, threshold, GT_sha256 and metric columns).
Checks per new row: scoring == S0, threshold == ref threshold, GT_sha256 == the archive's GT SHA256 for the sequence,
sequence == ref sequence. Mismatches are printed as WARN lines; the difference table is printed regardless.
--out  optional CSV for the difference table. Writing inside archive/ is refused.
"""
from __future__ import annotations
import os, sys, csv, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.dirname(HERE)
ARCHIVE_ROOT = os.path.dirname(ARCHIVE)            # .../archive
REG = os.path.join(ARCHIVE, "data", "results_registry_v1.csv")
M = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN", "recall", "precision"]

def inside_archive(path):
    a = os.path.normcase(os.path.realpath(ARCHIVE_ROOT))
    t = os.path.normcase(os.path.realpath(path))
    return t == a or t.startswith(a + os.sep)

def read_summary(path):
    lines = [l.split() for l in open(path, encoding="utf-8") if l.strip()]
    d = dict(zip(lines[0], lines[1]))
    return {"HOTA": d["HOTA"], "DetA": d["DetA"], "AssA": d["AssA"], "MOTA": d["MOTA"], "IDF1": d["IDF1"],
            "IDSW": d["IDSW"], "FP": d["CLR_FP"], "FN": d["CLR_FN"], "recall": d["CLR_Re"], "precision": d["CLR_Pr"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref-run", default="Arm1"); ap.add_argument("--ref-ckpt", default="ckpt_3")
    ap.add_argument("--ref-thresh", type=float, default=0.5); ap.add_argument("--seq", default="seq_02_30sec_a")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out and inside_archive(a.out):
        print(f"REFUSED: --out {a.out} is inside {ARCHIVE_ROOT}; the archive is read-only. Nothing written.")
        sys.exit(2)

    reg = list(csv.DictReader(open(REG, encoding="utf-8")))
    ref = [r for r in reg if r["run"] == a.ref_run and r["checkpoint"] == a.ref_ckpt and r["sequence"] == a.seq
           and float(r["det_thresh"]) == a.ref_thresh and r["c2"] == "off" and int(r["miss_tol"]) == 30]
    if len(ref) != 1:
        print(f"reference not found uniquely in registry: {a.ref_run} {a.ref_ckpt} {a.seq} {a.ref_thresh}")
        sys.exit(1)
    ref = ref[0]
    gt_sha = ref["GT_sha256"]
    print(f"reference: {ref['run']} {ref['checkpoint']} seq={ref['sequence']} scoring={ref['scoring']} "
          f"threshold={ref['threshold']} GT={gt_sha[:12]}…  HOTA={float(ref['HOTA']):.2f}")

    new = []
    if os.path.isdir(a.new):
        for dp, _, fs in os.walk(a.new):
            for fn in fs:
                if fn == "pedestrian_summary.txt":
                    r = read_summary(os.path.join(dp, fn))
                    r.update(run=os.path.relpath(dp, a.new), scoring=None, threshold=None, GT_sha256=None,
                             sequence=None)
                    new.append(r)
        print(f"WARN: {len(new)} bare TrackEval summaries: scoring, threshold, GT SHA256 and sequence are not "
              f"recorded in a summary file and cannot be checked; use a registry-format CSV for full checks.")
    else:
        for r in csv.DictReader(open(a.new, encoding="utf-8")):
            r.setdefault("sequence", a.seq)
            new.append(r)

    warns = 0
    print(f"\n{'run':40s} " + " ".join(f"{m:>9s}" for m in M) + "   checks")
    for r in new:
        checks = []
        if r.get("scoring") not in (None, "", "S0"):
            checks.append(f"scoring={r['scoring']}")
        if r.get("threshold") not in (None, "") and abs(float(r["threshold"]) - a.ref_thresh) > 1e-9:
            checks.append(f"threshold={r['threshold']}≠{a.ref_thresh}")
        if r.get("GT_sha256") not in (None, "") and r["GT_sha256"] != gt_sha:
            checks.append("GT_sha256 differs")
        if r.get("sequence") not in (None, "") and r["sequence"] != a.seq:
            checks.append(f"sequence={r['sequence']}")
        warns += bool(checks)
        diffs = []
        for m in M:
            d = float(r[m]) - float(ref[m])
            diffs.append(f"{d:+9.2f}" if m not in ("IDSW", "FP", "FN") else f"{int(round(d)):+9d}")
        print(f"{r['run'][:40]:40s} " + " ".join(diffs) + "   " + ("WARN " + "; ".join(checks) if checks else "ok"))
        r["_diffs"] = diffs; r["_checks"] = "; ".join(checks) or "ok"
    print(f"\n(values are new − reference; {warns} row(s) with protocol warnings)")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["run"] + [f"d_{m}" for m in M] + ["checks"])
            for r in new:
                w.writerow([r["run"]] + [x.strip() for x in r["_diffs"]] + [r["_checks"]])
        print(f"difference table written to {a.out}")

if __name__ == "__main__":
    main()
