"""
make_manifest.py — write results_after_tuning/guide_presentation/MANIFEST.json: SHA256 of every file in the folder
(except MANIFEST.json) and of every source CSV and TrackEval summary used. Refuses to overwrite.
"""
import os, sys, csv, json, hashlib, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(OUT, "..", ".."))
sys.path.insert(0, HERE)
from build_guide import SRC  # noqa: E402

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/")

def main():
    dst = os.path.join(OUT, "MANIFEST.json")
    if os.path.exists(dst):
        print(f"REFUSE: {dst} exists"); sys.exit(1)
    files = {}
    for dp, dn, fs in os.walk(OUT):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for fn in fs:
            p = os.path.join(dp, fn)
            if os.path.abspath(p) != os.path.abspath(dst):
                files[os.path.relpath(p, OUT).replace(os.sep, "/")] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    sources = {rel(p): sha(p) for p in SRC.values()}
    summaries = {}
    for key in ("model", "c2", "seed"):
        for r in csv.DictReader(open(SRC[key], encoding="utf-8")):
            p = r["summary_path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p); summaries[rel(p)] = sha(p)
    for r in csv.DictReader(open(SRC["hist"], encoding="utf-8")):
        if (float(r["det_thresh"]) == 0.5 and r["sequence"] == "seq_02_30sec_a" and r["c2"] == "off"
                and int(r["miss_tol"]) == 30 and r["run"] in ("Phase5", "Arm1", "Arm2")):
            p = os.path.join(ROOT, r["source_path"]); summaries[rel(p)] = sha(p)
    json.dump({"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "folder": "results_after_tuning/guide_presentation", "files": files, "source_csvs": sources,
               "trackeval_summaries": summaries}, open(dst, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(f"MANIFEST.json: {len(files)} folder files, {len(sources)} source CSVs, {len(summaries)} TrackEval summaries")

if __name__ == "__main__":
    main()
