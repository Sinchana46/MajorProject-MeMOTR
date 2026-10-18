"""
make_manifest.py — write MANIFEST.json: SHA256 of every file in the archive (except MANIFEST.json itself)
and of every project source file the archive was generated from. Run once, before freezing.
"""
import os, sys, json, hashlib, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ARCHIVE, "..", ".."))
sys.path.insert(0, HERE)
from generate_all import SRC, GT_FILES, SEQINFO  # noqa: E402
import csv  # noqa: E402

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

def main():
    dst = os.path.join(ARCHIVE, "MANIFEST.json")
    if os.path.exists(dst):
        print(f"REFUSE: {dst} exists"); sys.exit(1)
    files = {}
    for dp, _, fs in os.walk(ARCHIVE):
        for fn in fs:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, ARCHIVE).replace(os.sep, "/")
            if rel == "MANIFEST.json" or "__pycache__" in rel:
                continue
            files[rel] = {"sha256": sha(full), "bytes": os.path.getsize(full)}
    sources = {}
    srcs = set(SRC.values()) | set(GT_FILES.values()) | set(SEQINFO.values())
    with open(os.path.join(ARCHIVE, "data", "results_registry_v1.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            srcs.add(r["source_path"])
    for rel in sorted(srcs):
        sources[rel] = sha(os.path.join(ROOT, rel))
    m = {"created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "archive": "archive/pre_tuning_baseline_v1", "archive_files": files, "source_files": sources}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    print(f"MANIFEST.json: {len(files)} archive files, {len(sources)} source files")

if __name__ == "__main__":
    main()
