"""
verify_archive.py — recompute SHA256 of every file in archive/pre_tuning_baseline_v1 against MANIFEST.json and
report changed / missing / added (all must be 0). With --sources also re-checks the project source files.
Read-only.
"""
import os, sys, json, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(ARCHIVE, "..", ".."))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

def main():
    m = json.load(open(os.path.join(ARCHIVE, "MANIFEST.json"), encoding="utf-8"))
    exp = m["archive_files"]
    found = {}
    for dp, _, fs in os.walk(ARCHIVE):
        for fn in fs:
            rel = os.path.relpath(os.path.join(dp, fn), ARCHIVE).replace(os.sep, "/")
            if rel == "MANIFEST.json" or "__pycache__" in rel:
                continue
            found[rel] = os.path.join(dp, fn)
    changed = [r for r in exp if r in found and sha(found[r]) != exp[r]["sha256"]]
    missing = [r for r in exp if r not in found]
    added = [r for r in found if r not in exp]
    print(f"archive files: manifest={len(exp)} on disk={len(found)}  changed={len(changed)} missing={len(missing)} "
          f"added={len(added)}")
    for name, lst in (("changed", changed), ("missing", missing), ("added", added)):
        for r in lst:
            print(f"  {name}: {r}")
    bad = bool(changed or missing or added)
    if "--sources" in sys.argv:
        sc = [r for r, h in m["source_files"].items()
              if not os.path.exists(os.path.join(ROOT, r)) or sha(os.path.join(ROOT, r)) != h]
        print(f"source files: {len(m['source_files'])} checked, changed or missing={len(sc)}")
        for r in sc:
            print(f"  source changed/missing: {r}")
        bad |= bool(sc)
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
