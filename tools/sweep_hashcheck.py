"""
tools/sweep_hashcheck.py — re-hash all real IOTD files against data/IOTD/check/hashes_before.json.
Prints changed / missing / added counts (all must be 0). Read-only.
"""
import os, sys, json, hashlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HASH_FILE = os.path.join(ROOT, "data", "IOTD", "check", "hashes_before.json")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

def main():
    raw = json.load(open(HASH_FILE, encoding="utf-8"))
    entries = {}
    for seq, d in raw["sequences"].items():
        for sub, files in d.items():
            if not isinstance(files, dict):
                continue
            for fn, h in files.items():
                rel = f"data/IOTD/{seq}/{fn}" if sub == "annotation_files" else f"data/IOTD/{seq}/{sub}/{fn}"
                entries[rel] = h
    def check(item):
        rel, h = item
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            return "missing", rel
        return ("ok" if sha(p) == h else "changed"), rel
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, entries.items()))
    changed = [r for s, r in res if s == "changed"]
    missing = [r for s, r in res if s == "missing"]
    added = []
    for seq in raw["sequences"]:
        for dp, _, fs in os.walk(os.path.join(ROOT, "data", "IOTD", seq)):
            for fn in fs:
                rel = os.path.relpath(os.path.join(dp, fn), ROOT).replace(os.sep, "/")
                if rel not in entries:
                    added.append(rel)
    print(f"hash manifest: {HASH_FILE}")
    print(f"files checked={len(entries)} changed={len(changed)} missing={len(missing)} added={len(added)}")
    for name, lst in (("changed", changed), ("missing", missing), ("added", added)):
        for r in lst[:10]:
            print(f"  {name}: {r}")
    sys.exit(0 if not (changed or missing or added) else 1)

if __name__ == "__main__":
    main()
