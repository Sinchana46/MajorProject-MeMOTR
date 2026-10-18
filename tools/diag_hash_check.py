import json, hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor

HASH_FILE = "data/IOTD/check/hashes_before.json"
ROOT      = "data/IOTD"

def sha256file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

with open(HASH_FILE) as f:
    baseline = json.load(f)

# The hash file stores annotation files under key "annotation_files"
# but the real path has no such intermediate directory.
# img1 files: data/IOTD/<seq>/img1/<fname>
# annotation files (gt.txt, seqinfo.ini, labels.txt): data/IOTD/<seq>/gt/<fname> or data/IOTD/<seq>/<fname>
tasks = []
for seq, parts in baseline["sequences"].items():
    for part, files in parts.items():
        for fname, expected in files.items():
            if part == "img1":
                fpath = os.path.join(ROOT, seq, "img1", fname)
            else:  # annotation_files
                # fname can be "gt/gt.txt", "gt/labels.txt", or "seqinfo.ini"
                fpath = os.path.join(ROOT, seq, fname)
            fpath = fpath.replace("\\", "/")
            tasks.append((fpath, expected))

def check(item):
    path, expected = item
    if not os.path.exists(path):
        return path, "MISSING"
    try:
        h = sha256file(path)
        return path, "OK" if h == expected else "CHANGED"
    except Exception as e:
        return path, f"ERROR:{e}"

print(f"Checking {len(tasks)} files...")
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(check, tasks))

changed = [p for p,s in results if s == "CHANGED"]
missing = [p for p,s in results if s == "MISSING"]
errors  = [f"{p}:{s}" for p,s in results if s.startswith("ERROR")]

print(f"Total: {len(tasks)}  changed: {len(changed)}  missing: {len(missing)}  errors: {len(errors)}")
if changed: print("CHANGED:", changed[:3])
if missing: print("MISSING:", missing[:5])
if errors:  print("ERRORS:", errors[:3])

if len(changed) == 0 and len(missing) == 0 and len(errors) == 0:
    print("PRE-RUN HASH CHECK: PASS")
else:
    print("PRE-RUN HASH CHECK: FAIL")
    sys.exit(1)
