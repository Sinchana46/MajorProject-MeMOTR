"""
tools/create_junctions_and_verify.py
------------------------------------
Part C2:
  1. Create junction parents: data/IOTD_train_real, data/IOTD_train_aug, data/IOTD_val
  2. Create junctions
  3. Print listing of each parent
  4. Re-hash originals against check/hashes_before.json
"""
import os, sys, json, hashlib, subprocess

MAJOR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(MAJOR, "data")
IOTD = os.path.join(DATA, "IOTD")
AUG_V2 = os.path.join(IOTD, "augmented_train_v2")
CHECK = os.path.join(IOTD, "check")

def make_junction(target, link):
    try:
        import _winapi
        _winapi.CreateJunction(target, link)
        return True, "via _winapi.CreateJunction"
    except Exception as e1:
        res = subprocess.run(["cmd.exe", "/c", "mklink", "/J", link, target], capture_output=True, text=True)
        if res.returncode == 0:
            return True, "via mklink /J"
        return False, f"err1={e1}, err2={res.stderr.strip()}"

plan = {
    os.path.join(DATA, "IOTD_train_real"): [
        ("seq_01_1min", os.path.join(IOTD, "seq_01_1min")),
        ("seq_03_30sec_b", os.path.join(IOTD, "seq_03_30sec_b")),
    ],
    os.path.join(DATA, "IOTD_train_aug"): [
        ("seq_01_1min_aug_01", os.path.join(AUG_V2, "seq_01_1min_aug_01")),
        ("seq_01_1min_aug_02", os.path.join(AUG_V2, "seq_01_1min_aug_02")),
        ("seq_01_1min_aug_03", os.path.join(AUG_V2, "seq_01_1min_aug_03")),
        ("seq_03_30sec_b_aug_01", os.path.join(AUG_V2, "seq_03_30sec_b_aug_01")),
        ("seq_03_30sec_b_aug_02", os.path.join(AUG_V2, "seq_03_30sec_b_aug_02")),
        ("seq_03_30sec_b_aug_03", os.path.join(AUG_V2, "seq_03_30sec_b_aug_03")),
    ],
    os.path.join(DATA, "IOTD_val"): [
        ("seq_02_30sec_a", os.path.join(IOTD, "seq_02_30sec_a")),
    ],
}

print("=== C2: CREATING JUNCTIONS ===", flush=True)
for parent_dir, items in plan.items():
    print(f"\nParent: {parent_dir}", flush=True)
    os.makedirs(parent_dir, exist_ok=True)
    for link_name, target_path in items:
        link_path = os.path.join(parent_dir, link_name)
        if os.path.exists(link_path):
            print(f"  ALREADY EXISTS: {link_name}", flush=True)
            continue
        ok, msg = make_junction(target_path, link_path)
        if ok:
            print(f"  CREATED JUNCTION: {link_name} -> {target_path} ({msg})", flush=True)
        else:
            print(f"  FAIL: {link_name}: {msg}", flush=True)
            sys.exit(1)

print("\n=== C2: LISTING OF EACH PARENT ===", flush=True)
for parent_dir in plan.keys():
    rel_parent = os.path.relpath(parent_dir, MAJOR)
    print(f"\n{rel_parent}/:", flush=True)
    for entry in sorted(os.listdir(parent_dir)):
        entry_path = os.path.join(parent_dir, entry)
        is_dir = os.path.isdir(entry_path)
        img1 = os.path.join(entry_path, "img1")
        n_imgs = len(os.listdir(img1)) if os.path.isdir(img1) else 0
        gt_exists = os.path.isfile(os.path.join(entry_path, "gt", "gt.txt"))
        print(f"  {entry:<25} (dir={is_dir}, frames in img1={n_imgs}, gt.txt={gt_exists})", flush=True)

print("\n=== C2: RE-HASH ORIGINALS VS hashes_before.json ===", flush=True)
def sha256f(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()

with open(os.path.join(CHECK, "hashes_before.json")) as f:
    before = json.load(f)

changed = 0
missing = 0
added = 0
for seq in ["seq_01_1min", "seq_02_30sec_a", "seq_03_30sec_b"]:
    if seq not in before["sequences"]:
        continue
    seq_dir = os.path.join(IOTD, seq)
    img1_dir = os.path.join(seq_dir, "img1")
    b_ann = before["sequences"][seq]["annotation_files"]
    b_img = before["sequences"][seq]["img1"]

    for rel, bh in b_ann.items():
        ah = sha256f(os.path.join(seq_dir, *rel.split("/")))
        if ah.lower() != bh.lower():
            print(f"  CHANGED: {seq}/{rel}", flush=True)
            changed += 1

    cur_jpegs = {
        f: sha256f(os.path.join(img1_dir, f))
        for f in os.listdir(img1_dir)
        if f.endswith(".jpg")
    }
    for fname, bh in b_img.items():
        if fname not in cur_jpegs:
            print(f"  MISSING: {seq}/img1/{fname}", flush=True)
            missing += 1
        elif cur_jpegs[fname].lower() != bh.lower():
            print(f"  CHANGED: {seq}/img1/{fname}", flush=True)
            changed += 1
    for fname in cur_jpegs:
        if fname not in b_img:
            print(f"  ADDED:   {seq}/img1/{fname}", flush=True)
            added += 1

print(f"changed={changed}  missing={missing}  added={added}", flush=True)
if changed + missing + added == 0:
    print("RE-HASH ORIGINALS: PASS (0 changes)", flush=True)
else:
    print("RE-HASH ORIGINALS: FAIL", flush=True)
    sys.exit(1)
