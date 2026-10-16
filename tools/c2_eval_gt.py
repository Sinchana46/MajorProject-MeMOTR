"""
tools/c2_eval_gt.py — evaluation-only GT copies for the C2 three-sequence evaluation (read-only on data/).

For each IOTD sequence: copy gt/gt.txt with column 8 (class) set to 1 for every row, and copy seqinfo.ini,
to results/c2_eval/gt_eval/<seq>/. Same conversion as results/eval_iotd_val/gt_eval (checked by reproducing
the seq_02 copy's SHA256). Prints row count and SHA256 of every original (must equal hashes_before.json).
Also creates per-sequence split folders (directory junctions) under results/c2_eval/splits/<seq>/<seq>
so tools/run_inference_mot.py sees exactly one sequence.
"""
import os, sys, json, hashlib, shutil, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "c2_eval")
SEQS = {
    "seq_01_1min":    os.path.join(ROOT, "data", "IOTD_train_real", "seq_01_1min"),
    "seq_02_30sec_a": os.path.join(ROOT, "data", "IOTD_val", "seq_02_30sec_a"),
    "seq_03_30sec_b": os.path.join(ROOT, "data", "IOTD_train_real", "seq_03_30sec_b"),
}
HASHES = json.load(open(os.path.join(ROOT, "data", "IOTD", "check", "hashes_before.json"), encoding="utf-8"))
EXISTING_SEQ02_EVAL = os.path.join(ROOT, "results", "eval_iotd_val", "gt_eval", "seq_02_30sec_a", "gt", "gt.txt")

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def convert(src, dst):
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            parts = line.strip().split(",")
            if len(parts) >= 8:
                parts[7] = "1"
                fout.write(",".join(parts) + "\n")

def main():
    for seq, d in SEQS.items():
        gt = os.path.join(d, "gt", "gt.txt")
        rows = sum(1 for l in open(gt) if l.strip())
        s = sha(gt); exp = HASHES["sequences"][seq]["annotation_files"]["gt/gt.txt"]
        print(f"{seq}: original {gt}  rows={rows}  sha256={s}  matches hashes_before={s == exp}")
        dst_dir = os.path.join(OUT, "gt_eval", seq, "gt")
        dst = os.path.join(dst_dir, "gt.txt")
        if os.path.exists(dst):
            print(f"  eval copy exists, not overwritten: {dst} sha256={sha(dst)}")
        else:
            os.makedirs(dst_dir, exist_ok=True)
            convert(gt, dst)
            shutil.copy2(os.path.join(d, "seqinfo.ini"), os.path.join(OUT, "gt_eval", seq, "seqinfo.ini"))
            print(f"  eval copy written: {dst} rows={sum(1 for _ in open(dst))} sha256={sha(dst)}")
        classes = {}
        for l in open(gt):
            p = l.strip().split(",")
            if len(p) >= 8:
                classes[p[7]] = classes.get(p[7], 0) + 1
        print(f"  original class column counts: {dict(sorted(classes.items()))}")
        split = os.path.join(OUT, "splits", seq)
        link = os.path.join(split, seq)
        if not os.path.exists(link):
            os.makedirs(split, exist_ok=True)
            r = subprocess.run(["cmd", "/c", "mklink", "/J", link, d], capture_output=True, text=True)
            print(f"  junction: {r.stdout.strip() or r.stderr.strip()}")
        print(f"  split dir {split} -> {os.listdir(split)}")
    a = sha(os.path.join(OUT, "gt_eval", "seq_02_30sec_a", "gt", "gt.txt")); b = sha(EXISTING_SEQ02_EVAL)
    print(f"\nconversion check: new seq_02 copy sha256={a}")
    print(f"                  existing results/eval_iotd_val/gt_eval copy sha256={b}  identical={a == b}")
    sys.exit(0 if a == b else 1)

if __name__ == "__main__":
    main()
