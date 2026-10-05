"""
tools/validate_mot_schema.py — Cross-dataset MOT format schema validator (Phase 2)

PURPOSE
-------
Load all converted gt.txt files from UA-DETRAC and VisDrone_Vehicles and assert
that they conform to the expected MOT Challenge schema. Run this after both
converters have completed.

CHECKS PERFORMED
----------------
Per gt.txt row:
  1. Exactly 9 comma-separated columns
  2. x >= 0, y >= 0 (no negative top-left corner)
  3. w > 0, h > 0 (no zero-area boxes)
  4. col[6] (conf)  == 1 (our converters always write 1)
  5. col[7] (class) == 1 (MOT convention; our converters always write 1)
  6. col[8] (vis)   == 1

Per sequence:
  7. Frame IDs are positive integers (>= 1)
  8. seqinfo.ini exists and has all required keys

Per dataset split:
  9. Sequence-level split integrity: no sequence name appears in both train and test

Exit code: 0 if zero violations, 1 if any violation found.

USAGE
-----
  python tools/validate_mot_schema.py
  python tools/validate_mot_schema.py --roots data/UA_DETRAC data/VisDrone_Vehicles
"""

import os
import sys
import argparse
import configparser

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_ROOTS = [
    os.path.join(ROOT, "data", "UA_DETRAC"),
    os.path.join(ROOT, "data", "VisDrone_Vehicles"),
]

REQUIRED_INI_KEYS = {"name", "imdir", "framerate", "seqlength", "imwidth", "imheight", "imext"}


# ── Row-level validation ──────────────────────────────────────────────────────

def validate_gt_file(gt_path):
    """
    Validate one gt.txt file row by row.

    Returns
    -------
    list of dicts, each with keys: line_no, violation, value
    Empty list means no violations.
    """
    violations = []

    with open(gt_path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines

            parts = line.split(",")

            # Check 1: column count
            if len(parts) != 9:
                violations.append({
                    "line_no"  : line_no,
                    "violation": f"Expected 9 columns, got {len(parts)}",
                    "value"    : line[:80],
                })
                continue  # cannot safely check further columns

            try:
                frame = int(parts[0])
                _     = int(parts[1])   # track_id
                x     = float(parts[2])
                y     = float(parts[3])
                w     = float(parts[4])
                h     = float(parts[5])
                conf  = float(parts[6])
                cls   = float(parts[7])
                vis   = float(parts[8])
            except ValueError as e:
                violations.append({
                    "line_no"  : line_no,
                    "violation": f"Non-numeric value: {e}",
                    "value"    : line[:80],
                })
                continue

            # Check 2: frame index positive
            if frame < 1:
                violations.append({
                    "line_no"  : line_no,
                    "violation": f"frame_idx < 1: {frame}",
                    "value"    : line[:80],
                })

            # Check 3: non-negative top-left
            if x < 0:
                violations.append({"line_no": line_no, "violation": f"x < 0: {x}", "value": line[:80]})
            if y < 0:
                violations.append({"line_no": line_no, "violation": f"y < 0: {y}", "value": line[:80]})

            # Check 4: positive box size
            if w <= 0:
                violations.append({"line_no": line_no, "violation": f"w <= 0: {w}", "value": line[:80]})
            if h <= 0:
                violations.append({"line_no": line_no, "violation": f"h <= 0: {h}", "value": line[:80]})

            # Check 5–7: expected fixed values from our converters
            if conf != 1.0:
                violations.append({"line_no": line_no, "violation": f"conf != 1: {conf}", "value": line[:80]})
            if cls != 1.0:
                violations.append({"line_no": line_no, "violation": f"class != 1: {cls}", "value": line[:80]})
            if vis != 1.0:
                violations.append({"line_no": line_no, "violation": f"vis != 1: {vis}", "value": line[:80]})

    return violations


def validate_seqinfo(seqinfo_path):
    """
    Check seqinfo.ini has all required keys.

    Returns
    -------
    list of violation strings (empty = ok)
    """
    violations = []
    if not os.path.isfile(seqinfo_path):
        return [f"seqinfo.ini missing at {seqinfo_path}"]

    ini = configparser.ConfigParser()
    ini.read(seqinfo_path)

    if "Sequence" not in ini:
        return ["[Sequence] section missing from seqinfo.ini"]

    present_keys = set(ini["Sequence"].keys())
    missing = REQUIRED_INI_KEYS - present_keys
    if missing:
        violations.append(f"seqinfo.ini missing keys: {missing}")

    return violations


# ── Sequence enumeration ──────────────────────────────────────────────────────

def find_sequences(dataset_root):
    """
    Find all converted sequences under dataset_root.

    Returns
    -------
    list of (split_name, seq_name, gt_path, seqinfo_path)
    Looks for: dataset_root/{split}/{seq}/gt/gt.txt
    """
    sequences = []
    for split in sorted(os.listdir(dataset_root)):
        split_path = os.path.join(dataset_root, split)
        if not os.path.isdir(split_path):
            continue
        # Skip non-split directories (raw/, etc.)
        if split in ("raw",):
            continue
        for seq in sorted(os.listdir(split_path)):
            seq_path = os.path.join(split_path, seq)
            if not os.path.isdir(seq_path):
                continue
            gt_path      = os.path.join(seq_path, "gt", "gt.txt")
            seqinfo_path = os.path.join(seq_path, "seqinfo.ini")
            if os.path.isfile(gt_path):
                sequences.append((split, seq, gt_path, seqinfo_path))
    return sequences


# ── Split integrity check ─────────────────────────────────────────────────────

def check_split_integrity(sequences):
    """
    Verify no sequence name appears in more than one split.

    Returns list of violation strings.
    """
    from collections import defaultdict
    seq_to_splits = defaultdict(list)
    for split, seq, _, _ in sequences:
        seq_to_splits[seq].append(split)

    violations = []
    for seq, splits in seq_to_splits.items():
        if len(splits) > 1:
            violations.append(
                f"Sequence '{seq}' found in multiple splits: {splits}")
    return violations


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dataset_roots):
    total_violations  = 0
    total_sequences   = 0
    total_rows        = 0

    print("=" * 70)
    print("MOT Schema Validator — Phase 2 cross-dataset check")
    print("=" * 70)

    for dataset_root in dataset_roots:
        dataset_name = os.path.basename(dataset_root)

        if not os.path.isdir(dataset_root):
            print(f"\n[{dataset_name}] SKIP — directory not found: {dataset_root}")
            continue

        print(f"\n[{dataset_name}]")

        sequences = find_sequences(dataset_root)
        if not sequences:
            print("  WARNING: No converted sequences found (run converter first).")
            continue

        # Check split integrity across all sequences in this dataset
        integrity_violations = check_split_integrity(sequences)
        if integrity_violations:
            print("  SPLIT INTEGRITY VIOLATIONS:")
            for v in integrity_violations:
                print(f"    {v}")
            total_violations += len(integrity_violations)
        else:
            print(f"  Split integrity: OK ({len(sequences)} sequences found)")

        # Validate each sequence
        dataset_row_count   = 0
        dataset_violations  = 0
        violated_sequences  = []

        for split, seq, gt_path, seqinfo_path in sequences:
            # seqinfo.ini check
            ini_violations = validate_seqinfo(seqinfo_path)
            for v in ini_violations:
                dataset_violations += 1
                violated_sequences.append((split, seq, "seqinfo", v, ""))

            # gt.txt row-level check
            row_violations = validate_gt_file(gt_path)
            # Count rows
            with open(gt_path) as f:
                n_rows = sum(1 for line in f if line.strip())
            dataset_row_count += n_rows

            for v in row_violations:
                dataset_violations += 1
                violated_sequences.append(
                    (split, seq, gt_path, v["violation"], v.get("value", "")))

        total_rows      += dataset_row_count
        total_violations += dataset_violations
        total_sequences += len(sequences)

        if dataset_violations == 0:
            print(f"  Row-level checks: OK ({dataset_row_count:,} rows across "
                  f"{len(sequences)} sequences — zero violations)")
        else:
            print(f"  Row-level checks: {dataset_violations} VIOLATION(S)")
            for split, seq, path, violation, value in violated_sequences[:20]:
                print(f"    [{split}/{seq}] {violation}")
                if value:
                    print(f"      row: {value}")
            if len(violated_sequences) > 20:
                print(f"    ... and {len(violated_sequences)-20} more")

    print("\n" + "=" * 70)
    print("FINAL VALIDATION REPORT")
    print(f"  Datasets checked : {len(dataset_roots)}")
    print(f"  Sequences checked: {total_sequences}")
    print(f"  Rows checked     : {total_rows:,}")
    print(f"  Total violations : {total_violations}")
    if total_violations == 0:
        print("  STATUS           : PASS — zero schema violations")
    else:
        print("  STATUS           : FAIL — fix violations before Phase 3")
    print("=" * 70)

    return 0 if total_violations == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate converted MOT datasets for schema compliance")
    parser.add_argument(
        "--roots", nargs="+", default=DEFAULT_ROOTS,
        help="Dataset root directories to validate (default: UA_DETRAC + VisDrone_Vehicles)")
    args = parser.parse_args()

    sys.exit(main(args.roots))
