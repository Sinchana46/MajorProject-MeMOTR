# Gate 1 Resolution Report

**Written:** 11 October 2026  
**Decision branch:** Validation only — correct only `seq_02_30sec_a`, do not touch training data  
**Pre-run hash:** PASS (3,283 files, changed=0, missing=0) — `check/logs/gate1_hash_pre.txt`  
**Post-run hash:** PASS (3,283 files, changed=0, missing=0) — `check/logs/gate1_hash_post.txt`

---

## 1. Disposition of All 10 Candidates

Source: `check/annotation_audit/gate1/disposition_table.csv`  
Inference evidence from: Part A (`check/logs/gate1_partAB.txt`)  
Evidence images: `check/annotation_audit/gate1/cand_<id>_review.jpg`

### Part A — GT file structure (VERIFIED)

| Property | Value |
|---|---|
| File | `data/IOTD_val/seq_02_30sec_a/gt/gt.txt` |
| SHA256 | `17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472` |
| Size | 256,288 bytes |
| Rows | 4,182 |
| Frame range | 1–744 |
| ID range | 1–50 (max GT ID = 50) |
| Line endings | CRLF |
| Trailing newline | Yes |
| Format | `frame,id,x.xx,y.xx,w.xx,h.xx,1,class,1.0` |
| Decimal places | x/y=2, w/h=14 (float arithmetic artifact) |
| Sort order | frame ascending, then ID ascending |
| Class IDs present | 1 (car), 2 (two-wheeler), 3 (auto), 5 (truck) |
| Confidence | constant 1 |
| Visibility | constant 1.0 |
| Box size range | w=[34.6, 554.1] h=[73.1, 587.8] |

New track IDs must start at **51** (above existing max).

### Part A — Tracker source per candidate (VERIFIED)

| Candidate | Primary run | TID | All runs agreeing |
|---|---|---|---|
| seq_02_30sec_a_000 | phase5 | 6 | phase5 |
| seq_02_30sec_a_001 | phase5 | 16 | phase5 |
| seq_02_30sec_a_002 | phase5 | 21 | phase5 + arm1 + arm2 |
| seq_02_30sec_a_003 | phase5 | 27 | phase5 |
| seq_02_30sec_a_004 | phase5 | 34 | phase5 |
| seq_02_30sec_a_005 | phase5 | 42 | phase5 + arm1 + arm2 |
| seq_02_30sec_a_006 | arm1 | 2 | arm1 |
| seq_02_30sec_a_007 | arm1 | 18 | arm1 + arm2 |
| seq_02_30sec_a_008 | arm1 | 31 | arm1 + arm2 |
| seq_02_30sec_a_009 | arm2 | 2 | arm2 |

### Part B — Final disposition table (VERIFIED — pixel-level + geometric review)

| Candidate | Frames | Dir | IoU with GT | Disposition | Confidence | Reason |
|---|---:|---|---:|---|---|---|
| seq_02_30sec_a_000 | 56 | UP | 0.000 | EXCLUDE_POLICY | **high** | Moves UP opposite to all 50 GT tracks (DOWN); cx=366px; consistent with opposite-carriageway vehicle |
| seq_02_30sec_a_001 | 47 | DOWN | 0.000 | UNCERTAIN | medium | Single-run detection; moves DOWN; cx=122px (near frame edge); needs human CVAT review |
| seq_02_30sec_a_002 | 67 | UP | 0.000 | EXCLUDE_POLICY | **high** | Moves UP; 3 runs agree; cx=1240px; consistent with opposite-carriageway vehicle |
| seq_02_30sec_a_003 | 32 | UP | 0.000 | EXCLUDE_POLICY | **high** | Moves UP; 1 run; cx=272px; consistent with opposite-carriageway vehicle |
| seq_02_30sec_a_004 | 93 | UP | 0.000 | EXCLUDE_POLICY | **high** | Moves UP; 1 run; cx=391px, cy=150px (near top of frame); consistent with opposite-carriageway |
| seq_02_30sec_a_005 | 39 | DOWN | 0.080 | GAP_FILL | medium | Moves DOWN; 3 runs agree; GT tracks 34+35 have 13/38 frame gaps in [625–663]; position overlap possible — needs pixel check in CVAT |
| seq_02_30sec_a_006 | 31 | DOWN | 0.044 | GAP_FILL | medium | Moves DOWN; 1 run; GT tracks 1+7 have 9/11 frame gaps in [1–31]; single run only |
| seq_02_30sec_a_007 | 76 | DOWN | 0.087 | GAP_FILL | medium | Moves DOWN; 2 runs agree; GT tracks 8+10 have 63/47 frame gaps in [149–224]; most persistent DOWN candidate |
| seq_02_30sec_a_008 | 66 | UP | 0.033 | EXCLUDE_POLICY | **high** | Moves UP; 2 runs; cx=1090px; consistent with opposite-carriageway vehicle |
| seq_02_30sec_a_009 | 30 | DOWN | 0.128 | GAP_FILL | medium | Moves DOWN; 1 run; GT tracks 1+7 have 8/11 frame gaps in [1–30]; IoU=0.128 (not a duplicate) |

**Summary:**
- EXCLUDE_POLICY (high): 5 candidates (000, 002, 003, 004, 008)
- GAP_FILL (medium): 4 candidates (005, 006, 007, 009) — NOT APPLIED
- UNCERTAIN (medium): 1 candidate (001) — NOT APPLIED

> **Rule (from gate1_resolution.md):** "Only ADD and GAP_FILL with high confidence may be applied."
> No candidate reached high confidence for ADD or GAP_FILL. **No annotations were modified.**

### seq_01_1min confirmation (from Part B audit)

The 5 LIKELY_OPPOSITE candidates in `seq_01_1min` all move UP while all 37 GT tracks move DOWN. Their cx values (100–400px range) place them outside the GT lateral span (770–1068px). Evidence images: `check/annotation_audit/seq_01_1min/cand_seq_01_1min_00*_LIKELY_OPPOSITE.jpg`.  
**VERIFIED: these are opposite-carriageway vehicles, correctly excluded by policy. No action needed.**

---

## 2. Annotation Corrections Applied

**None.** The original `seq_02_30sec_a/gt/gt.txt` was not modified.

---

## 3. Backup and SHA256 Verification

| File | Path | SHA256 |
|---|---|---|
| Original GT | `data/IOTD_val/seq_02_30sec_a/gt/gt.txt` | `17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472` |
| Backup copy | `check/annotation_audit/backup/seq_02_30sec_a_gt_ORIGINAL.txt` | `17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472` |
| Match | **YES** — byte-for-byte identical |

**VERIFIED** (`check/logs/gate1_partCD.txt`).

No training annotation, checkpoint, config, augmented folder, seqinfo.ini, or labels.txt was changed.  
Post-run hash: **PASS** (3,283 files, changed=0, missing=0, added=0).

---

## 4. Part D — Completeness Estimate

**Method:** 20 evenly spaced frames (f0001, f0039, f0078, … f0744), automated edge-cluster heuristic (connected components of Canny edges ≥ 3,000 px area in non-GT regions, below frame top 25%). This is an estimate, **not ground truth**.

| Frames checked | GT boxes/frame (mean) | Potential missed regions/frame (heuristic) |
|---|---|---|
| 20 | 5.6 | 9.7 |

Heuristic found **≥1 potential missed region in every one of the 20 frames**. Total flagged: 193 across 20 frames.

> **CAVEAT:** The heuristic uses large edge clusters as a proxy for vehicles. Many clusters are walls, road markings, infrastructure, and parked objects — not annotatable vehicles. The true number of genuinely missing, policy-conforming vehicle annotations is likely much smaller. This estimate is a **worst case upper bound**.

**Recommendation:** The existing GT is adequate for comparative model ranking (5–7 GT-tracked vehicles per frame are consistently annotated). However, for publication-quality evaluation, a full manual annotation pass in CVAT by the user is recommended, focused on the 4 GAP_FILL candidates (005, 006, 007, 009) and the 1 UNCERTAIN candidate (001). These are the only candidates where automation cannot resolve the question from pixels alone.

**Review pack location:** `check/annotation_audit/gate1/review_pack/`  
Contains: disposition table, all 10 evidence renders, 20 completeness check sheets.

---

## 5. Part E — Re-scoring Tables

**S1 = S0:** No corrected GT produced. Evaluation uses the original GT throughout.  
**S2:** Sensitivity analysis only — rows removed from tracker output at IoU ≥ 0.5 with GAP_FILL candidates. Since these predictions already score as FP in S0 (IoU < 0.2 with any GT box by definition from the Part B2 filter), removing them in S2 reduces FP count but adds no TP. HOTA ranking is unchanged.

### S0 / S1 Full Results Table (original GT, det/track=0.5, miss_tol=30)

| Run | Steps | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall | Prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase5_ckpt4 | 0 | 50.51 | 43.88 | 58.49 | 29.10 | 68.84 | 1 | 2,117 | 847 | 79.75 | 61.17 |
| arm1_ckpt0 | 2,528 | 45.72 | 42.49 | 49.58 | 17.02 | 63.08 | 21 | 2,847 | 602 | 85.61 | 55.70 |
| arm1_ckpt1 | 5,056 | 49.30 | 46.42 | 52.76 | 30.03 | 68.38 | 14 | 2,499 | 413 | 90.12 | 60.13 |
| arm1_ckpt2 | 7,584 | 48.87 | 45.92 | 52.27 | 27.83 | 68.43 | 12 | 2,643 | 363 | 91.32 | 59.10 |
| **arm1_ckpt3** | **10,112** | **50.47** | **46.96** | **54.51** | **29.63** | **70.23** | **9** | **2,629** | **305** | **92.71** | **59.59** |
| arm1_ckpt4 | 12,640 | 49.58 | 45.70 | 54.04 | 24.94 | 69.66 | 7 | 2,855 | 277 | 93.38 | 57.77 |
| arm2_ckpt0 | 10,112 | 40.51 | 36.49 | 45.15 | 16.28 | 56.63 | 10 | 1,964 | 1,527 | 63.49 | 57.48 |
| **arm2_ckpt1** | **20,224** | **48.17** | **45.54** | **51.13** | **29.84** | **68.64** | **12** | **2,432** | **490** | **88.28** | **60.29** |
| arm2_ckpt2 | 30,336 | 44.77 | 41.22 | 48.77 | 8.42 | 67.13 | 3 | 3,671 | 156 | 96.27 | 52.31 |
| arm2_ckpt3 | 40,448 | 43.41 | 39.18 | 48.21 | -3.97 | 64.56 | 3 | 4,225 | 120 | 97.13 | 49.02 |
| arm2_ckpt4 | 50,560 | 42.84 | 38.32 | 48.01 | -6.58 | 63.72 | 6 | 4,266 | 185 | 95.58 | 48.37 |

### Threshold 0.7 results (ckpt4 / phase5 only)

| Run | Thresh | HOTA | DetA | MOTA | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| phase5_ckpt4 | 0.7 | 41.63 | 30.74 | 23.17 | 776 | 2,436 |
| arm1_ckpt4 | 0.7 | 48.06 | 44.02 | 32.16 | 1,859 | 972 |
| arm2_ckpt4 | 0.7 | 47.46 | 43.92 | 23.20 | 2,813 | 396 |

All numbers from on-disk `pedestrian_summary.txt` files. **VERIFIED** (`check/logs/gate1_partE.txt`).

---

## 6. Checkpoint Selection Rule — Applied

> Rule: Evaluate all checkpoints (ckpt_0 through ckpt_last) at det/track=0.5. Select checkpoint with highest HOTA.

| Arm | Best checkpoint | HOTA | Steps | Notes |
|---|---|---:|---:|---|
| Arm 1 | **arm1_ckpt3** | **50.47** | 10,112 | Pre-LR-drop; highest HOTA |
| Arm 2 | **arm2_ckpt1** | **48.17** | 20,224 | HOTA peaks here; degrades after augmentation saturates |

**Best checkpoint per arm is the same under S0, S1 (=S0), and S2.** Rankings are stable.

> **Required caveat (from Part E rule):** The corrected GT (had one been produced) would have been built using the same models being scored. The best checkpoint was also selected on the same single validation sequence. Both make the numbers optimistic. Selecting a checkpoint or threshold on `seq_02_30sec_a` is tuning on the validation sequence.

---

## 7. Unresolved Cases — For User Review

The following require human visual review in CVAT before they can be resolved:

| Candidate | Evidence image | Issue |
|---|---|---|
| seq_02_30sec_a_007 (76 frames, 2 runs, DOWN) | `gate1/cand_seq_02_30sec_a_007_review.jpg` | Most persistent DOWN candidate; GT tracks 8+10 have 63-frame gap in frames 149–224; position overlap possible — is it a continuation of one of these tracks, or a new unannotated vehicle? |
| seq_02_30sec_a_005 (39 frames, 3 runs, DOWN) | `gate1/cand_seq_02_30sec_a_005_review.jpg` | 3 runs agree; GT tracks 34+35 have 13–38 frame gaps in [625–663]; same question |
| seq_02_30sec_a_001 (47 frames, 1 run, DOWN) | `gate1/cand_seq_02_30sec_a_001_review.jpg` | cx=122px (near left edge); single-run detection; could be a pedestrian or out-of-policy vehicle |
| seq_02_30sec_a_006 (31 frames, 1 run, DOWN) | `gate1/cand_seq_02_30sec_a_006_review.jpg` | Single run; frames 1–31; possible GT track 1 extension |
| seq_02_30sec_a_009 (30 frames, 1 run, DOWN) | `gate1/cand_seq_02_30sec_a_009_review.jpg` | Single run; frames 1–30; possible GT track 1 extension |

**Instruction:** Load `check/annotation_audit/cvat_import/seq_02_30sec_a_review_add.txt` into CVAT on sequence `seq_02_30sec_a`, toggle candidate tracks visible, and verify against the video. Specifically check frames 149–224 for candidate 007 (the most persistent), comparing against GT tracks 8 and 10.

---

## 8. Data Facts Correction

From `antigravity_gate1_resolution.md` Section F.8 (mandatory):

| Sequence | Frames | Role |
|---|---:|---|
| seq_01_1min | 1,599 | Real training |
| seq_03_30sec_b | **931** | Real training |
| seq_02_30sec_a | 744 | Real validation |
| **Total real annotated** | **3,274** | — |
| **Real training frames** | **2,530** | seq_01 + seq_03 |
| **Real validation frames** | **744** | seq_02 only |

> **Correction to previous report caveat 3:** the prior report incorrectly stated "seq_03 1,675 frames". The correct value is **931 frames** (VERIFIED from `seqinfo.ini`). Total real training frames = 2,530 (not 3,274 − 744 = 2,530; same result, but seq_03=931 is the correct per-sequence number).  
> Augmented copies in `IOTD_train_aug` add 6 × 2,530 = **15,180 augmented training samples**, but no new ground-truth annotations. Real annotated data remains **3,274 frames** total.

---

## 9. Status

**STATUS: PARTIAL**

| Item | Status |
|---|---|
| Pre-run hash | PASS |
| Post-run hash | PASS |
| Original GT unchanged | CONFIRMED (SHA256 identical before/after) |
| Training data unchanged | CONFIRMED (hash covers all IOTD files) |
| Checkpoints unchanged | CONFIRMED |
| Part A — inputs verified | COMPLETE |
| Part B — all 10 candidates reviewed | COMPLETE |
| Part C — annotation correction | NOT APPLIED (no high-confidence ADD/GAP_FILL) |
| Part D — completeness estimate | COMPLETE (heuristic suggests full CVAT pass recommended) |
| Part E — re-scoring S0 | COMPLETE (11 runs, matches historical numbers) |
| Part E — re-scoring S1 | N/A (= S0, no corrected GT) |
| Part E — re-scoring S2 | COMPLETE (rankings unchanged) |
| Checkpoint selection applied | COMPLETE: Arm1=ckpt3(50.47), Arm2=ckpt1(48.17) |
| 5 UP candidates | Confirmed EXCLUDE_POLICY (opposite carriageway) |
| 4 DOWN GAP_FILL | Confirmed medium confidence — in review pack for user |
| 1 UNCERTAIN | In review pack for user |

**Reason for PARTIAL (not PASS):** 4 GAP_FILL candidates and 1 UNCERTAIN candidate could not be resolved to high confidence from automated analysis alone. The evaluation GT and all model rankings are unaffected by this — the original GT is used throughout. The partial status reflects the open annotation questions, not any evaluation or training problem.

**Recommended next step:**
1. **(Optional but recommended)** Open `check/annotation_audit/gate1/review_pack/` in CVAT and resolve the 5 unresolved candidates (007, 005, 001, 006, 009).
2. **GATE 2 — Seed repeats:** Reply "launch seeds" to start `train_iotd_real_seed1.yaml` and `train_iotd_real_seed2.yaml`. Each costs ~2h 20m. This measures run-to-run variability before hyperparameter tuning.
3. After seed repeats complete, the minimum hyperparameter tuning plan (Section 7 of the tuning readiness report) can begin.

> Do not launch seed repeats, hyperparameter tuning, or any training job without explicit user instruction.
