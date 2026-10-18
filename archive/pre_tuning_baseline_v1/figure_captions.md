# Figure captions (measured facts only)

**Figure 1.** HOTA against optimizer steps for Arm 1 (real IOTD frames, 2,528 steps/epoch) and Arm 2 (augmented, 10,112 steps/epoch). Dashed line: Phase 5 (HOTA 50.51). Stars: highest-HOTA checkpoint per arm (Arm 1 ckpt_3, 50.47; Arm 2 ckpt_1, 48.17). Dotted lines: learning-rate drop. At 10,112 steps Arm 1 ckpt_3 exceeds Arm 2 ckpt_0 (40.51) by 9.96 HOTA.

**Figure 2.** DetA (top) and AssA (bottom) against optimizer steps. Phase 5: DetA 43.88, AssA 58.49. No Arm 1 or Arm 2 checkpoint reaches the Phase 5 AssA; the highest is 54.51.

**Figure 3.** False positives (top) and false negatives (bottom) against optimizer steps. Phase 5: FP 2,117, FN 847. Arm 1 FN falls from 602 to 277; Arm 2 FP rises from 1,964 to 4,266.

**Figure 4.** Detection outcomes from per-frame Hungarian matching (IoU ≥ 0.5) for Phase5 ckpt_4: TP 3,335, FP 2,117, FN 847, precision 61.17%, recall 79.75%; Arm1 ckpt_3: TP 3,877, FP 2,629, FN 305, precision 59.59%, recall 92.71%; Arm2 ckpt_1: TP 3,692, FP 2,432, FN 490, precision 60.29%, recall 88.28%. True negatives are not defined for single-class detection.

**Figure 5.** HOTA, DetA, AssA, MOTA and IDF1 of the three candidates. HOTA: Phase 5 50.51, Arm 1 ckpt_3 50.47, Arm 2 ckpt_1 48.17.

**Figure 6.** Identity metrics of the three candidates. IDSW: 1, 9, 12; AssA: 58.49, 54.51, 51.13; IDF1: 68.84, 70.23, 68.64 (Phase 5, Arm 1 ckpt_3, Arm 2 ckpt_1).

**Figure 7.** HOTA of three training runs of the Arm 1 configuration (seeds 42, 1234, 2024) against optimizer steps; thick line: mean; band: ±1 sample SD (n = 3). Mean ± SD per checkpoint: 46.78 ± 1.41, 48.94 ± 0.83, 49.40 ± 0.60, 48.94 ± 1.34, 48.99 ± 0.99. Range at the same step count up to 2.67 HOTA.

**Figure 8.** Inference-time settings for Arm 1 ckpt_3 (det/track threshold and miss tolerance). HOTA ranges from 45.52 to 55.45 (S_F); S_A (0.5/0.5/30) 50.47. Dash-dot line: seed mean + 2 SD at ckpt_3 (51.63). Settings passing the decision rule: S_D.

**Figure 9.** Phase 5 ckpt_4 with C2 off (fixed τ = 0.5) and on (adapter epoch 4) on the three IOTD sequences. HOTA off → on: seq_01_1min 27.44 → 7.71; seq_02_30sec_a 50.51 → 45.76; seq_03_30sec_b 29.82 → 30.72. Mean τ with C2 on: seq_01_1min 0.311; seq_02_30sec_a 0.332; seq_03_30sec_b 0.404.
