Corrected protocol: training variants against the Arm 1 seed noise bar (mean + 2×SD at ckpt_3 = 66.14).

| Run | Checkpoint | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall % | Precision % | Passed rule | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 5 | ckpt_4 | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | **0** | 940 | 1,097 | 73.77 | 76.65 | reference | pre-IOTD baseline |
| Arm 1 (seed mean, n=3) | ckpt_3 | 61.64 | 57.74 | 66.14 | 64.55 | 82.49 | 2.33 | 906.67 | 573.33 | 86.29 | 80.39 | reference | default configuration |
| assoc_lr0 | ckpt_0 | 62.76 | 59.37 | 66.62 | 64.28 | 82.47 | 5 | 1,134 | **355** | **91.51** | 77.14 | NO | best 62.76 < bar 66.14; collapses at ckpt_1 (IDSW 21) |
| lr_half | ckpt_0 | **64.26** | **59.95** | **69.23** | **69.20** | **83.67** | 4 | **607** | 677 | 83.81 | **85.24** | NO | best 64.26 < bar 66.14; single run; collapses at ckpt_1 (IDSW 21) |

*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. Corrected protocol: upstream output filter score > 0.5, area > 100 px.*
