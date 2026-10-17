Corrected protocol: inference settings on Arm 1 ckpt_3. Rule: HOTA $>$ 66.14, IDSW ≤ S_A and ≥ 1 HOTA over S_A.

| Setting | det/track/miss | HOTA | FP | FN | IDSW | Recall % | Precision % | Decision rule |
|---|---|---|---|---|---|---|---|---|
| S_A | 0.5/0.5/30 | 64.12 | 841 | 488 | 3 | 88.33 | 81.45 | reference (retained) |
| S_B | 0.6/0.6/30 | 62.10 | 723 | 660 | 8 | 84.22 | 82.97 | FAIL |
| S_C | 0.7/0.7/30 | 60.19 | **393** | 1,138 | 5 | 72.79 | **88.57** | FAIL |
| S_D | 0.5/0.5/20 | 63.89 | 903 | **430** | **2** | **89.72** | 80.60 | FAIL |
| S_E | 0.5/0.5/40 | 62.77 | 736 | 644 | 6 | 84.60 | 82.78 | FAIL |
| S_F | 0.6/0.6/20 | **64.25** | 770 | 497 | 6 | 88.12 | 82.72 | FAIL |
| S_G | 0.7/0.7/20 | 59.83 | 423 | 1,047 | 9 | 74.96 | 88.11 | FAIL |

*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. Corrected protocol: upstream output filter score > 0.5, area > 100 px.*
