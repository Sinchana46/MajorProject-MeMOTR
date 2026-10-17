The same checkpoints scored under two DIFFERENT protocols. The difference is caused by the output filter, not by any change to the model; it is not an improvement.

| Model | Checkpoint | Protocol | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall % | Precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 5 | ckpt_4 | HISTORICAL (unfiltered) | 50.51 | 43.88 | 58.49 | 29.10 | 68.84 | 1 | 2,117 | 847 | 79.75 | 61.17 |
| Phase 5 | ckpt_4 | CORRECTED (filtered) | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | 0 | 940 | 1,097 | 73.77 | 76.65 |
| Arm 1 | ckpt_3 | HISTORICAL (unfiltered) | 50.47 | 46.96 | 54.51 | 29.63 | 70.23 | 9 | 2,629 | 305 | 92.71 | 59.59 |
| Arm 1 | ckpt_3 | CORRECTED (filtered) | 64.12 | 60.54 | 68.21 | 68.15 | 84.64 | 3 | 841 | 488 | 88.33 | 81.45 |

*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. Corrected protocol: upstream output filter score > 0.5, area > 100 px.*
