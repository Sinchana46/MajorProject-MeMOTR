Corrected protocol: candidate checkpoints and difference to Phase 5 (positive IDSW/FP/FN = more errors).

| Model | Checkpoint | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall % | Precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 5 | ckpt_4 | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | **0** | 940 | 1,097 | 73.77 | 76.65 |
| Arm 1 | ckpt_3 ★ | **64.12** | **60.54** | **68.21** | **68.15** | **84.64** | 3 | 841 | **488** | **88.33** | 81.45 |
| Δ vs Phase 5 | – | 6.38 | 10.67 | 0.91 | 16.86 | 9.46 | 3 | -99 | -609 | 14.56 | 4.81 |
| Arm 2 | ckpt_1 ★ | 61.61 | 58.10 | 65.51 | 67.43 | 82.83 | 3 | **601** | 758 | 81.88 | **85.07** |
| Δ vs Phase 5 | – | 3.87 | 8.22 | -1.79 | 16.14 | 7.65 | 3 | -339 | -339 | 8.11 | 8.42 |

*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. Corrected protocol: upstream output filter score > 0.5, area > 100 px.*
