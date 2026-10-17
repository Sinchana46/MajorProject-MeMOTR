| Run | Checkpoint | Steps | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | Recall % | Precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phase 5 | ckpt_4 | 0 | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | **0** | 940 | 1,097 | 73.77 | 76.65 |
| Arm 1 | ckpt_3 ★ | 10,112 | 64.12 | **60.54** | 68.21 | 68.15 | **84.64** | 3 | 841 | 488 | 88.33 | 81.45 |
| Arm 2 | ckpt_1 | 20,224 | 61.61 | 58.10 | 65.51 | 67.43 | 82.83 | 3 | **601** | 758 | 81.88 | 85.07 |
| assoc_lr0 | ckpt_0 | 2,528 | 62.76 | 59.37 | 66.62 | 64.28 | 82.47 | 5 | 1,134 | **355** | **91.51** | 77.14 |
| lr_half | ckpt_0 | 2,528 | **64.26** | 59.95 | **69.23** | **69.20** | 83.67 | 4 | 607 | 677 | 83.81 | **85.24** |

*Corrected protocol. Val: seq_02_30sec_a. Best HOTA marked ★. ★ = chosen model (Arm 1 ckpt_3); lr_half ckpt_0 is 0.14 higher in a single, unstable run inside the seed noise.*
