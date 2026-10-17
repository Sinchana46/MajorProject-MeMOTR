Corrected protocol: Arm 1 configuration, three training seeds at checkpoint_3 (n = 3; rough estimate). Corrected seed values exist only for checkpoint_3.

| Seed | Checkpoint | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| seed 42 (Arm 1) | ckpt_3 | **64.12** | **60.54** | **68.21** | **68.15** | **84.64** | 3 | 841 | 488 |
| seed 1234 | ckpt_3 | 61.07 | 57.02 | 65.81 | 60.64 | 81.25 | 3 | 1,297 | **346** |
| seed 2024 | ckpt_3 | 59.74 | 55.67 | 64.39 | 64.87 | 81.59 | **1** | **582** | 886 |
| mean ± SD | ckpt_3 | 61.64 ± 2.25 | 57.74 ± 2.51 | 66.14 ± 1.93 | 64.55 ± 3.76 | 82.49 ± 1.87 | 2.33 ± 1.15 | 906.67 ± 361.99 | 573.33 ± 279.93 |
| noise bar (mean + 2×SD) | ckpt_3 | 66.14 | – | – | – | – | – | – | – |

*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. Corrected protocol: upstream output filter score > 0.5, area > 100 px.*
