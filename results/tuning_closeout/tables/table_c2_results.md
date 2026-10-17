Corrected protocol: Phase 5 ckpt_4 with C2 off (fixed τ = 0.5) and on (adapter epoch 4) on all three IOTD sequences; seq_01 and seq_03 are IOTD training sequences.

| Sequence | Light | C2 | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | mean τ |
|---|---|---|---|---|---|---|---|---|---|
| seq_01_1min | night | off | 28.87 | 16.69 | 50.02 | 3.30 | 34.66 | 0 | 0.50 |
| seq_01_1min | night | on | 27.82 | 14.60 | 53.17 | -41.33 | 29.57 | 139 | 0.31 |
| seq_01_1min | night | Δ on − off | -1.05 | -2.09 | 3.15 | -44.63 | -5.09 | 139 | – |
| seq_02_30sec_a | night | off | 57.74 | 49.87 | 67.30 | 51.29 | 75.18 | 0 | 0.50 |
| seq_02_30sec_a | night | on | 56.42 | 49.17 | 65.34 | 46.37 | 73.83 | 7 | 0.33 |
| seq_02_30sec_a | night | Δ on − off | -1.32 | -0.70 | -1.96 | -4.93 | -1.34 | 7 | – |
| seq_03_30sec_b | dusk | off | 26.25 | 18.39 | 37.54 | 23.39 | 37.54 | 2 | 0.50 |
| seq_03_30sec_b | dusk | on | 27.04 | 19.24 | 38.08 | 24.64 | 39.19 | 2 | 0.40 |
| seq_03_30sec_b | dusk | Δ on − off | 0.79 | 0.86 | 0.55 | 1.25 | 1.65 | 0 | – |

*Val: seq_02_30sec_a (744 frames, 50 tracks, night). Single run per setting, seed 42 unless stated. Corrected protocol: upstream output filter score > 0.5, area > 100 px.*
