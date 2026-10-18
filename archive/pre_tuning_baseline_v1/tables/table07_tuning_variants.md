| Run | Best ckpt | HOTA | DetA | AssA | MOTA | IDF1 | Rule passed |
|---|---|---|---|---|---|---|---|
| Phase 5 | ckpt_4 | 50.51 | 43.88 | **58.49** | 29.10 | 68.84 | reference |
| Arm 1 (seed mean, n=3) | 7,584 steps | 49.40 | 46.41 | 52.84 | 32.65 | 70.48 | reference |
| assoc_lr0 | ckpt_0 | 50.45 | 47.49 | 53.81 | 28.89 | 70.41 | No (above bar at 1 of 5 checkpoints) |
| lr_half | ckpt_0 | **51.19** | **48.68** | 54.16 | **36.56** | **70.53** | No (above bar at 1 of 5 checkpoints) |

Rule (Stage 1): HOTA > Arm 1 seed mean + 2 SD at > 1 checkpoint without worse AssA or IDSW. Arm 1 row: checkpoint with the highest seed-mean HOTA.
