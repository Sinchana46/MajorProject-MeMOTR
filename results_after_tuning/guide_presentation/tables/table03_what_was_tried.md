| Configuration | Change from Arm 1 | Best corrected HOTA | Passed decision rule | Outcome |
|---|---|---|---|---|
| Phase 5 (baseline) | no IOTD data (start point) | 57.74 | reference | baseline |
| Arm 1 (real only) | — | 64.12 | reference (default) | CHOSEN (ckpt_3) |
| Arm 2 (augmented) | +7,590 augmented frames | 61.61 | NO | discarded: lower at every checkpoint |
| assoc_lr0 | query-updater LR = 0 | 62.76 | NO | discarded: below noise bar |
| lr_half | all LRs × 0.5 | 64.26 | NO | discarded: single run below noise bar, unstable |
| C2 adapter | scene-conditioned detection threshold (Phase 5 base) | 56.42 | NO | no gain: -1.05, -1.32, +0.79 HOTA on seq_01/02/03 |

*Decision rule: HOTA > seed mean + 2 SD (= 66.14) at more than one checkpoint and IDSW does not increase. Under the corrected protocol the seed band exists at ckpt_3 only, so each best value is compared with 66.14. C2 HOTA is Phase 5 ckpt_4 with C2 on, seq_02.*
