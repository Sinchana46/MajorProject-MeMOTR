# Hyperparameter Tuning: Completion Status

Date: 17 Oct 2026

## What counts as "tuning done"

Tuning is considered complete when:
- A checkpoint is selected by the written rule and its metrics are stable across seeds.
- No untried setting is likely to beat the noise bar given current data.
- The evaluation protocol is consistent and verified.
- Results are documented and reproducible.

## Status

| Item | Status | Notes |
|---|---|---|
| Baseline (Phase 5 ckpt_4) evaluated | DONE | Corrected HOTA 57.74 |
| Real-only fine-tuning (Arm 1) | DONE | Best: ckpt_3, HOTA 64.12 |
| Augmented fine-tuning (Arm 2) | DONE | Best: ckpt_1, HOTA 61.61; discarded |
| Seed repeats (n=3 noise estimate) | DONE | Mean 61.64 ± 2.25 at ckpt_3 (corrected seed values exist for ckpt_3 only); noise bar 66.14 |
| assoc_lr0 variant | DONE | Negative result (best 62.76, ckpt_0) |
| lr_half variant | DONE | Negative result (best 64.26, ckpt_0; single run, unstable after epoch 1) |
| Inference threshold sweep | DONE | S_A retained (no setting above 66.14) |
| Output filter bug found and fixed | DONE | All results re-scored (21 checkpoints, 7 settings, 6 C2 runs, 4 seed runs) |
| C2 evaluation | DONE | No gain (HOTA −1.05 / −1.32 / +0.79); future work |
| C1 evaluation | NOT DONE | Deferred: calibration not run, PositionMLP not trained |
| Best checkpoint selected | DONE | Arm 1 ckpt_3 |
| Protocol verified reproducible | PARTIAL | Filtered runner with filters off reproduces the original predictions byte for byte; unfiltered S_A reproduced the registry (HOTA 50.47) exactly. The corrected 64.12 itself comes from one run and was not re-run. |
| Pre-tuning archive frozen | DONE | archive/pre_tuning_baseline_v1/ (unchanged, verified) |
| Corrected results tables and figures | DONE (this prompt) | results/tuning_closeout/ |
| Tuning narrative written | DONE (this prompt) | tuning_narrative.md |

## Definition of done: PASS (with stated caveats)

The training hyperparameter search is complete. The chosen model is
Arm 1 checkpoint_3 with the corrected inference tool. C1 and C2 are
deferred to a separate evaluation phase with redesigned supervision.

Caveats against the four criteria above:
- "Stable across seeds": the corrected seed SD at ckpt_3 is 2.25 HOTA (range 4.38), measured at one checkpoint with
  n = 3. All three seeds are above Phase 5, but the estimate is rough.
- "No untried setting likely to beat the noise bar": an assumption based on the settings tried; not proven.
- Reproducibility of the corrected value: see the PARTIAL row above.

## Next steps (outside tuning scope)

1. Revise C2 supervision target and retrain adapter on IOTD data.
2. Calibrate C1 perspective zones using seq_01 and seq_03 original frames.
3. Annotate more IOTD sequences (target: day and dusk lighting, 500+ total tracks).
4. Run final evaluation on the held-out test set once it exists.
