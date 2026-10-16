# Review talking points (short answers)

**What is the project?** MeMOTR, a transformer multi-object tracker, adapted to track vehicles in Indian road videos
(IOTD). The main score is HOTA, which rewards both finding vehicles (DetA) and keeping their IDs (AssA).

**What data?** Three annotated IOTD videos, 3,274 real frames. Two are for training (2,530 frames), one night video
(`seq_02_30sec_a`, 744 frames) is for validation. Augmented copies change brightness, colour, compression and noise; they
are extra training samples, not extra annotated data.

**What did you train?** From a vehicle-trained baseline (Phase 5): Arm 1 on the real frames, Arm 2 on real + augmented
frames, three seeds of Arm 1, and two learning-rate variants (`assoc_lr0`, `lr_half`). Five checkpoints each.

**What is the final result?** Arm 1 checkpoint_3: HOTA 64.12 vs 57.74 for the baseline on the validation video, with
detection 0.5, tracking 0.5, miss tolerance 30. All three seeds of Arm 1 beat the baseline (59.74–64.12).

**Why did the numbers change so much late in the project?** The original evaluation script wrote out every live track,
even ones the model had lost (confidence as low as 0.003). Upstream MeMOTR filters those out (score > 0.5, box > 100 px).
After matching upstream, every model scored 7–14 HOTA higher. This is a bug fix in our evaluation, not a model
improvement, and we re-scored everything with it before drawing conclusions.

**How did you know the fix was correct?** With the filter switched off the new script reproduced the old output byte for
byte; with it on, it kept a subset of exactly the same boxes. The threshold value comes from the project config, not
from tuning.

**Why not miss tolerance 20 (S_D)?** It looked +4 HOTA better before the fix, but only because it accidentally removed
some of those junk boxes. After the fix it is −0.07 HOTA on average over three seeds — no difference.

**S_F had the highest score (64.25). Why not choose it?** One run, +0.12 over S_A, more ID switches (6 vs 3). Our rule
needs a gain beyond two seed standard deviations (bar 66.14). We don't pick settings from one lucky number.

**Did augmentation help?** No. Arm 2's best was 61.61 vs Arm 1's 64.12, and later Arm 2 checkpoints produce many false
positives. This is one run each, so we say "no evidence it helps", not "it hurts".

**What about the learning-rate variants?** Negative results. `lr_half` checkpoint_0 is +0.14 over Arm 1 (inside the noise)
and both variants collapse one epoch later (HOTA ~50, 21 ID switches). Freezing the association module did not help.

**Why seeds?** The same training with a different random seed gave HOTA 59.74–64.12 — a 4.4-point spread. Any
difference smaller than that cannot be trusted from one run.

**What happened to C2?** C2 learns a detection threshold from scene brightness. It does lower the threshold at night
(about 0.31–0.33) and raise it at dusk (0.40), but at night that adds false alarms and ID switches: HOTA −1.05 and −1.32 on
the two night videos, +0.79 on dusk (within noise). It stays off until it is calibrated for this camera.

**And C1?** Implemented and unit-tested, but its zone boundaries are placeholders and it has not been trained or
evaluated, so we make no claim.

**Main limitations?** One night validation video, few seeds, choices made on the validation video, no day-time test,
no statistical significance.

**What would you do next?** Calibrate C1 and C2 on IOTD, add a day-time validation sequence, and repeat the main
comparison with more seeds.
