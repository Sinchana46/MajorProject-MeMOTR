# Implementation Instructions for Google Antigravity
## Project: Perspective-Aware and Scene-Adaptive MeMOTR for Oblique Fixed-Camera Indian Traffic Surveillance

> **How to use this file:** Paste one phase at a time into Antigravity as a task description. Do not paste the whole file in one go — Antigravity plans and executes at the task level, and each phase below is scoped to be one coherent task/implementation-plan cycle. Wait for the Walkthrough artifact at the end of a phase, verify it against the "Verification" checklist for that phase, and only then proceed to the next. Full project context (architecture, contributions, novelty claims) is in `memotr_project_prompt.md` — attach it to the project so the agent has it in context throughout.

---

## PHASE 1 — Environment & Codebase Foundation (NO DATASET USAGE)

**Why this phase has no dataset dependency:** IOTD annotation is still in progress. Everything here is infrastructure — verifiable using only the code repositories and their own bundled sample/demo assets, never IOTD, UA-DETRAC, or VisDrone.

### Task 1.1 — Environment setup
- Create conda environment `memotr_project`, Python 3.9.
- Install PyTorch 2.0.1 / torchvision 0.15.2 matched to the local CUDA version (verify CUDA version first with `nvidia-smi` before picking the index URL — do not assume cu118).
- Verify GPU visibility: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- **Verification artifact:** terminal output confirming GPU name and 24GB VRAM detected.

### Task 1.2 — Clone and install repositories
- Clone `MCG-NJU/MeMOTR`, install `requirements.txt`.
- Clone `JonathonLuiten/TrackEval`, install its requirements.
- Install `wandb` for experiment tracking (do not configure a project yet — that happens in Phase 3 when there's something to log).
- **Verification artifact:** clean `pip list` showing all packages installed without conflicts; no dataset download attempted.

### Task 1.3 — Sanity-check MeMOTR runs at all
- Locate MeMOTR's own bundled demo/sample inference assets (check `MeMOTR/README.md` and any `demo/` or `assets/` folder in the repo — do NOT download UA-DETRAC or VisDrone for this).
- Run the pretrained MeMOTR checkpoint (download only the official pretrained `.pth`, not a dataset) on whatever sample images/video ship with the repo or a single publicly-available stock traffic clip used only for a smoke test.
- **Verification artifact:** Walkthrough screenshot showing MeMOTR producing bounding boxes on the sample input, confirming the install is functional end-to-end.

### Task 1.4 — Locate and annotate (in comments only) the two hook points
- Open `models/memorizing_motr.py`.
- Find the `TemporalInteractionModule` class and the line(s) implementing `λ = 0.01` in the memory update. Add a `# TODO(C1)` comment marking it — do not change behavior yet.
- Find where `tau_det = 0.5` (or equivalent fixed confidence threshold) is applied in the inference forward pass. Add a `# TODO(C2)` comment marking it — do not change behavior yet.
- **Verification artifact:** diff showing only comments added, zero functional change, confirmed by running Task 1.3's smoke test again with identical output.

### Task 1.5 — Repo scaffolding
- Create the full directory structure from Section 11 of the project doc (`data/`, `checkpoints/`, `results/`, `modules/`, `tools/`) with `.gitkeep` placeholders — no data goes in `data/` yet.
- Create empty stub files with docstrings only (no logic) for: `modules/scene_analyzer.py`, `modules/threshold_adapter.py`, `modules/position_decay.py`, `tools/calibrate_perspective.py`, `tools/convert_detrac.py`, `tools/convert_visdrone.py`, `tools/visualize_zones.py`.
- **Verification artifact:** directory tree matching Section 11 exactly; each stub file has a docstring describing its future role (copy from Sections 4 and 10 of the project doc) but raises `NotImplementedError` if called.

**Phase 1 exit criteria:** environment reproducible from scratch on a clean machine via a single setup script; MeMOTR verified functional on non-project data; both modification hook points located and marked; full folder skeleton exists. No IOTD, UA-DETRAC, or VisDrone data has touched disk.

---

## PHASE 2 — Public Dataset Pipeline (UA-DETRAC + VisDrone only — IOTD still excluded)

These two datasets are publicly available now and don't depend on annotation progress, so this phase can run while IOTD annotation continues in parallel.

### Task 2.1 — UA-DETRAC ingestion
- Download UA-DETRAC per the official 60/40 train/test split.
- Implement `tools/convert_detrac.py`: parse the per-sequence XML ground truth, emit MOT Challenge format (`gt.txt` per sequence + `seqinfo.ini`), matching the directory layout in Section 11.
- **Verification artifact:** script run log showing N sequences converted, frame counts matching the source, and one rendered sample frame with boxes overlaid from the converted `gt.txt` (visual proof the conversion is correct, not just that it ran).

### Task 2.2 — VisDrone-MOT ingestion
- Download the **MOT version** of VisDrone (explicitly confirm this is not the DET version before proceeding — file structure differs).
- Implement `tools/convert_visdrone.py`: filter to vehicle classes only (car=4, van=5, truck=6, bus=9), discard all other classes, emit MOT format.
- **Verification artifact:** class distribution printout pre- and post-filtering, plus one rendered sample frame confirming only vehicle classes remain.

### Task 2.3 — Cross-dataset schema validation
- Write a validation script (can live in `tools/`) that loads all converted sequences from both datasets and asserts: consistent `gt.txt` column schema, no negative coordinates, no zero-area boxes, sequence-level (not frame-level) split integrity.
- **Verification artifact:** validation report with zero schema violations across both datasets.

**Phase 2 exit criteria:** both public datasets fully converted to MOT format, validated, and organized under `data/UA_DETRAC/` and `data/VisDrone_Vehicles/` per Section 11. IOTD folder remains empty.

---

## PHASE 3 — Module Scaffolding with Synthetic Data (still no real dataset training)

This phase implements the actual logic for Contributions 1 and 2 as isolated, unit-testable modules using synthetic tensors — so implementation and testing can proceed before real training data (especially IOTD) is ready.

### Task 3.1 — Implement `modules/scene_analyzer.py`
- `SceneAnalyzer` class: no learned parameters. Input: a frame tensor. Output: 3 scalars (mean pixel brightness, pixel std as contrast, normalized active-track-count density).
- **Verification artifact:** unit test with a synthetic bright frame, dark frame, and mid-range frame confirming brightness/contrast values move in the expected direction.

### Task 3.2 — Implement `modules/threshold_adapter.py`
- `ThresholdAdapter`: 2-layer MLP (~500 params), maps the 3-scalar scene vector to τ_adaptive, clamped to [0.3, 0.7].
- **Verification artifact:** unit test confirming output is always within bounds across random input ranges, and that the module is untrained/randomly initialized at this stage (training happens in Phase 5).

### Task 3.3 — Implement `modules/position_decay.py`
- `PerspectiveZoneEncoder`: fixed, non-learned zone boundary logic (top 30% / mid 40% / bottom 30%, per Section 4, Contribution 1 — boundaries are placeholders until Phase 4's calibration script produces real ones).
- `PositionMLP`: tiny 3-layer network (~1000 params) mapping (zone, normalized box height) → λ_adaptive.
- **Verification artifact:** unit test confirming a synthetic top-of-frame box yields lower λ than a synthetic bottom-of-frame box, and output stays in a sane [0,1] range.

### Task 3.4 — Wire modules into MeMOTR without activating them
- In `models/memorizing_motr.py`, import the three new modules and add them to `__init__`, but gate their use behind a config flag (e.g., `use_c1`, `use_c2`) defaulting to `False`, so existing fixed-λ / fixed-τ behavior is unchanged by default.
- **Verification artifact:** re-run the Phase 1 smoke test with flags off — output must be bit-identical to Phase 1 baseline, proving zero regression. Then re-run with flags on against synthetic input only, confirming no crashes.

**Phase 3 exit criteria:** all three new modules exist, are unit-tested against synthetic inputs, and are wired into MeMOTR behind feature flags with proven zero-regression when disabled. No real training has occurred.

---

## PHASE 4 — IOTD Integration (gate: only start once first annotated sequences exist)

> **Do not start this phase until you confirm with the user that at least the first 3 IOTD sequences are annotated and exported from CVAT in MOT format.** This is the first phase that touches IOTD.

### Task 4.1 — IOTD ingestion and validation
- Import the first annotated sequences into `data/IOTD/` per the train/val/test split (12/4/4, by sequence, never by frame — enforce this in the loader, not just as a convention).
- Run the Task 2.3-style schema validator against IOTD, plus an occlusion-attribute completeness check (none/partial/heavy must be present on every box).
- **Verification artifact:** validation report; rendered sample frames confirming annotation import correctness.

### Task 4.2 — Perspective calibration
- Implement and run `tools/calibrate_perspective.py`: regress bounding-box height against normalized vertical position across annotated IOTD boxes to derive real zone boundaries and report R².
- Replace the Phase 3 placeholder zone boundaries in `position_decay.py` with the calibrated constants.
- **Verification artifact:** fills Section 15 of the project doc (slope, R², zone boundaries) and produces the zone-boundary visualization figure via `tools/visualize_zones.py`.

**Phase 4 exit criteria:** IOTD (as much as is annotated so far) is validated and integrated; perspective zones are calibrated from real data, not placeholders.

---

## PHASES 5–8 (outline only — detail once Phases 1–4 are complete and IOTD annotation is further along)

- **Phase 5 — Baseline establishment:** fine-tune MeMOTR (flags off) on combined UA-DETRAC + VisDrone + IOTD-so-far; evaluate with TrackEval to produce the "MeMOTR fine-tuned" row of Section 12's ablation table.
- **Phase 6 — Contribution training:** `train_c2_only.py` (frozen MeMOTR, train ThresholdAdapter, 5 epochs) → verify → `train_full.py` (TIM + PositionMLP + ThresholdAdapter, 15–20 epochs, per Section 4's LR schedule). Implement C2 first, verify independently, only then add C1 — per Section 19's explicit instruction not to implement both simultaneously.
- **Phase 7 — Full ablation + per-condition evaluation:** populate Sections 12–14's tables (`eval_per_condition.py`, `eval_per_zone.py`), plus baseline comparisons against ByteTrack, OC-SORT, and STFN-ADM-MTHS.
- **Phase 8 — Paper-support artifacts:** generate final figures, tables, and a reproducibility checklist for submission.

Each of these should be expanded into its own detailed phase (matching the granularity of Phases 1–4 above) once you're ready to start it — expanding all 8 phases up front isn't useful, since Phases 5–8 depend on decisions (e.g., final IOTD sequence count, calibration results) that don't exist yet.

---

## Standing rules for every phase (apply throughout, not just once)

- Never split any dataset by frame — sequence-level splits only, enforced in code, not just convention.
- Never let Antigravity touch IOTD test sequences until Phase 7's final evaluation.
- Never modify the MeMOTR backbone or encoder — out of scope for both contributions.
- Every module Antigravity generates must be small enough and clearly-commented enough for you to explain it line-by-line to your guide — if a Walkthrough's implementation is too dense to explain, ask Antigravity to simplify before accepting it, per Section 19.
- Reject any Walkthrough that silently changes default behavior (fixed λ or fixed τ) without an explicit feature flag.
