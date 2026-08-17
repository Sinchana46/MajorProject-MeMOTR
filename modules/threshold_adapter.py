"""
modules/threshold_adapter.py — ThresholdAdapter (Contribution 2, Phase 3 implementation)

PURPOSE
-------
Learn a mapping from the 3-scalar scene vector produced by SceneAnalyzer
to tau_adaptive — a per-frame detection confidence threshold in [0.3, 0.7].

This replaces MeMOTR's hardcoded tau_det = 0.5 with a scene-conditioned value,
enabling the tracker to detect genuine vehicles in dark/dusk frames that score
0.35–0.45 and would otherwise be discarded.

ARCHITECTURE (Section 4, Contribution 2 of memotr_project_prompt.md)
----------------------------------------------------------------------
  Input  : 3-dim scene vector  [brightness, contrast, density]
  Hidden : Linear(3 → 16) → ReLU → Linear(16 → 8) → ReLU
  Output : Linear(8 → 1) → Sigmoid → scale to [0.3, 0.7]

  Total parameters ≈ 500  (intentionally tiny — explainable to guide)

TRAINING STRATEGY (Section 4 of memotr_project_prompt.md)
----------------------------------------------------------
  - Phase 5 / train_c2_only.py
  - ALL MeMOTR weights frozen — only this module trains
  - Dataset: IOTD training sequences, all 3 lighting conditions
  - LR: 1e-3, 5 epochs, fast convergence expected
  - Loss: tracking-quality-driven (HOTA proxy or ID-switch penalty)

WIRING INTO memorizing_motr.py (Task 3.4 / Phase 3)
------------------------------------------------------
  Inserted behind a config flag `use_c2` (default False, zero-regression guarantee).
  When use_c2=True the inference forward pass becomes:
      scene_vec   = scene_analyzer(frame, active_track_count)
      tau_adaptive = threshold_adapter(scene_vec)
      if score > tau_adaptive:   # replaces: if score > 0.5  # TODO(C2)
          ...

NOVELTY CLAIM (do NOT weaken)
------------------------------
  No published transformer-MOT paper adapts the detection threshold
  at inference time using a learned scene-state vector.
  ByteTrack uses low-confidence detections post-detection (different paradigm).
  Ours operates pre-decision, on the same detection head output.
"""

import torch
import torch.nn as nn


class ThresholdAdapter(nn.Module):
    """
    2-layer MLP mapping a 3-dim scene vector to tau_adaptive in [0.3, 0.7].

    Args
    ----
    tau_min : float  Lower bound of the output range.  Default 0.3.
    tau_max : float  Upper bound of the output range.  Default 0.7.
    """

    TAU_MIN_DEFAULT = 0.3
    TAU_MAX_DEFAULT = 0.7

    def __init__(self, tau_min: float = TAU_MIN_DEFAULT, tau_max: float = TAU_MAX_DEFAULT):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        # Architecture will be defined in Phase 3 (Task 3.2)
        # Placeholder — do not call forward() until implemented.
        self._mlp = None

    def forward(self, scene_vec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        scene_vec : torch.Tensor  shape (3,) or (B, 3)
            Output of SceneAnalyzer.forward().

        Returns
        -------
        tau_adaptive : torch.Tensor  scalar or shape (B,)
            Detection confidence threshold, always in [tau_min, tau_max].
        """
        raise NotImplementedError(
            "ThresholdAdapter.forward() is not yet implemented. "
            "Implement in Phase 3 (Task 3.2) of the project execution plan."
        )

    def count_parameters(self) -> int:
        """Utility: count trainable parameters (target ~500)."""
        if self._mlp is None:
            return 0
        return sum(p.numel() for p in self._mlp.parameters() if p.requires_grad)
