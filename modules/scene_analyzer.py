"""
modules/scene_analyzer.py — SceneAnalyzer (Contribution 2, Phase 3 implementation)

PURPOSE
-------
Compute three lightweight scene-level scalars from a single video frame tensor.
These scalars are fed to ThresholdAdapter (threshold_adapter.py) to produce
tau_adaptive — the per-frame detection confidence threshold that replaces
MeMOTR's fixed tau_det = 0.5.

No learned parameters — this module is purely analytic / statistical.

INPUTS
------
  frame : torch.Tensor  shape (3, H, W) or (1, H, W) or (H, W, 3), float32 in [0, 1]

OUTPUTS
-------
  scene_vec : torch.Tensor  shape (3,)
      [0] brightness  — mean pixel intensity across all channels, in [0, 1]
      [1] contrast    — std of pixel intensity across all channels, in [0, 1]
      [2] density     — normalized active-track count passed in at call time,
                        clamped to [0, 1]; caller must supply this value

DESIGN NOTES (Section 4, Contribution 2 of memotr_project_prompt.md)
----------------------------------------------------------------------
The three scalars are:
  1. Brightness : mean pixel  →  low means night / dusk scene
  2. Contrast   : pixel std   →  low means foggy / flat-lit scene
  3. Density    : track count normalized by a maximum expected count (e.g. 50
                  vehicles is a dense Indian intersection).  Density indirectly
                  captures occlusion pressure on the tracker.

The ThresholdAdapter MLP maps this 3-vector to tau_adaptive in [0.3, 0.7].
In dark scenes (low brightness) the adapter is expected to output a lower tau
so that dim vehicles that score 0.35–0.45 are not discarded.

RELATIONSHIP TO memorizing_motr.py
-----------------------------------
In the inference forward pass of MeMOTR, the fixed line:
    if score > 0.5:   # TODO(C2)
is replaced with:
    if score > tau_adaptive:
where tau_adaptive comes from ThresholdAdapter(SceneAnalyzer(frame, track_count)).

USAGE (once implemented in Phase 3)
-------------------------------------
    analyzer = SceneAnalyzer(max_tracks=50)
    scene_vec = analyzer(frame_tensor, active_track_count=12)
    # → tensor([0.31, 0.08, 0.24])  # e.g. dark scene, low contrast, sparse traffic
"""

import torch
import torch.nn as nn


class SceneAnalyzer(nn.Module):
    """
    Analytic (no learned parameters) scene feature extractor.

    Args
    ----
    max_tracks : int
        Maximum expected number of simultaneously active tracks used to
        normalize the density scalar.  Default 50 (typical dense intersection).
    """

    def __init__(self, max_tracks: int = 50):
        super().__init__()
        self.max_tracks = max_tracks

    def forward(self, frame: torch.Tensor, active_track_count: int) -> torch.Tensor:
        """
        Parameters
        ----------
        frame : torch.Tensor
            Shape (3, H, W) or (H, W, 3) or (1, H, W), values in [0, 1].
            If not already float, it will be cast automatically.
        active_track_count : int
            Number of tracks currently maintained by the tracker at this frame.

        Returns
        -------
        scene_vec : torch.Tensor  shape (3,)
        """
        raise NotImplementedError(
            "SceneAnalyzer.forward() is not yet implemented. "
            "Implement in Phase 3 (Task 3.1) of the project execution plan."
        )
