"""
modules/position_decay.py — PerspectiveZoneEncoder + PositionMLP (Contribution 1)

PURPOSE
-------
Replace MeMOTR's fixed memory decay rate λ = 0.01 with a per-object adaptive
rate λ_adaptive that depends on where in the frame the tracked vehicle is.

In an oblique fixed-camera view, vehicles near the top of frame are far from
the camera: their bounding boxes are small, features are weak, and occlusion is
frequent.  Vehicles near the bottom are close: large, well-lit, reliable.
These two regimes should NOT share the same memory decay constant.

ARCHITECTURE (Section 4, Contribution 1 of memotr_project_prompt.md)
----------------------------------------------------------------------

  PerspectiveZoneEncoder (NO learned parameters)
  -----------------------------------------------
  Divides the normalized vertical image coordinate y_norm ∈ [0, 1]
  (0 = top of frame, 1 = bottom) into three zones using fixed boundaries:
      FAR  zone : y_norm < boundary_far_mid    (top ~30% of frame)
      MID  zone : boundary_far_mid ≤ y_norm < boundary_mid_near  (middle ~40%)
      NEAR zone : y_norm ≥ boundary_mid_near   (bottom ~30%)

  Zone boundaries are PLACEHOLDER constants (top 30% / middle 40% / bottom 30%)
  until Phase 4 (Task 4.2) runs calibrate_perspective.py on real IOTD annotations
  and replaces them with regression-derived values.

  BOUNDARY PLACEHOLDERS (replace after Phase 4 calibration):
      BOUNDARY_FAR_MID  = 0.30   # normalized y where FAR transitions to MID
      BOUNDARY_MID_NEAR = 0.70   # normalized y where MID transitions to NEAR

  PositionMLP (~1000 parameters)
  --------------------------------
  Input  : 4-dim vector [zone_far, zone_mid, zone_near, norm_box_height]
               zone_*      : one-hot encoding of the zone (3 dims)
               norm_box_height : bounding-box height / frame height ∈ [0, 1]
  Hidden : Linear(4 → 32) → ReLU → Linear(32 → 16) → ReLU
  Output : Linear(16 → 1) → Sigmoid → scale to [lambda_min, lambda_max]

  Output range: [0.002, 0.05] — keeps λ_adaptive in a physically meaningful range.
      FAR  zone → low  λ (e.g. ~0.003): rely on memory, current observation is noisy
      MID  zone → mid  λ (e.g. ~0.010): standard MeMOTR behavior
      NEAR zone → high λ (e.g. ~0.040): trust current observation, features strong

MEMORY UPDATE EQUATION (Section 4, Contribution 1)
----------------------------------------------------
  Original (fixed): M̃(t+1) = (1 - 0.01) × M(t) + 0.01 × O(t)
  Proposed:         M̃(t+1) = (1 - λ_adaptive) × M(t) + λ_adaptive × O(t)
  where λ_adaptive = PositionMLP(zone_encoding, norm_box_height)

TRAINING STRATEGY (Section 4)
------------------------------
  Phase 6 / train_full.py:
  - Load fine-tuned MeMOTR checkpoint (Phase 5 output)
  - Freeze backbone and encoder
  - Train TIM + PositionMLP on combined dataset, 15–20 epochs
  - LR: 2e-4, decay at epoch 12

WIRING INTO memorizing_motr.py (Task 3.4 / Phase 3)
------------------------------------------------------
  Inserted behind config flag `use_c1` (default False).
  When use_c1=True, inside QueryUpdater.forward():
      lambda_val = position_mlp(zone_encoder(cy_norm, h_norm))
      # replaces: lambda_val = 0.01   # TODO(C1)

NOVELTY CLAIM vs. STFN-ADM-MTHS (do NOT weaken)
-------------------------------------------------
  STFN-ADM-MTHS conditions decay on trajectory motion state.
  Ours conditions decay on geometric perspective position in the oblique
  camera's field of view — an orthogonal and previously unaddressed signal.
"""

import torch
import torch.nn as nn


# ── Zone boundary placeholders (replace after Phase 4 calibration) ─────────────
BOUNDARY_FAR_MID  = 0.30   # normalized y: top of frame = 0.0, bottom = 1.0
BOUNDARY_MID_NEAR = 0.70

# ── λ output range ────────────────────────────────────────────────────────────
LAMBDA_MIN = 0.002
LAMBDA_MAX = 0.050


class PerspectiveZoneEncoder(nn.Module):
    """
    Deterministic zone encoder — no learned parameters.

    Maps each tracked object's vertical center (normalized) and height
    (normalized) to a 4-dim input vector for PositionMLP.

    The zone assignment is a hard threshold:
        FAR  zone (top ~30%):    y_norm < boundary_far_mid
        MID  zone (middle ~40%): boundary_far_mid <= y_norm < boundary_mid_near
        NEAR zone (bottom ~30%): y_norm >= boundary_mid_near

    Args
    ----
    boundary_far_mid  : float  Normalized y boundary between FAR and MID zones.
    boundary_mid_near : float  Normalized y boundary between MID and NEAR zones.
    """

    def __init__(
        self,
        boundary_far_mid: float  = BOUNDARY_FAR_MID,
        boundary_mid_near: float = BOUNDARY_MID_NEAR,
    ):
        super().__init__()
        assert 0.0 < boundary_far_mid < boundary_mid_near < 1.0, \
            "Zone boundaries must satisfy 0 < far_mid < mid_near < 1"
        self.boundary_far_mid  = boundary_far_mid
        self.boundary_mid_near = boundary_mid_near

    def forward(self, cy_norm: torch.Tensor, h_norm: torch.Tensor) -> torch.Tensor:
        """
        Encode each object's position as a 4-dim zone vector.

        Parameters
        ----------
        cy_norm : torch.Tensor  shape (N,)
            Normalized vertical center of each bounding box, in [0, 1].
            0 = top of frame (FAR), 1 = bottom (NEAR).
        h_norm : torch.Tensor  shape (N,)
            Normalized height of each bounding box (box_height / frame_height).
            Also in [0, 1].

        Returns
        -------
        encoding : torch.Tensor  shape (N, 4)
            Columns: [zone_far (0/1), zone_mid (0/1), zone_near (0/1), h_norm]
            Exactly one of the three zone columns is 1 per row.
        """
        # One-hot zone assignment (float, for MLP input)
        # zone_far  : 1 where cy_norm is in [0, boundary_far_mid)
        # zone_mid  : 1 where cy_norm is in [boundary_far_mid, boundary_mid_near)
        # zone_near : 1 where cy_norm is in [boundary_mid_near, 1]
        zone_far  = (cy_norm <  self.boundary_far_mid).float()
        zone_near = (cy_norm >= self.boundary_mid_near).float()
        zone_mid  = 1.0 - zone_far - zone_near           # exactly 1 or 0

        # Stack to (N, 4): [zone_far, zone_mid, zone_near, h_norm]
        encoding = torch.stack([zone_far, zone_mid, zone_near, h_norm], dim=1)
        return encoding


class PositionMLP(nn.Module):
    """
    Tiny 3-layer MLP (~1000 params) mapping zone encoding to λ_adaptive.

    Architecture:
        Linear(4 → 32) → ReLU → Linear(32 → 16) → ReLU → Linear(16 → 1)
        → Sigmoid → scale to [lambda_min, lambda_max]

    Exact parameter count:
        (4×32 + 32) + (32×16 + 16) + (16×1 + 1) = 160 + 528 + 17 = 705 params

    The sigmoid-scaling output guarantees the output is always strictly inside
    (lambda_min, lambda_max) regardless of weight values.

    Args
    ----
    lambda_min : float  Lower bound of the λ output range.  Default 0.002.
    lambda_max : float  Upper bound of the λ output range.  Default 0.050.
    """

    def __init__(
        self,
        lambda_min: float = LAMBDA_MIN,
        lambda_max: float = LAMBDA_MAX,
    ):
        super().__init__()
        assert lambda_min < lambda_max, "lambda_min must be strictly less than lambda_max"
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # 4 → 32 → 16 → 1
        self._layers = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, zone_encoding: torch.Tensor) -> torch.Tensor:
        """
        Map zone encoding to an adaptive memory decay rate.

        Parameters
        ----------
        zone_encoding : torch.Tensor  shape (N, 4) or (4,)
            Output of PerspectiveZoneEncoder.forward().
            4 columns: [zone_far, zone_mid, zone_near, h_norm]

        Returns
        -------
        lambda_adaptive : torch.Tensor  shape (N,) or scalar
            Adaptive memory decay rate, always in (lambda_min, lambda_max).
            - FAR zone input → tends toward lambda_min (small decay; keep memory)
            - NEAR zone input → tends toward lambda_max (large decay; trust obs)
            (The direction of this tendency is learned, not hardcoded.)
        """
        unbatched = zone_encoding.ndim == 1
        if unbatched:
            zone_encoding = zone_encoding.unsqueeze(0)  # (1, 4)

        raw = self._layers(zone_encoding)               # (N, 1)
        # Scale from (0, 1) to (lambda_min, lambda_max)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(raw)
        lam = lam.squeeze(-1)                           # (N,)

        if unbatched:
            lam = lam.squeeze(0)                        # scalar

        return lam

    def count_parameters(self) -> int:
        """Return the number of trainable parameters (should be ~705)."""
        return sum(p.numel() for p in self._layers.parameters() if p.requires_grad)
