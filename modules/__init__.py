# modules/__init__.py
#
# Exposes the three novel modules for Contributions 1 & 2.
# Imports are guarded — each module raises NotImplementedError until implemented
# in Phase 3.  This file is safe to import at any stage.

# Contribution 1 — Perspective-Zone-Conditioned Memory Decay
from modules.position_decay import PerspectiveZoneEncoder, PositionMLP  # noqa: F401

# Contribution 2 — Scene-Conditioned Adaptive Detection Confidence
from modules.scene_analyzer import SceneAnalyzer          # noqa: F401
from modules.threshold_adapter import ThresholdAdapter    # noqa: F401
