"""Rule-based geometric driving decision logic.

Translates lane geometry (lateral offset + road curvature) into a discrete
driving command without any learned components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.lane_detection.lane_geometry import LaneGeometry

# ---------------------------------------------------------------------------
# Decision string constants
# ---------------------------------------------------------------------------

FORWARD = "FORWARD"
LEFT = "LEFT"
RIGHT = "RIGHT"
STOP = "STOP"


def make_geometric_decision(geometry: "LaneGeometry", config: dict) -> str:
    """Determine a driving command from lane geometry.

    Decision hierarchy (evaluated in priority order):

    1. **Curvature gate** — if |curvature| ≥ *curve_threshold*, the road is
       curving; issue a turn command in the direction of the curve.
    2. **Offset gate** — if |offset| ≥ *offset_threshold*, the vehicle has
       drifted; issue a correction command opposite to the drift direction.
    3. **Default** — return ``FORWARD``.

    Sign conventions:
        * ``offset > 0``    → vehicle drifted **right** → correct with LEFT.
        * ``offset < 0``    → vehicle drifted **left**  → correct with RIGHT.
        * ``curvature > 0`` → right curve ahead          → steer RIGHT.
        * ``curvature < 0`` → left curve ahead           → steer LEFT.

    Args:
        geometry: Lane geometry for the current frame.
        config: Dictionary matching the ``decision_engine`` section of the
            YAML config.  Reads ``offset_threshold`` (default 0.3) and
            ``curve_threshold`` (default 0.005).

    Returns:
        One of ``"FORWARD"``, ``"LEFT"``, ``"RIGHT"``, or ``"STOP"``.
    """
    offset_threshold = float(config.get("offset_threshold", 0.3))
    curve_threshold = float(config.get("curve_threshold", 0.005))

    # Priority 1: road curvature
    if abs(geometry.curvature) >= curve_threshold:
        return LEFT if geometry.curvature < 0 else RIGHT

    # Priority 2: lateral drift correction
    if geometry.offset > offset_threshold:
        return LEFT   # drifted right → steer left to re-centre
    if geometry.offset < -offset_threshold:
        return RIGHT  # drifted left  → steer right to re-centre

    return FORWARD
