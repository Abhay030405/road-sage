"""Safety gate — hard-overrides the geometric decision when a hazard is present.

The gate is intentionally stateless and conservative: a single trigger in
any one frame is sufficient to force a STOP.  Temporal smoothing and
debouncing must be applied *downstream* by the caller, never inside this
module, so that STOP commands are always immediate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneContext:
    """Aggregated scene state consumed by the safety gate each frame.

    Attributes:
        immediate_hazard: Hard-stop trigger — e.g. a pedestrian has stepped
            into the vehicle's path or another critical safety event fired.
        obstacle_detected: Whether any blocking obstacle was found by the
            object detector within the drivable corridor.
        obstacle_distance: Distance to the nearest obstacle in metres.
            Use ``float('inf')`` when no obstacle is detected.
        confidence: Overall system confidence in [0, 1], typically the
            minimum of lane-detection and object-detector confidences.
    """

    immediate_hazard: bool = False
    obstacle_detected: bool = False
    obstacle_distance: float = float("inf")
    confidence: float = 1.0


def should_stop(scene: SceneContext, config: dict) -> bool:
    """Evaluate whether the safety gate must override with a STOP command.

    Three independent triggers are checked (any one is sufficient to stop):

    1. ``scene.immediate_hazard`` is ``True``.
    2. ``scene.confidence < min_confidence`` (system below acceptable threshold).
    3. An obstacle is detected within *obstacle_stop_distance* metres.

    This function is **stateless** — it decides purely from the current frame.
    Do not add smoothing or frame-history here; any filtering belongs in the
    caller *after* the safety check.

    Args:
        scene: Current scene context.
        config: Dictionary matching the ``decision_engine`` section of the
            YAML config.  Reads ``min_confidence`` (default 0.60) and
            ``obstacle_stop_distance`` (default 2.0).

    Returns:
        ``True`` if a STOP must be issued immediately, ``False`` otherwise.
    """
    min_confidence = float(config.get("min_confidence", 0.60))
    stop_distance = float(config.get("obstacle_stop_distance", 2.0))

    if scene.immediate_hazard:
        return True
    if scene.confidence < min_confidence:
        return True
    if scene.obstacle_detected and scene.obstacle_distance < stop_distance:
        return True
    return False
