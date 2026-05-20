"""Lane detection and driving-decision visualizer for RoadSage.

Renders detection results, lane geometry metrics, driving commands, and an
optional BEV minimap onto BGR camera frames.  Every overlay is composited
non-destructively (the original image is never mutated) and is legible enough
for a live demo or a judge walkthrough.

Coordinate convention
---------------------
All images are OpenCV BGR arrays (H × W × 3, uint8).
All colors are expressed as ``(B, G, R)`` tuples to match OpenCV.

Usage::

    from app.explainability.visualizer import (
        VisualizationConfig, create_full_visualization,
    )

    cfg   = VisualizationConfig.from_yaml("configs/lane_detection.yaml")
    frame = create_full_visualization(
        original_image=bgr_frame,
        detection=result,
        geometry=geom,
        bev_image=bev_frame,
        command="FORWARD",
        confidence=0.92,
        decision_path="Geometric",
        config=cfg,
    )
    cv2.imwrite("outputs/demo.png", frame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from app.lane_detection.lane_geometry import LaneGeometry
from app.lane_detection.ufld_model import LaneDetectionResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Command → color map (BGR)
# ---------------------------------------------------------------------------

_COMMAND_COLORS: dict[str, Tuple[int, int, int]] = {
    "FORWARD": (0,   200,  0),    # green
    "LEFT":    (0,   165, 255),   # amber
    "RIGHT":   (255, 100,  0),    # blue-ish
    "STOP":    (0,    0,  220),   # red
}
_DEFAULT_COMMAND_COLOR: Tuple[int, int, int] = (180, 180, 180)  # grey fallback


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class VisualizationConfig:
    """Visual styling parameters for all RoadSage overlays.

    Colors are ``(B, G, R)`` tuples (OpenCV convention).

    Attributes:
        left_lane_color:   BGR color for a high-confidence left lane marking.
        right_lane_color:  BGR color for a high-confidence right lane marking.
        center_lane_color: BGR color for center / middle lane markings.
        low_conf_color:    BGR color used when lane confidence is between
            0.65 and 0.85.
        no_lane_color:     BGR color used when confidence is below 0.65.
        lane_thickness:    Polyline stroke width in pixels.
        fill_alpha:        Transparency of the lane-corridor fill in [0, 1].
            0 = fully transparent, 1 = fully opaque.
        font:              OpenCV font face constant.
        font_scale:        OpenCV font scale factor.
        font_thickness:    Stroke width of rendered text.
    """

    left_lane_color:   Tuple[int, int, int] = (0, 255,   0)    # green
    right_lane_color:  Tuple[int, int, int] = (0, 255,   0)    # green
    center_lane_color: Tuple[int, int, int] = (255, 255, 0)    # cyan
    low_conf_color:    Tuple[int, int, int] = (0, 165, 255)    # orange
    no_lane_color:     Tuple[int, int, int] = (0,   0, 255)    # red
    lane_thickness:    int   = 3
    fill_alpha:        float = 0.25
    font:              int   = cv2.FONT_HERSHEY_SIMPLEX
    font_scale:        float = 0.7
    font_thickness:    int   = 2

    @classmethod
    def from_yaml(cls, config_path: str = "configs/lane_detection.yaml") -> "VisualizationConfig":
        """Load visualization config from the ``visualization`` section of a YAML file.

        Falls back to defaults for any missing key so partial configs work.

        Args:
            config_path: Path to a YAML file that contains a top-level
                ``visualization`` key (e.g. ``configs/lane_detection.yaml``).

        Returns:
            A :class:`VisualizationConfig` populated from the YAML values.

        Raises:
            FileNotFoundError: When *config_path* does not exist on disk.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: '{config_path}'")
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        vis = raw.get("visualization", {})

        def _color(key: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
            val = vis.get(key)
            if val is not None:
                return tuple(int(v) for v in val)  # type: ignore[return-value]
            return default

        return cls(
            left_lane_color=_color("left_lane_color",   (0, 255,   0)),
            right_lane_color=_color("right_lane_color",  (0, 255,   0)),
            center_lane_color=_color("center_lane_color",(255, 255, 0)),
            low_conf_color=_color("low_conf_color",   (0, 165, 255)),
            no_lane_color=_color("no_lane_color",    (0,   0, 255)),
            lane_thickness=int(vis.get("lane_thickness", 3)),
            fill_alpha=float(vis.get("fill_alpha", 0.25)),
            font_scale=float(vis.get("font_scale", 0.7)),
            font_thickness=int(vis.get("font_thickness", 2)),
        )


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------


def get_lane_color(
    confidence: float,
    config: VisualizationConfig,
) -> Tuple[int, int, int]:
    """Return the BGR lane color appropriate for a given confidence score.

    Thresholds::

        > 0.85  → left_lane_color  (green)  — high confidence
        0.65–0.85 → low_conf_color (orange) — medium confidence
        < 0.65  → no_lane_color    (red)    — unreliable

    Args:
        confidence: Lane detection confidence in ``[0, 1]``.
        config:     Active :class:`VisualizationConfig`.

    Returns:
        A ``(B, G, R)`` color tuple.
    """
    if confidence > 0.85:
        return config.left_lane_color
    if confidence >= 0.65:
        return config.low_conf_color
    return config.no_lane_color


# ---------------------------------------------------------------------------
# Lane-line drawing
# ---------------------------------------------------------------------------


def draw_lane_lines(
    image: np.ndarray,
    detection: LaneDetectionResult,
    config: VisualizationConfig,
) -> np.ndarray:
    """Draw polylines and point markers for every detected lane boundary.

    Draws left, right, and (when present) center lane markings.  Colors are
    chosen per-lane based on confidence using :func:`get_lane_color`.

    Args:
        image:     BGR source frame.  Never mutated.
        detection: Raw lane detection result containing pixel coordinates and
            confidence scores.
        config:    Active :class:`VisualizationConfig`.

    Returns:
        A new BGR array with lane markings drawn on top.
    """
    out = image.copy()

    confs      = detection.confidence
    left_conf  = confs[0] if len(confs) > 0 else 0.0
    right_conf = confs[1] if len(confs) > 1 else 0.0

    lanes = [
        (detection.left_lane,  get_lane_color(left_conf,  config)),
        (detection.right_lane, get_lane_color(right_conf, config)),
    ]
    if detection.center_lane:
        lanes.append((detection.center_lane, config.center_lane_color))

    for pts, color in lanes:
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [arr], isClosed=False,
                      color=color, thickness=config.lane_thickness,
                      lineType=cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(out, (x, y), radius=3, color=color, thickness=-1,
                       lineType=cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Lane corridor fill
# ---------------------------------------------------------------------------


def draw_lane_corridor(
    image: np.ndarray,
    detection: LaneDetectionResult,
    geometry: LaneGeometry,
    config: VisualizationConfig,
) -> np.ndarray:
    """Fill the drivable corridor between left and right lane boundaries.

    If both lanes are detected a semi-transparent polygon is composited
    between them using ``cv2.addWeighted``.  Lane line markings are then
    drawn on top so they remain visible through the fill.

    Args:
        image:     BGR source frame.  Never mutated.
        detection: Lane detection result (provides point lists).
        geometry:  Computed geometry (provides lane-detected flags).
        config:    Active :class:`VisualizationConfig`.

    Returns:
        A new BGR array with corridor fill and lane lines drawn on top.
    """
    out = image.copy()

    if (geometry.left_lane_detected and geometry.right_lane_detected
            and len(detection.left_lane) >= 2
            and len(detection.right_lane) >= 2):

        confs = detection.confidence
        left_conf  = confs[0] if len(confs) > 0 else 0.0
        right_conf = confs[1] if len(confs) > 1 else 0.0
        fill_color = (
            config.left_lane_color
            if (left_conf > 0.85 and right_conf > 0.85)
            else config.low_conf_color
        )

        # Build corridor polygon: left boundary top→bottom, right boundary bottom→top
        left_pts  = detection.left_lane
        right_pts = detection.right_lane
        polygon   = np.array(left_pts + list(reversed(right_pts)), dtype=np.int32)

        overlay = out.copy()
        cv2.fillPoly(overlay, [polygon], color=fill_color)
        cv2.addWeighted(overlay, config.fill_alpha, out, 1.0 - config.fill_alpha, 0, out)

    return draw_lane_lines(out, detection, config)


# ---------------------------------------------------------------------------
# Decision command overlay
# ---------------------------------------------------------------------------


def draw_decision_overlay(
    image: np.ndarray,
    command: str,
    confidence: float,
    decision_path: str,
) -> np.ndarray:
    """Render a driving-command banner at the bottom of the image.

    Draws a semi-transparent dark rectangle that spans the full image width,
    then prints the command in a large color-coded font, the confidence as a
    percentage, and the decision path label in smaller text.

    Args:
        image:         BGR source frame.  Never mutated.
        command:       Driving command string — one of
                       ``"FORWARD"``, ``"LEFT"``, ``"RIGHT"``, ``"STOP"``.
        confidence:    Decision confidence in ``[0, 1]``.
        decision_path: Human-readable source of the decision, e.g.
                       ``"Geometric"``, ``"ML Fallback"``, or
                       ``"Safety Gate"``.

    Returns:
        A new BGR array with the decision banner composited at the bottom.
    """
    out = image.copy()
    H, W = out.shape[:2]
    banner_h = max(80, H // 6)
    y0 = H - banner_h

    # Semi-transparent dark background
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (W, H), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    color = _COMMAND_COLORS.get(command.upper(), _DEFAULT_COMMAND_COLOR)

    # Large command text centered horizontally
    cmd_scale  = min(W / 400, 2.4)
    cmd_thick  = max(2, int(cmd_scale * 2.5))
    (tw, th), baseline = cv2.getTextSize(
        command, cv2.FONT_HERSHEY_DUPLEX, cmd_scale, cmd_thick
    )
    cmd_x = (W - tw) // 2
    cmd_y = y0 + (banner_h + th) // 2 - baseline - 8
    cv2.putText(out, command, (cmd_x, cmd_y),
                cv2.FONT_HERSHEY_DUPLEX, cmd_scale, color, cmd_thick,
                cv2.LINE_AA)

    # Confidence percentage — below command
    conf_str = f"{confidence * 100:.1f}%"
    conf_scale = 0.55
    (cw, ch), _ = cv2.getTextSize(
        conf_str, cv2.FONT_HERSHEY_SIMPLEX, conf_scale, 1
    )
    cv2.putText(out, conf_str,
                ((W - cw) // 2, cmd_y + ch + 6),
                cv2.FONT_HERSHEY_SIMPLEX, conf_scale, (220, 220, 220), 1,
                cv2.LINE_AA)

    # Decision path — small text bottom-right
    path_str = decision_path
    path_scale = 0.45
    (pw, ph), _ = cv2.getTextSize(
        path_str, cv2.FONT_HERSHEY_SIMPLEX, path_scale, 1
    )
    cv2.putText(out, path_str,
                (W - pw - 10, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, path_scale, (160, 160, 160), 1,
                cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Geometry info panel
# ---------------------------------------------------------------------------


def draw_geometry_info(
    image: np.ndarray,
    geometry: LaneGeometry,
) -> np.ndarray:
    """Render a metric info panel in the top-left corner of the image.

    Displays lateral offset (with drift direction arrow), curvature, road
    width, and per-lane detection status.  A dark background rectangle is
    drawn behind the text for readability against any road colour.

    Args:
        image:    BGR source frame.  Never mutated.
        geometry: Computed lane geometry for the current frame.

    Returns:
        A new BGR array with the info panel composited in the top-left.
    """
    out = image.copy()

    offset_sign  = "→" if geometry.offset_m > 0 else ("←" if geometry.offset_m < 0 else "·")
    left_mark    = "L✓" if geometry.left_lane_detected  else "L✗"
    right_mark   = "R✓" if geometry.right_lane_detected else "R✗"
    center_mark  = " C✓" if geometry.center_lane_detected else ""

    lines = [
        f"Offset:    {geometry.offset_m:+.3f}m {offset_sign}",
        f"Curvature: {geometry.curvature_inv_m:+.4f} m⁻¹",
        f"Width:     {geometry.road_width_m:.2f}m",
        f"Lanes:     {left_mark} {right_mark}{center_mark}",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.55
    thickness  = 1
    pad        = 8
    line_gap   = 6

    # Measure all lines to size the background rectangle
    sizes = [cv2.getTextSize(ln, font, scale, thickness)[0] for ln in lines]
    panel_w = max(s[0] for s in sizes) + pad * 2
    line_h  = max(s[1] for s in sizes)
    panel_h = (line_h + line_gap) * len(lines) + pad * 2

    # Dark background with slight transparency
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

    # Render text lines
    y = pad + line_h
    for ln in lines:
        cv2.putText(out, ln, (pad, y), font, scale, (230, 230, 230), thickness,
                    cv2.LINE_AA)
        y += line_h + line_gap

    return out


# ---------------------------------------------------------------------------
# BEV minimap
# ---------------------------------------------------------------------------


def draw_bev_minimap(
    image: np.ndarray,
    bev_image: np.ndarray,
    position: str = "top_right",
    scale: float = 0.25,
) -> np.ndarray:
    """Overlay a scaled BEV image in one corner of the main frame.

    Args:
        image:     BGR source frame.  Never mutated.
        bev_image: Bird's-eye-view BGR image to embed as a minimap.
        position:  Corner to place the minimap.  One of
                   ``"top_right"``, ``"top_left"``, ``"bottom_right"``,
                   ``"bottom_left"``.  Defaults to ``"top_right"``.
        scale:     Fraction of the main image dimensions used for the minimap.
                   ``0.25`` means the minimap is 25% of the original width and
                   height.  Clamped to ``[0.05, 0.5]``.

    Returns:
        A new BGR array with the minimap composited in the requested corner.
    """
    out = image.copy()
    H, W = out.shape[:2]
    scale = float(np.clip(scale, 0.05, 0.5))

    mini_w = int(W * scale)
    mini_h = int(H * scale)
    mini   = cv2.resize(bev_image, (mini_w, mini_h), interpolation=cv2.INTER_AREA)

    border = 2
    pos    = position.lower()
    if pos == "top_left":
        x0, y0 = 0, 0
    elif pos == "bottom_left":
        x0, y0 = 0, H - mini_h
    elif pos == "bottom_right":
        x0, y0 = W - mini_w, H - mini_h
    else:  # top_right (default)
        x0, y0 = W - mini_w, 0

    # White border
    x0b, y0b = max(0, x0 - border), max(0, y0 - border)
    x1b, y1b = min(W, x0 + mini_w + border), min(H, y0 + mini_h + border)
    cv2.rectangle(out, (x0b, y0b), (x1b, y1b), (255, 255, 255), thickness=border)

    # Paste minimap (clip to image bounds)
    paste_h = min(mini_h, H - y0)
    paste_w = min(mini_w, W - x0)
    out[y0:y0 + paste_h, x0:x0 + paste_w] = mini[:paste_h, :paste_w]

    return out


# ---------------------------------------------------------------------------
# Full pipeline visualization
# ---------------------------------------------------------------------------


def create_full_visualization(
    original_image: np.ndarray,
    detection: LaneDetectionResult,
    geometry: LaneGeometry,
    bev_image: Optional[np.ndarray] = None,
    command: Optional[str] = None,
    confidence: Optional[float] = None,
    decision_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
) -> np.ndarray:
    """Compose all overlays into one demo-ready annotated frame.

    Overlay order (each layer is composited on the result of the previous):

    1. :func:`draw_lane_corridor` — filled corridor between lanes (when both
       are detected).
    2. :func:`draw_lane_lines`   — per-lane polylines and point markers.
    3. :func:`draw_geometry_info` — top-left metric panel.
    4. :func:`draw_bev_minimap`  — BEV thumbnail in the top-right corner
       (only when *bev_image* is provided).
    5. :func:`draw_decision_overlay` — bottom banner with command, confidence,
       and decision source (only when *command* is provided).

    Args:
        original_image: BGR camera frame.  Never mutated.
        detection:      Lane detection result for this frame.
        geometry:       Computed lane geometry for this frame.
        bev_image:      Optional bird's-eye-view frame to embed as a minimap.
        command:        Optional driving command string (``"FORWARD"`` etc.).
        confidence:     Decision confidence in ``[0, 1]``.  Used only when
                        *command* is provided.
        decision_path:  Human-readable decision source label.  Used only when
                        *command* is provided.
        config:         Visualization styling config.  Uses
                        :class:`VisualizationConfig` defaults when ``None``.

    Returns:
        A fully annotated BGR image ready for ``cv2.imwrite`` or display.
    """
    if config is None:
        config = VisualizationConfig()

    frame = draw_lane_corridor(original_image, detection, geometry, config)
    frame = draw_geometry_info(frame, geometry)

    if bev_image is not None:
        frame = draw_bev_minimap(frame, bev_image, position="top_right", scale=0.25)

    if command is not None:
        frame = draw_decision_overlay(
            frame,
            command=command,
            confidence=float(confidence) if confidence is not None else 0.0,
            decision_path=decision_path or "Geometric",
        )

    return frame


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from app.lane_detection.bev_transform import BEVConfig, BEVTransform
    from app.lane_detection.lane_geometry import LaneGeometryComputer
    from app.lane_detection.ufld_model import UFLDLaneDetector

    SOURCE_DIR  = Path("rgb")
    OUTPUT_DIR  = Path("outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = "configs/lane_detection.yaml"

    # Collect up to 3 source images
    image_paths = sorted(SOURCE_DIR.glob("rgb_image_*.png"))[:3]
    if not image_paths:
        log.error("No images found in '%s/'. Nothing to do.", SOURCE_DIR)
        sys.exit(1)

    # Load components
    try:
        detector = UFLDLaneDetector(CONFIG_PATH)
        bev_cfg  = BEVConfig.from_yaml(CONFIG_PATH)
        bev      = BEVTransform(bev_cfg)
        vis_cfg  = VisualizationConfig.from_yaml(CONFIG_PATH)
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        log.error("Initialisation error: %s", exc)
        sys.exit(1)

    import yaml as _yaml
    with open(CONFIG_PATH, encoding="utf-8") as _fh:
        _full_cfg = _yaml.safe_load(_fh)
    computer = LaneGeometryComputer(bev, _full_cfg.get("confidence", {}))

    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning("Cannot read '%s' — skipping.", img_path)
            continue

        log.info("Processing %s …", img_path.name)

        result   = detector.predict(frame)
        geometry = computer.compute(result)
        bev_img  = bev.transform_image(frame)

        annotated = create_full_visualization(
            original_image=frame,
            detection=result,
            geometry=geometry,
            bev_image=bev_img,
            command="FORWARD",
            confidence=0.88,
            decision_path="Geometric",
            config=vis_cfg,
        )

        out_path = OUTPUT_DIR / f"viz_{idx}.png"
        cv2.imwrite(str(out_path), annotated)
        log.info("Saved → %s", out_path)

    print("Visualizations saved to outputs/")
