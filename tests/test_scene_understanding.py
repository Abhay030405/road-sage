"""Phase 3 test suite — scene understanding: ObjectDetector, DepthEstimator,
SurfaceClassifier, and SceneAnalyzer.

All tests must pass without ONNX model weights being present.

Sections
--------
1. BoundingBox tests
2. DetectedObject and ObjectDetectionResult tests
3. NMS and IoU tests
4. DepthEstimationResult tests
5. SurfaceClassifier tests
6. SceneContext tests
7. SceneAnalyzer integration tests (no model needed)
"""

from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.scene_understanding.depth_estimator import DepthEstimationResult
from app.scene_understanding.object_detector import (
    BoundingBox,
    DetectedObject,
    NanoDetPostprocessor,
    ObjectDetectionResult,
    ObjectDetector,
)
from app.scene_understanding.surface_classifier import (
    SurfaceClass,
    SurfaceClassificationResult,
    SurfaceClassifier,
)
from app.scene_understanding import SceneAnalyzer, SceneContext


# ===========================================================================
# Section 1 — BoundingBox Tests
# ===========================================================================


def test_bbox_dimensions() -> None:
    """BoundingBox correctly reports width, height, area, and centre."""
    bbox = BoundingBox(x1=100, y1=150, x2=300, y2=400)
    assert bbox.width() == 200
    assert bbox.height() == 250
    assert bbox.area() == 50000
    assert bbox.center() == (200, 275)


def test_bbox_in_corridor_center() -> None:
    """A box centred at x=320 lies inside the central 40 % corridor (640 px)."""
    image_width = 640
    bbox = BoundingBox(x1=270, y1=100, x2=370, y2=300)  # centre x = 320
    assert bbox.is_in_corridor(image_width, 0.4) is True


def test_bbox_outside_corridor_left() -> None:
    """A box on the far left lies outside the corridor."""
    bbox = BoundingBox(x1=10, y1=100, x2=80, y2=300)  # centre x = 45
    assert bbox.is_in_corridor(640, 0.4) is False


def test_bbox_outside_corridor_right() -> None:
    """A box on the far right lies outside the corridor."""
    bbox = BoundingBox(x1=580, y1=100, x2=630, y2=300)  # centre x = 605
    assert bbox.is_in_corridor(640, 0.4) is False


def test_bbox_corridor_boundary_left_edge() -> None:
    """A box centred exactly on the left corridor edge is inside (inclusive)."""
    # corridor_left = 640 * (0.5 - 0.4/2) = 640 * 0.3 = 192
    bbox = BoundingBox(x1=184, y1=0, x2=200, y2=100)   # centre x = 192
    assert bbox.is_in_corridor(640, 0.4) is True


def test_bbox_corridor_boundary_right_edge() -> None:
    """A box centred exactly on the right corridor edge is inside (inclusive)."""
    # corridor_right = 640 * (0.5 + 0.4/2) = 640 * 0.7 = 448
    bbox = BoundingBox(x1=440, y1=0, x2=456, y2=100)   # centre x = 448
    assert bbox.is_in_corridor(640, 0.4) is True


# ===========================================================================
# Section 2 — DetectedObject and ObjectDetectionResult Tests
# ===========================================================================


def test_detected_object_dataclass() -> None:
    """DetectedObject stores all fields correctly."""
    bbox = BoundingBox(100, 150, 300, 400)
    obj = DetectedObject(
        bbox=bbox,
        class_id=0,
        class_name="person",
        confidence=0.87,
        is_path_obstacle=True,
    )
    assert obj.confidence == pytest.approx(0.87)
    assert obj.class_name == "person"
    assert obj.is_path_obstacle is True
    assert obj.bbox.area() == bbox.area()


def test_detection_result_nearest_obstacle_none() -> None:
    """nearest_obstacle() returns None when there are no path obstacles."""
    result = ObjectDetectionResult(
        detections=[],
        path_obstacles=[],
        inference_time_ms=10.0,
        image_shape=(480, 640, 3),
    )
    assert result.nearest_obstacle() is None


def test_detection_result_nearest_obstacle_largest() -> None:
    """nearest_obstacle() returns the obstacle with the largest bbox area."""
    small = DetectedObject(
        BoundingBox(300, 200, 350, 250), 0, "person", 0.8, True
    )  # area = 50 * 50 = 2500
    large = DetectedObject(
        BoundingBox(200, 100, 500, 400), 2, "car", 0.9, True
    )  # area = 300 * 300 = 90000
    result = ObjectDetectionResult(
        detections=[small, large],
        path_obstacles=[small, large],
        inference_time_ms=10.0,
        image_shape=(480, 640, 3),
    )
    nearest = result.nearest_obstacle()
    assert nearest is not None
    assert nearest.class_name == "car"


def test_detection_result_nearest_obstacle_single() -> None:
    """nearest_obstacle() returns the only obstacle when there is exactly one."""
    obj = DetectedObject(BoundingBox(100, 100, 200, 200), 2, "car", 0.95, True)
    result = ObjectDetectionResult(
        detections=[obj],
        path_obstacles=[obj],
        inference_time_ms=5.0,
        image_shape=(480, 640, 3),
    )
    assert result.nearest_obstacle() is obj


# ===========================================================================
# Section 3 — NMS and IoU Tests
# ===========================================================================


@pytest.fixture()
def postprocessor() -> NanoDetPostprocessor:
    """Return a default NanoDetPostprocessor for IoU / NMS tests."""
    return NanoDetPostprocessor(
        conf_threshold=0.3,
        nms_iou_threshold=0.5,
        relevant_classes=[0, 2],
        class_names={0: "person", 2: "car"},
    )


def test_iou_no_overlap(postprocessor: NanoDetPostprocessor) -> None:
    """IoU is 0 for two non-overlapping boxes."""
    box1 = BoundingBox(0, 0, 100, 100)
    box2 = BoundingBox(200, 200, 300, 300)
    assert postprocessor._compute_iou(box1, box2) == pytest.approx(0.0)


def test_iou_full_overlap(postprocessor: NanoDetPostprocessor) -> None:
    """IoU is 1 for two identical boxes."""
    box1 = BoundingBox(0, 0, 100, 100)
    box2 = BoundingBox(0, 0, 100, 100)
    assert postprocessor._compute_iou(box1, box2) == pytest.approx(1.0)


def test_iou_partial_overlap(postprocessor: NanoDetPostprocessor) -> None:
    """IoU is computed correctly for two partially overlapping boxes.

    Intersection = 50×50 = 2500.
    Union = 10000 + 10000 - 2500 = 17500.
    Expected IoU = 2500 / 17500 ≈ 0.1429.
    """
    box1 = BoundingBox(0, 0, 100, 100)
    box2 = BoundingBox(50, 50, 150, 150)
    expected = 2500.0 / 17500.0
    assert postprocessor._compute_iou(box1, box2) == pytest.approx(expected, abs=1e-4)


def test_nms_suppresses_duplicate_boxes(postprocessor: NanoDetPostprocessor) -> None:
    """NMS keeps only the highest-confidence box when two boxes heavily overlap."""
    raw = np.zeros((1, 2, 5), dtype=np.float32)  # 4 + 1 class slot per row
    # First row: box at [10,20,110,120], class 0 confidence 0.9
    raw[0, 0, :4] = [10.0, 20.0, 110.0, 120.0]
    raw[0, 0, 4]  = 0.9
    # Second row: almost the same box, slightly lower confidence
    raw[0, 1, :4] = [12.0, 22.0, 112.0, 122.0]
    raw[0, 1, 4]  = 0.8

    dets = postprocessor.postprocess(raw, scale_factor=1.0, original_shape=(480, 640, 3))
    # Only one detection should survive NMS
    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.9, abs=0.01)


def test_nms_keeps_non_overlapping_boxes(postprocessor: NanoDetPostprocessor) -> None:
    """NMS keeps both boxes when they do not overlap."""
    raw = np.zeros((1, 2, 5), dtype=np.float32)
    raw[0, 0, :4] = [0.0, 0.0, 100.0, 100.0]
    raw[0, 0, 4]  = 0.9
    raw[0, 1, :4] = [300.0, 300.0, 400.0, 400.0]
    raw[0, 1, 4]  = 0.85

    dets = postprocessor.postprocess(raw, scale_factor=1.0, original_shape=(480, 640, 3))
    assert len(dets) == 2


# ===========================================================================
# Section 4 — DepthEstimationResult Tests
# ===========================================================================


def test_depth_result_at_bbox() -> None:
    """get_depth_at_bbox() returns the maximum depth value within the ROI."""
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[100:300, 200:400] = 0.85
    normalized = (depth_map * 255).astype(np.uint8)
    result = DepthEstimationResult(
        depth_map=depth_map,
        normalized_map=normalized,
        inference_time_ms=25.0,
        image_shape=(480, 640, 3),
    )
    bbox = BoundingBox(x1=200, y1=100, x2=400, y2=300)
    depth_val = result.get_depth_at_bbox(bbox, padding=0)
    assert abs(depth_val - 0.85) < 0.01


def test_depth_obstacle_close_true() -> None:
    """is_obstacle_close() returns True when depth exceeds the threshold."""
    depth_map = np.ones((480, 640), dtype=np.float32) * 0.9
    normalized = (depth_map * 255).astype(np.uint8)
    result = DepthEstimationResult(
        depth_map=depth_map,
        normalized_map=normalized,
        inference_time_ms=25.0,
        image_shape=(480, 640, 3),
    )
    bbox = BoundingBox(200, 100, 400, 300)
    assert result.is_obstacle_close(bbox, threshold=0.7) is True


def test_depth_obstacle_not_close() -> None:
    """is_obstacle_close() returns False when depth is below the threshold."""
    depth_map = np.ones((480, 640), dtype=np.float32) * 0.3
    normalized = (depth_map * 255).astype(np.uint8)
    result = DepthEstimationResult(
        depth_map=depth_map,
        normalized_map=normalized,
        inference_time_ms=25.0,
        image_shape=(480, 640, 3),
    )
    bbox = BoundingBox(200, 100, 400, 300)
    assert result.is_obstacle_close(bbox, threshold=0.7) is False


def test_depth_result_at_point() -> None:
    """get_depth_at_point() returns the mean depth in a small window."""
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[145:156, 295:306] = 0.75
    normalized = (depth_map * 255).astype(np.uint8)
    result = DepthEstimationResult(
        depth_map=depth_map, normalized_map=normalized,
        inference_time_ms=0.0, image_shape=(480, 640, 3),
    )
    val = result.get_depth_at_point(300, 150, radius=5)
    assert val > 0.0


def test_depth_result_empty_bbox_returns_zero() -> None:
    """get_depth_at_bbox() returns 0.0 when depth_map is the 1×1 default."""
    result = DepthEstimationResult()
    bbox = BoundingBox(0, 0, 100, 100)
    assert result.get_depth_at_bbox(bbox) == pytest.approx(0.0)


# ===========================================================================
# Section 5 — SurfaceClassifier Tests
# ===========================================================================


def test_surface_classifier_unknown_when_not_loaded() -> None:
    """SurfaceClassifier returns UNKNOWN when the ONNX model is absent."""
    classifier = SurfaceClassifier()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = classifier.classify(blank)
    assert result.surface_class == SurfaceClass.UNKNOWN
    assert result.confidence == pytest.approx(0.0)


def test_surface_classifier_center_patch_unknown_when_not_loaded() -> None:
    """classify_center_patch() also returns UNKNOWN without a model."""
    classifier = SurfaceClassifier()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = classifier.classify_center_patch(blank)
    assert result.surface_class == SurfaceClass.UNKNOWN


def test_surface_class_is_hazard() -> None:
    """SurfaceClassificationResult.is_hazard() flags pothole and waterlogged."""
    r_pothole  = SurfaceClassificationResult(SurfaceClass.POTHOLE,  0.90, 5.0)
    r_water    = SurfaceClassificationResult(SurfaceClass.WATERLOGGED, 0.88, 5.0)
    r_clean    = SurfaceClassificationResult(SurfaceClass.CLEAN, 0.95, 5.0)
    r_speed    = SurfaceClassificationResult(SurfaceClass.SPEED_BREAKER, 0.70, 5.0)
    r_unknown  = SurfaceClassificationResult(SurfaceClass.UNKNOWN, 0.0, 0.0)

    assert r_pothole.is_hazard() is True
    assert r_water.is_hazard()   is True
    assert r_clean.is_hazard()   is False
    assert r_speed.is_hazard()   is False
    assert r_unknown.is_hazard() is False


def test_surface_class_enum_values() -> None:
    """SurfaceClass enum values are lowercase strings."""
    assert SurfaceClass.CLEAN.value         == "clean"
    assert SurfaceClass.POTHOLE.value       == "pothole"
    assert SurfaceClass.SPEED_BREAKER.value == "speed_breaker"
    assert SurfaceClass.WATERLOGGED.value   == "waterlogged"
    assert SurfaceClass.UNKNOWN.value       == "unknown"


# ===========================================================================
# Section 6 — SceneContext Tests
# ===========================================================================


def test_scene_context_safe() -> None:
    """SceneContext with no hazard reports is_safe() == True."""
    context = SceneContext(immediate_hazard=False)
    assert context.is_safe() is True
    assert "Clear" in context.describe()


def test_scene_context_hazard() -> None:
    """SceneContext with an immediate hazard reports is_safe() == False."""
    context = SceneContext(
        immediate_hazard=True,
        hazard_reason="person detected in path",
    )
    assert context.is_safe() is False
    assert "HAZARD" in context.describe()
    assert "person detected in path" in context.describe()


def test_scene_context_describe_obstacle_count() -> None:
    """describe() includes the path obstacle count when the scene is clear."""
    detection_result = ObjectDetectionResult(
        path_obstacles=[
            DetectedObject(BoundingBox(100, 100, 200, 200), 2, "car", 0.8, True),
            DetectedObject(BoundingBox(300, 100, 400, 200), 0, "person", 0.7, True),
        ]
    )
    context = SceneContext(detections=detection_result, immediate_hazard=False)
    desc = context.describe()
    assert "2" in desc


def test_scene_context_surface_label_in_describe() -> None:
    """describe() includes the surface class label when the scene is clear."""
    surface = SurfaceClassificationResult(SurfaceClass.POTHOLE, 0.5, 3.0)
    context = SceneContext(surface=surface, immediate_hazard=False)
    desc = context.describe()
    assert "pothole" in desc


# ===========================================================================
# Section 7 — SceneAnalyzer Integration Tests (no model needed)
# ===========================================================================


def test_scene_analyzer_instantiates() -> None:
    """SceneAnalyzer can be constructed without model weights."""
    analyzer = SceneAnalyzer()
    assert hasattr(analyzer, "_detector")
    assert hasattr(analyzer, "_depth")
    assert hasattr(analyzer, "_classifier")


def test_scene_analyzer_is_not_fully_ready_without_models() -> None:
    """is_fully_ready() returns False when ONNX weights are absent."""
    analyzer = SceneAnalyzer()
    assert analyzer.is_fully_ready() is False


def test_scene_analyzer_runs_on_blank_image() -> None:
    """SceneAnalyzer.analyze() completes on a blank image without crashing."""
    analyzer = SceneAnalyzer()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(blank)
    assert isinstance(result, SceneContext)
    assert result.total_inference_time_ms >= 0
    # Without models no obstacle is detected, so scene should be safe
    assert result.is_safe() is True


def test_scene_analyzer_returns_scene_context_type() -> None:
    """analyze() always returns a SceneContext instance."""
    analyzer = SceneAnalyzer()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(blank)
    assert isinstance(result, SceneContext)
    assert isinstance(result.detections, ObjectDetectionResult)
    assert isinstance(result.depth, DepthEstimationResult)
    assert isinstance(result.surface, SurfaceClassificationResult)


def test_scene_analyzer_describe_is_non_empty() -> None:
    """describe() always returns a non-empty string."""
    analyzer = SceneAnalyzer()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.analyze(blank)
    assert len(result.describe()) > 0


def test_scene_analyzer_from_rgb_folder() -> None:
    """SceneAnalyzer can process a real image from the rgb/ folder."""
    images = sorted(glob.glob("rgb/rgb_image_*.png"))
    assert len(images) > 0, "rgb/ folder must contain at least one image"

    analyzer = SceneAnalyzer()
    image = cv2.imread(images[0])
    assert image is not None, f"Failed to read {images[0]}"

    result = analyzer.analyze(image)
    assert isinstance(result, SceneContext)
    assert len(result.describe()) > 0
