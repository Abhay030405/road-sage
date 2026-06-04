"""
api.metrics
============

Prometheus metric objects shared across all API layers.

Centralising metric definitions here prevents circular imports between
``api.main``, ``api.routes``, and ``api.websocket``.  All consumers import
directly from this module; ``api.main`` re-exports everything for backwards
compatibility.

Metric reference
----------------
``roadsage_http_requests_total`` — counter, labels: method/endpoint/status
``roadsage_request_latency_seconds`` — histogram, label: endpoint
``roadsage_command_total`` — counter, label: command
``roadsage_safety_gate_triggers_total`` — counter
``roadsage_lane_detection_failures_total`` — counter
``roadsage_ml_fallback_activations_total`` — counter
``roadsage_confidence_histogram`` — histogram (confidence score buckets)
``roadsage_inference_latency_seconds`` — histogram, label: component
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from app.engine import PredictionResult

# ---------------------------------------------------------------------------
# HTTP-level metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "roadsage_http_requests_total",
    "Total HTTP requests received by the API.",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "roadsage_request_latency_seconds",
    "End-to-end HTTP request latency in seconds.",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1.0],
)

# ---------------------------------------------------------------------------
# Prediction-level metrics
# ---------------------------------------------------------------------------

COMMAND_COUNT = Counter(
    "roadsage_command_total",
    "Cumulative count of driving commands predicted, by command type.",
    ["command"],
)

SAFETY_GATE_COUNT = Counter(
    "roadsage_safety_gate_triggers_total",
    "Number of frames where the safety gate forced a STOP.",
)

LANE_DETECTION_FAILURES = Counter(
    "roadsage_lane_detection_failures_total",
    "Number of frames where neither left nor right lane was detected.",
)

ML_FALLBACK_COUNT = Counter(
    "roadsage_ml_fallback_activations_total",
    "Number of frames decided by the ML fallback model instead of geometric logic.",
)

# ---------------------------------------------------------------------------
# New metrics (Phase 6 observability)
# ---------------------------------------------------------------------------

CONFIDENCE_HISTOGRAM = Histogram(
    "roadsage_confidence_histogram",
    "Distribution of fused prediction confidence scores.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

INFERENCE_LATENCY = Histogram(
    "roadsage_inference_latency_seconds",
    "Per-component inference latency in seconds.",
    ["component"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def record_prediction_metrics(result: "PredictionResult") -> None:
    """Record all per-prediction Prometheus metrics for one inference result.

    Call this once after every successful ``engine.predict()`` invocation,
    both from the HTTP predict endpoint and the WebSocket stream handler.

    Parameters
    ----------
    result:
        The :class:`~app.engine.PredictionResult` returned by
        :meth:`~app.engine.RoadSageEngine.predict`.
    """
    COMMAND_COUNT.labels(command=result.command).inc()
    CONFIDENCE_HISTOGRAM.observe(result.confidence)

    lat = result.latency_ms
    INFERENCE_LATENCY.labels(component="lane").observe(lat.get("lane", 0.0) / 1000.0)
    INFERENCE_LATENCY.labels(component="scene").observe(lat.get("scene", 0.0) / 1000.0)
    INFERENCE_LATENCY.labels(component="decision").observe(lat.get("decision", 0.0) / 1000.0)
    INFERENCE_LATENCY.labels(component="total").observe(lat.get("total", 0.0) / 1000.0)

    if result.decision_path == "safety_gate":
        SAFETY_GATE_COUNT.inc()

    if not result.left_lane_detected and not result.right_lane_detected:
        LANE_DETECTION_FAILURES.inc()

    if result.decision_path == "ml_fallback":
        ML_FALLBACK_COUNT.inc()
