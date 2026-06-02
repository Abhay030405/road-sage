# RoadSage 🛣️

**Vision-Based Lane Understanding & Intelligent Driving Decision Engine**
for MNNIT Allahabad Campus Roads

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-orange)](https://onnxruntime.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

RoadSage is a production-grade autonomous driving decision engine that takes a single forward-facing road image and outputs a driving command — `FORWARD`, `LEFT`, `RIGHT`, or `STOP` — along with lane geometry, obstacle context, a confidence score, and a GradCAM explainability overlay. It is designed specifically for MNNIT campus roads and runs entirely in CPU mode using ONNX Runtime, with no GPU required.

The core insight driving the architecture: **driving decisions are deterministic functions of geometry**. If you can accurately measure lane offset and curvature, the correct command follows from physics — no labeled training data needed. ML is used only where geometry is ambiguous, and every decision is explainable and traceable.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Model Choices](#model-choices)
- [Decision Engine](#decision-engine)
- [API Reference](#api-reference)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Edge Deployment](#edge-deployment)
- [Development Status](#development-status)

---

## System Architecture

```
Raw Image
    │
    ▼
┌─────────────────────┐
│   Preprocessing     │  CLAHE, denoise, perspective warp
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────┐
    ▼                             ▼
┌──────────────────┐   ┌──────────────────────┐
│  Lane Detection  │   │  Scene Understanding  │
│  UFLD v2 R18     │   │  NanoDet + MiDaS      │
│  + BEV Transform │   │  + Surface Classifier │
└────────┬─────────┘   └──────────┬────────────┘
         │                        │
         └───────────┬────────────┘
                     ▼
         ┌───────────────────────┐
         │   Decision Engine     │
         │   (Hybrid Logic)      │
         │                       │
         │  1. Safety Gate       │  Hard obstacle stop
         │  2. Geometric Logic   │  Lane offset + curvature
         │  3. Single-Lane Fall  │  One lane detected
         │  4. ML Fallback CNN   │  MobileNetV3-Small
         │  5. Confidence Gate   │  STOP if uncertain
         └──────────┬────────────┘
                    │
         ┌──────────┴────────────┐
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│  Safety &       │   │  Explainability       │
│  Confidence     │   │  GradCAM + Lane Viz   │
│  Gate           │   └──────────────────────┘
└────────┬────────┘
         ▼
┌─────────────────────────────────────┐
│   FastAPI  /predict  /batch  /ws    │
└─────────────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│   React Dashboard  (WebSocket)      │
└─────────────────────────────────────┘
```

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/yourteam/roadsage
cd roadsage

# Download model weights
bash models/download_models.sh

# Start the full stack
docker-compose up --build
```

Services:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

### Python (local development)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Single image inference
python -m roadsage.engine --image path/to/road_image.jpg --output results/

# Start API server
uvicorn api.main:app --reload --port 8000
```

### Single Inference (Python SDK)

```python
from roadsage import RoadSageEngine

engine = RoadSageEngine(config="configs/production.yaml")
result = engine.predict("frame_042.jpg")

# {
#   "command":      "FORWARD",
#   "confidence":   0.91,
#   "lane_offset":  0.12,      # meters from center (+ = right drift)
#   "curvature":    0.002,     # 1/radius m⁻¹
#   "hazard":       False,
#   "gradcam_path": "outputs/gradcam_042.jpg",
#   "lane_viz_path": "outputs/lane_042.jpg",
#   "latency_ms":   47.3
# }
```

---

## Project Structure

```
roadsage/
├── configs/
│   ├── production.yaml
│   ├── development.yaml
│   ├── lane_detection.yaml
│   ├── decision_engine.yaml
│   └── augmentation.yaml
│
├── roadsage/                      # Main Python package
│   ├── engine.py                  # RoadSageEngine — single inference entry point
│   ├── preprocessing/
│   │   ├── augmentation.py        # albumentations pipeline
│   │   ├── image_quality.py       # 4-gate quality filter
│   │   └── perspective.py         # IPM / BEV transform
│   ├── lane_detection/
│   │   ├── ufld_model.py          # UFLD v2 wrapper
│   │   ├── bev_transform.py       # Bird's eye view calibration
│   │   └── lane_geometry.py       # Offset, curvature, vanishing point
│   ├── scene_understanding/
│   │   ├── object_detector.py     # NanoDet-Plus-m wrapper
│   │   ├── depth_estimator.py     # MiDaS v2.1 Small wrapper
│   │   └── surface_classifier.py  # Road surface quality
│   ├── decision/
│   │   ├── geometric_logic.py     # Pure rule-based engine
│   │   ├── ml_fallback.py         # MobileNetV3-Small inference
│   │   ├── safety_gate.py         # Hard safety overrides
│   │   └── confidence_fusion.py   # Weighted confidence blending
│   ├── explainability/
│   │   ├── gradcam.py             # GradCAM overlay generation
│   │   └── visualizer.py          # Lane + decision visualization
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── kalman_filter.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── predict.py             # POST /api/v1/predict
│   │   ├── health.py              # GET  /api/v1/health
│   │   └── batch.py               # POST /api/v1/batch
│   ├── websocket/
│   │   └── stream.py              # WS  /ws/live
│   └── middleware/
│       ├── logging.py
│       └── rate_limit.py
│
├── training/
│   ├── trainers/
│   │   ├── train_lane.py
│   │   └── train_decision.py
│   ├── scripts/
│   │   ├── generate_pseudo_labels.py
│   │   ├── filter_dataset.py
│   │   └── export_onnx.py
│   └── evaluation/
│       ├── evaluate_lane.py
│       └── evaluate_decision.py
│
├── dashboard/                     # React + TailwindCSS
│   └── src/components/
│       ├── VideoFeed.jsx
│       ├── DecisionPanel.jsx
│       ├── ConfidenceMeter.jsx
│       ├── LaneMetrics.jsx
│       ├── GradCamView.jsx
│       ├── DecisionHistory.jsx
│       └── SystemHealth.jsx
│
├── models/
│   └── download_models.sh
│
├── tests/
│   ├── test_lane_detection.py
│   ├── test_decision_engine.py
│   ├── test_safety_gate.py
│   └── test_api.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/roadsage.json
│
├── docker-compose.yml
├── requirements.txt
└── requirements-edge.txt          # Inference-only deps for Pi/Jetson
```

---

## Pipeline Overview

### 1. Data Quality Filtering

Every collected image passes four automatic gates before entering any training pipeline:

| Gate | Method | Threshold | Rejects |
|---|---|---|---|
| Blur | Laplacian variance | < 50 | Motion blur, camera shake |
| Brightness | Mean pixel value | < 30 or > 220 | Under/overexposed frames |
| Road coverage | HSV road pixel ratio | < 20% | Non-road images |
| Deduplication | Perceptual hash (pHash) | similarity > 0.98 | Near-duplicate frames |

Result on the MNNIT dataset: 1,302 verified images from 1,403 originals (98 duplicates removed).

### 2. Lane Detection

UFLD v2 reformulates lane detection as a **row-anchor classification** problem. For each horizontal row of the image, it predicts which column the lane passes through. This is dramatically faster than dense segmentation and handles curves naturally.

After detection, an Inverse Perspective Mapping (BEV) transform converts the perspective view to top-down, where polynomial fitting yields:

- **Lane offset** (meters from center; negative = left drift, positive = right drift)
- **Road curvature** (1/radius in m⁻¹; positive = right curve)
- **Vanishing point** (where lanes converge ahead)

BEV calibration uses real MNNIT road geometry:
```python
src_points = [[65,455],[555,440],[380,285],[195,285]]
output_size = (640, 480)
pixels_per_meter = 25.0
```

### 3. Scene Understanding

NanoDet-Plus-m detects obstacles in the lane corridor (center 40% of frame width). MiDaS v2.1 Small produces a relative inverse depth map. Fusion logic extracts the maximum inverse depth value within each bounding box — the closest path obstacle.

If `nearest_obstacle_depth > stop_threshold`, `immediate_hazard = True`.

### 4. Decision Engine

The engine applies a strict priority chain:

```
Priority 1  Safety Gate       Hard obstacle stop — no exceptions
Priority 2  Geometric Logic   Both lanes detected → offset + curvature rule
Priority 3  Single-Lane Fall  One lane detected → steer toward center
Priority 4  ML Fallback CNN   No lanes → MobileNetV3-Small prediction
Priority 5  Confidence Gate   STOP if final_confidence < 0.60
```

**Critical design rule: STOP is never smoothed or delayed.** The temporal persistence filter (command must appear in ≥ 3 consecutive frames) and moving-average smoothing (over last 5 frames) are bypassed entirely for STOP commands. Safety cannot be traded for stability.

### 5. Confidence Fusion

```python
final_confidence = (
    0.40 * lane_detection_confidence +
    0.35 * geometric_signal_strength +
    0.25 * ml_softmax_max_value
)
```

Weights are configurable. If `final_confidence < 0.60`, the output command is overridden to STOP.

### 6. Explainability

Every prediction generates:
- **GradCAM overlay** — highlights which image regions drove the decision
- **Lane visualization** — color-coded by confidence (green ≥ 0.85, yellow 0.65–0.85, red < 0.65)
- **Decision trace log** — structured JSON recording every intermediate measurement and which decision path was taken

---

## Model Choices

All models run in CPU mode via ONNX Runtime. GPU-mode equivalents are defined and ready to swap via environment variable.

| Layer | CPU Mode (active) | GPU Mode (swap-ready) | CPU Latency |
|---|---|---|---|
| Lane detection | UFLD v2 ResNet-18 + ONNX | UFLD v2 ResNet-50 | ~30ms |
| Object detection | NanoDet-Plus-m + ONNX | YOLOv8n | ~10ms |
| Depth estimation | MiDaS v2.1 Small + ONNX | Depth Anything v2 | ~28ms |
| Fallback CNN | MobileNetV3-Small + ONNX | EfficientNet-Lite0 | ~4ms |
| Decision engine | Pure Python rules | — | ~2ms |
| **Total** | | | **< 74ms** |

### Switching to GPU mode

```bash
cp .env.gpu .env
uvicorn api.main:app --reload
```

---

## Decision Engine

### Geometric Command Logic

```
Dual-lane detected:
  |offset| > 0.3m  AND  offset > 0   →  LEFT   (drifted right)
  |offset| > 0.3m  AND  offset < 0   →  RIGHT  (drifted left)
  curvature > +0.005 m⁻¹             →  RIGHT  (road curves right)
  curvature < -0.005 m⁻¹             →  LEFT   (road curves left)
  otherwise                          →  FORWARD

Single-lane detected:
  Only right lane visible             →  LEFT   (steer toward center)
  Only left lane visible              →  RIGHT

No lanes detected:
  → ML fallback CNN
```

### Tunable Thresholds (`configs/decision_engine.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `OFFSET_THRESHOLD` | 0.3 m | Lateral drift to trigger correction |
| `CURVE_THRESHOLD` | 0.005 m⁻¹ | Curvature to trigger turn command |
| `OBSTACLE_STOP_DIST` | 2.0 m | Distance to trigger hard stop |
| `MIN_CONFIDENCE` | 0.60 | Below this → STOP |
| `LANE_CONF_THRESHOLD` | 0.75 | Minimum lane confidence to use |
| `PERSISTENCE_FRAMES` | 3 | Frames a command must persist before execution |
| `SMOOTHING_WINDOW` | 5 | Frames for moving-average smoothing |

---

## API Reference

### `POST /api/v1/predict`

Single image inference.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
     -F "image=@road_image.jpg" \
     -F "include_viz=true"
```

Response:

```json
{
  "command":       "FORWARD",
  "confidence":    0.91,
  "lane_offset":   0.12,
  "curvature":     0.002,
  "hazard":        false,
  "decision_path": "geometric",
  "latency_ms":    47.3,
  "gradcam_url":   "/outputs/gradcam_042.jpg",
  "lane_viz_url":  "/outputs/lane_042.jpg"
}
```

### `GET /api/v1/health`

Returns model load state, rolling average latency, memory usage, and config hash. Runs a dummy inference to verify the model stack is operational.

### `POST /api/v1/batch`

Accepts a ZIP file or list of base64-encoded images. Processes in parallel (concurrency limit: 4). Returns full trace for each frame.

### `WS /ws/live`

WebSocket streaming. Send frames as binary messages; receive JSON predictions. Rate: up to 30 FPS input, processed at 15 FPS.

Rate limits: `/predict` — 30 req/min per IP; `/batch` — 5 req/min per IP.

---

## Training Pipeline

### Self-Training Loop

RoadSage uses no manual command labels. The training loop derives commands from geometry and uses them to train the ML fallback:

```
Iteration 0  Pretrain UFLD v2 on TuSimple + CULane (public labeled data)

Iteration 1  Run on MNNIT images
             Accept predictions where lane_confidence > 0.85
             Derive commands via geometric logic → pseudo-labels
             Fine-tune lane detector + decision CNN
             Coverage: ~60% of MNNIT images

Iteration 2  Improved model → more images exceed threshold
             Coverage: ~80–85%
             Fine-tune again

Iteration 3  Human review of 50 random samples
             Final fine-tune on full pseudo-labeled set
```

### Running the Pipeline

```bash
# Pretrain lane detector on public data
python training/trainers/train_lane.py \
    --config configs/lane_detection.yaml \
    --dataset tusimple+culane \
    --epochs 100

# Generate pseudo-labels for MNNIT data
python training/scripts/generate_pseudo_labels.py \
    --model checkpoints/lane_best.pth \
    --input data/mnnit/raw/ \
    --output data/mnnit/pseudo_labels/ \
    --min_confidence 0.85

# Train decision CNN on geometric pseudo-labels
python training/trainers/train_decision.py \
    --config configs/decision_cnn.yaml \
    --dataset mnnit_with_commands \
    --epochs 50

# Export all models to ONNX
python training/scripts/export_onnx.py --all
```

### Augmentation Strategy

The augmentation suite targets specific MNNIT road conditions:

| Augmentation | Targets |
|---|---|
| CLAHE | Morning haze, low contrast, faded markings |
| RandomShadow | Tree shadows crossing lane lines |
| RandomBrightness/Contrast | Lighting across time of day |
| GaussianBlur + Sharpen | Camera shake and focus variation |
| HorizontalFlip (with lane mirroring) | Doubles dataset without collection |
| PerspectiveTransform | Camera mount angle variation |
| RandomRain/Fog | Edge case weather |

Library: `albumentations` (10× faster than torchvision transforms for complex augmentation pipelines).

---

## Configuration

All runtime parameters live in `configs/`. Nothing is hardcoded in source.

```yaml
# configs/decision_engine.yaml (excerpt)
thresholds:
  offset_m: 0.3
  curvature_inv_m: 0.005
  obstacle_stop_depth: 0.82      # MiDaS inverse depth calibrated for ~2.0m
  min_confidence: 0.60
  lane_confidence: 0.75

temporal:
  persistence_frames: 3
  smoothing_window: 5
  stop_bypasses_smoothing: true  # CRITICAL: never change to false

confidence_weights:
  lane: 0.40
  geometric: 0.35
  ml_fallback: 0.25
```

```yaml
# configs/lane_detection.yaml (excerpt)
bev:
  src_points: [[65,455],[555,440],[380,285],[195,285]]
  output_size: [640, 480]
  pixels_per_meter: 25.0

model:
  backbone: resnet18              # switch to resnet50 on GPU
  confidence_threshold: 0.75
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# Individual suites
pytest tests/test_lane_detection.py   # BEV transform, geometry, UFLD wrapper
pytest tests/test_decision_engine.py  # All 5 priority chain branches
pytest tests/test_safety_gate.py      # Safety invariants (STOP never delayed)
pytest tests/test_api.py              # All endpoints via httpx TestClient
```

Tests are pure unit tests — no model weights required. The decision engine modules are fully functional (no I/O, no side effects), so every branch of the priority chain is testable with synthetic inputs.

**Property test (safety invariant):**

```python
@given(offset=st.floats(min_value=0.31, max_value=5.0))
def test_stop_is_never_smoothed(offset):
    # For any frame with immediate_hazard=True,
    # the output command must always be STOP regardless of history
    ...
```

---

## Monitoring

Prometheus metrics at `/metrics`:

| Metric | Type | Meaning |
|---|---|---|
| `roadsage_inference_latency_seconds` | Histogram | End-to-end prediction latency |
| `roadsage_command_total` | Counter | Predictions by command type |
| `roadsage_confidence_histogram` | Histogram | Distribution of confidence scores |
| `roadsage_safety_gate_triggers_total` | Counter | Safety gate activations |
| `roadsage_lane_detection_failures_total` | Counter | Frames with no lanes detected |
| `roadsage_ml_fallback_activations_total` | Counter | ML fallback usage rate |

Import the pre-built Grafana dashboard:

```bash
# Grafana available at localhost:3001 after docker-compose up
# Import monitoring/grafana/dashboards/roadsage.json
```

A spike in `safety_gate_triggers` indicates a new obstacle type not being handled. A spike in `ml_fallback_activations` indicates deteriorating lane detection — likely faded markings or a new road section.

---

## Edge Deployment

All models are exported to ONNX and run via ONNX Runtime. This removes PyTorch and all training dependencies from the inference image.

```bash
# Build edge image (~1.2GB vs ~8GB full image)
docker build -f Dockerfile.edge -t roadsage-edge .

# Uses requirements-edge.txt:
# onnxruntime, opencv-python-headless, fastapi, uvicorn, numpy
```

Target latency budget on Raspberry Pi 4 / Jetson Nano: P95 < 100ms end-to-end. GradCAM is lazy on edge — generated only on explicit request, not per frame.

---

## Development Status

### Completed

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Foundation & Data Pipeline | ✅ Complete | 1,302 verified images, 4-gate quality filter, augmentation pipeline, Docker stack |
| Phase 2 — BEV & Lane Detection | ✅ Complete | BEV calibrated on real MNNIT image, UFLD v2 wrapper, lane geometry; 46/50 tests pass |
| Phase 3 — Scene Understanding | ✅ Complete | NanoDet, MiDaS, surface classifier, SceneAnalyzer fusion; 34/34 tests pass |
| Phase 4 — Decision Engine | ✅ Complete | Full priority chain, safety gate, confidence fusion, ML fallback training scripts; all logic tests pass |
| Phase 5 — System Integration & API | 🔄 Next | RoadSageEngine, GradCAM, all FastAPI endpoints, WebSocket streaming |

### Pending (Phase 5+)

- `RoadSageEngine` full integration wiring all modules
- GradCAM implementation against UFLD v2 backbone
- All FastAPI endpoints and WebSocket stream
- React dashboard (7 components)
- ONNX export and edge validation
- Full evaluation on held-out MNNIT frames

### Known Gaps

Two items from Phase 4 are blocked on external ONNX weights download (not a code issue):
- `data/mnnit/pseudo_labels/*.jsonl` — run `generate_pseudo_labels.py` once weights are available
- `models/decision_cnn.onnx` — run `train_decision.py` then `export_onnx.py`

All downstream phases proceed independently of these.

---

## Production Readiness Targets

| Metric | Target |
|---|---|
| Command accuracy | > 88% on manually annotated MNNIT frames |
| STOP precision | > 99% on safety-critical test cases |
| P95 inference latency | < 100ms |
| Reliability | Zero crashes over 500-frame continuous run |
| Uncertainty calibration (ECE) | < 0.05 |
| Lane detection F1 | > 0.85 on held-out test set |

---

## Key References

- UltraFast Lane Detection v2 — Qinghao Feng et al., 2022
- MiDaS: Towards Robust Monocular Depth Estimation — Ranftl et al., 2020
- Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method — Lee, 2013
- GradCAM: Visual Explanations from Deep Networks — Selvaraju et al., 2017
- End-to-End Learning for Self-Driving Cars (DAVE-2) — Bojarski et al., NVIDIA, 2016

---

*RoadSage — Seeing the road, understanding the path.*
*Built for MNNIT Allahabad | CPU-native autonomous navigation*
