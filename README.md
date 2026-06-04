# RoadSage 🛣️
**Vision-Based Lane Understanding & Intelligent Driving Decision Engine**  
*MNNIT Allahabad — Campus Roads*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-orange)](https://onnxruntime.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

RoadSage takes a raw road image and produces:

| Output | Detail |
|---|---|
| **Driving Command** | `FORWARD` \| `LEFT` \| `RIGHT` \| `STOP` |
| **Lane Geometry** | lateral offset (m), curvature (m⁻¹), road width (m) |
| **Hazard Detection** | obstacle presence + monocular depth estimate |
| **Explainability** | GradCAM heatmap showing which pixels drove the decision |
| **Confidence Score** | fused score via Monte Carlo Dropout + geometric signal strength |

The core design principle: **driving decisions are deterministic functions of geometry**. If you can accurately measure lane offset and curvature, the correct command follows from physics — no labeled training data needed. ML is used only where geometry is ambiguous, and every decision is explainable and traceable.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt          # or: make install

# 2. Start the API
uvicorn api.main:app --reload --port 8000  # or: make api

# 3. Start the dashboard
cd dashboard && npm install && npm run dev  # or: make dashboard

# 4. Predict on a single image
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@rgb/rgb_image_1.png" | python -m json.tool

# 5. Full Docker stack (API + Dashboard + Prometheus + Grafana)
docker-compose up --build               # or: make docker-up
```

### Python SDK

```python
from app.engine import RoadSageEngine

engine = RoadSageEngine()                           # loads all model weights
result = engine.predict(cv2.imread("frame.png"))

print(result.command)          # "FORWARD"
print(result.confidence)       # 0.91
print(result.lane_offset_m)    # +0.12  (positive = drifted right)
print(result.decision_path)    # "geometric"
print(result.latency_ms)       # {"lane": 28.1, "scene": 9.4, "decision": 1.8, "total": 39.3}
```

### Makefile Targets

```
make install          install Python dependencies
make test             pytest tests/ -v
make test-integration pytest tests/test_integration.py -v
make lint             ruff check .
make lint-fix         ruff check . --fix
make api              uvicorn dev server on :8000
make dashboard        Vite dev server on :3000
make docker-up        full stack with Prometheus + Grafana
make evaluate         lane + decision evaluation on rgb/
make ablation         5-variant ablation study (50 images)
make stress-test      2-minute API stress test → populates Grafana
make phase7-check     Phase 7 exit-criteria validation
```

### Services (after `docker-compose up`)

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Dashboard | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

---

## Architecture

5-layer hybrid decision engine:

```
Raw Image
    │
    ▼
┌─────────────────────┐
│   Preprocessing     │  CLAHE, denoise, perspective warp
└──────────┬──────────┘
           │
    ┌──────┴───────────────────────┐
    ▼                              ▼
┌──────────────────┐   ┌───────────────────────┐
│  Lane Detection  │   │   Scene Understanding  │
│  UFLD v2 R18     │   │   NanoDet + MiDaS      │
│  + BEV Transform │   │   + Surface Classifier │
└────────┬─────────┘   └───────────┬────────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
          ┌───────────────────────┐
          │    Decision Engine    │
          │    (Hybrid Logic)     │
          │                       │
          │  1. Safety Gate       │  ← hard obstacle stop
          │  2. Geometric Logic   │  ← lane offset + curvature
          │  3. Single-Lane Fall  │  ← one lane detected
          │  4. ML Fallback CNN   │  ← MobileNetV3-Small
          │  5. Confidence Gate   │  ← STOP if uncertain
          └──────────┬────────────┘
                     │
          ┌──────────┴────────────┐
          ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│  Temporal        │   │  Explainability       │
│  Smoothing       │   │  GradCAM + Lane Viz   │
│  (STOP bypasses) │   └──────────────────────┘
└────────┬─────────┘
         ▼
┌──────────────────────────────────────┐
│   FastAPI  /predict  /batch  /ws     │
└──────────────────────────────────────┘
         ▼
┌──────────────────────────────────────┐
│   React Dashboard  (WebSocket)       │
└──────────────────────────────────────┘
```

**Critical design rule:** `STOP` is never smoothed, delayed, or buffered. The temporal consistency filter (3-frame persistence) and moving-average smoothing (5-frame window) are bypassed entirely for `STOP` commands. Safety cannot be traded for stability.

---

## Project Structure

```
Road-Sage/
├── app/                               # Core Python package
│   ├── engine.py                      # RoadSageEngine — single inference entry point
│   ├── lane_detection/
│   │   ├── ufld_model.py              # UFLD v2 ResNet-18 ONNX wrapper
│   │   ├── bev_transform.py           # Bird's-eye-view calibration
│   │   └── lane_geometry.py           # Offset, curvature, vanishing point
│   ├── scene_understanding/
│   │   ├── object_detector.py         # NanoDet-Plus-m wrapper
│   │   ├── depth_estimator.py         # MiDaS v2.1 Small wrapper
│   │   └── surface_classifier.py      # Road surface quality classifier
│   ├── decision/
│   │   ├── geometric_logic.py         # Pure rule-based engine
│   │   ├── ml_fallback.py             # MobileNetV3-Small inference
│   │   ├── safety_gate.py             # Hard safety overrides
│   │   └── confidence_fusion.py       # Weighted confidence blending
│   ├── explainability/
│   │   ├── gradcam.py                 # GradCAM overlay generation
│   │   └── visualizer.py             # Lane + decision visualization
│   └── utils/
│       ├── config_validator.py
│       ├── logger.py
│       ├── metrics.py
│       └── kalman_filter.py
│
├── api/
│   ├── main.py                        # FastAPI app + lifespan
│   ├── metrics.py                     # Prometheus metric objects (shared)
│   ├── routes/
│   │   ├── predict.py                 # POST /api/v1/predict
│   │   ├── health.py                  # GET  /api/v1/health
│   │   └── batch.py                   # POST /api/v1/batch
│   ├── websocket/
│   │   └── stream.py                  # WS  /ws/live
│   └── middleware/
│       ├── logging.py
│       └── rate_limit.py
│
├── dashboard/                         # React + TailwindCSS
│   └── src/components/
│       ├── VideoFeed.jsx
│       ├── DecisionPanel.jsx
│       ├── ConfidenceMeter.jsx
│       ├── LaneMetrics.jsx
│       ├── GradCamView.jsx
│       ├── DecisionHistory.jsx
│       └── SystemHealth.jsx
│
├── training/
│   ├── trainers/
│   │   ├── train_lane.py
│   │   └── train_decision.py
│   ├── scripts/
│   │   ├── generate_pseudo_labels.py
│   │   ├── filter_dataset.py
│   │   ├── export_onnx.py
│   │   ├── run_ablation.py            # 5-variant component ablation study
│   │   └── metrics_stress_test.py     # API load test → populates Grafana
│   └── evaluation/
│       ├── evaluate_lane.py           # Lane detector proxy metrics
│       ├── evaluate_decision.py       # Decision pipeline + confusion matrix
│       ├── evaluate_phase5.py         # Phase 5 exit checklist
│       └── evaluate_phase7.py         # Phase 7 exit checklist (production gate)
│
├── tests/
│   ├── conftest.py
│   ├── test_lane_detection.py
│   ├── test_decision_engine.py
│   ├── test_safety_gate.py
│   ├── test_scene_understanding.py
│   ├── test_api.py
│   └── test_integration.py            # End-to-end integration tests
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/roadsage.json
│
├── configs/
│   ├── production.yaml
│   ├── development.yaml
│   ├── lane_detection.yaml
│   ├── scene_understanding.yaml
│   └── decision_engine.yaml
│
├── models/                            # ONNX weights (download separately)
├── rgb/                               # MNNIT campus images (1,302 frames)
├── outputs/                           # Inference visualizations
├── docker-compose.yml
├── Makefile
├── pyproject.toml                     # ruff config
├── requirements.txt
├── requirements-edge.txt              # Inference-only for Pi/Jetson
└── DEMO_SCRIPT.md
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

Result on the MNNIT dataset: **1,302 verified images** from 1,403 originals (98 duplicates removed).

### 2. Lane Detection

UFLD v2 reformulates lane detection as a **row-anchor classification** problem. For each horizontal row of the image, it predicts which column the lane passes through — dramatically faster than dense segmentation and handles curves naturally.

After detection, an Inverse Perspective Mapping (BEV) transform converts the perspective view to top-down, where polynomial fitting yields:

- **Lane offset** (meters from center; positive = right drift)
- **Road curvature** (1/radius in m⁻¹; positive = right curve)
- **Vanishing point** (where lanes converge ahead)

BEV calibration uses real MNNIT road geometry:
```python
src_points      = [[65, 455], [555, 440], [380, 285], [195, 285]]
output_size     = (640, 480)
pixels_per_meter = 25.0
```

### 3. Scene Understanding

NanoDet-Plus-m detects obstacles in the lane corridor (center 40% of frame width). MiDaS v2.1 Small produces a relative inverse depth map. Fusion logic extracts the maximum inverse depth within each bounding box as a proxy for distance.

If `nearest_obstacle_depth > stop_threshold` → `immediate_hazard = True`.

### 4. Decision Engine

Priority chain:

```
Priority 1  Safety Gate       Hard obstacle stop — no exceptions
Priority 2  Geometric Logic   Both lanes detected → offset + curvature rule
Priority 3  Single-Lane Fall  One lane detected → steer toward center
Priority 4  ML Fallback CNN   No lanes → MobileNetV3-Small prediction
Priority 5  Confidence Gate   STOP if final_confidence < 0.60
```

Geometric command logic:

```
|offset| > 0.6 m (strong)    → LEFT or RIGHT  (conf 0.95)
|offset| > 0.3 m (moderate)  → LEFT or RIGHT  (conf ~0.85)
|curve|  > 0.012 m⁻¹         → RIGHT or LEFT  (conf 0.90)
|curve|  > 0.005 m⁻¹         → RIGHT or LEFT  (conf ~0.82)
otherwise                    → FORWARD
```

### 5. Confidence Fusion

```
final_confidence =
    0.40 × lane_detection_confidence
  + 0.35 × geometric_signal_strength
  + 0.25 × ml_softmax_max_value
```

If `final_confidence < 0.60` → override to `STOP`.

### 6. Explainability

Every prediction generates:
- **GradCAM overlay** — highlights which image regions drove the decision
- **Lane visualization** — color-coded by confidence (green ≥ 0.85, yellow 0.65–0.85, red < 0.65)
- **Temporal buffer trace** — last 5 commands for smoothing context

---

## Model Choices

All models run CPU-only via ONNX Runtime. GPU equivalents are defined and can be activated via config.

| Layer | CPU Model (active) | CPU Latency |
|---|---|---|
| Lane detection | UFLD v2 ResNet-18 + ONNX | ~28 ms |
| Object detection | NanoDet-Plus-m + ONNX | ~10 ms |
| Depth estimation | MiDaS v2.1 Small + ONNX | ~28 ms |
| Fallback CNN | MobileNetV3-Small + ONNX | ~4 ms |
| Decision engine | Pure Python rules | ~2 ms |
| **Total** | | **< 74 ms** |

---

## API Reference

### `POST /api/v1/predict`

```bash
curl -X POST http://localhost:8000/api/v1/predict \
     -F "file=@road_image.jpg"
```

Response:
```json
{
  "command":                "FORWARD",
  "confidence":             0.91,
  "decision_path":          "geometric",
  "lane_offset_m":          0.12,
  "curvature_inv_m":        0.002,
  "left_lane_detected":     true,
  "right_lane_detected":    true,
  "hazard_detected":        false,
  "hazard_reason":          null,
  "surface_class":          "clean",
  "nearest_obstacle_class": null,
  "latency_ms": {
    "lane": 28.1, "scene": 9.4, "decision": 1.8, "total": 39.3
  },
  "frame_id":               42,
  "timestamp":              "2025-06-04T10:30:00Z"
}
```

### `GET /api/v1/health`

Returns model load status, latency percentiles (P50/P95), and frame counter.

### `POST /api/v1/batch`

Accepts a directory path or list of images. Returns a full prediction for each frame.

### `WS /ws/live`

Push binary image frames; receive JSON predictions at up to 15 FPS output.  
Rate limits: `/predict` — 30 req/min per IP; `/batch` — 5 req/min per IP.

---

## Testing

```bash
make test               # all tests
make test-integration   # end-to-end tests against real engine

# Individual suites
pytest tests/test_lane_detection.py     # BEV transform, geometry, UFLD wrapper
pytest tests/test_decision_engine.py    # all 5 priority-chain branches
pytest tests/test_safety_gate.py        # safety invariants (STOP never delayed)
pytest tests/test_api.py                # all endpoints via httpx TestClient
pytest tests/test_integration.py        # full pipeline end-to-end
```

Tests are pure unit tests — no model weights required. The decision engine modules are fully functional (no I/O, no side effects), so every branch of the priority chain is testable with synthetic inputs.

---

## Monitoring

Prometheus metrics exposed at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `roadsage_inference_latency_seconds` | Histogram | Per-component latency (lane/scene/decision/total) |
| `roadsage_request_latency_seconds` | Histogram | End-to-end HTTP request latency |
| `roadsage_command_total` | Counter | Commands issued, by type |
| `roadsage_confidence_histogram` | Histogram | Distribution of fused confidence scores |
| `roadsage_safety_gate_triggers_total` | Counter | Safety gate activations |
| `roadsage_lane_detection_failures_total` | Counter | Frames with no lanes detected |
| `roadsage_ml_fallback_activations_total` | Counter | ML fallback usage rate |

Import the pre-built Grafana dashboard after `docker-compose up`:

```
Grafana → Dashboards → Import → Upload monitoring/grafana/dashboards/roadsage.json
```

Populate it with live traffic:
```bash
make stress-test          # 2-minute API load test
# or:
python training/scripts/metrics_stress_test.py --duration-minutes 5
```

---

## Evaluation

```bash
# Lane detector proxy metrics (detection rate, confidence, latency)
python training/evaluation/evaluate_lane.py --source rgb

# Decision pipeline (auto mode — no GT needed)
python training/evaluation/evaluate_decision.py --source rgb --benchmark

# With ground truth annotations
python training/evaluation/evaluate_decision.py --source rgb --gt data/mnnit/gt.json

# Create annotation template (100 images)
python training/evaluation/evaluate_decision.py --create-gt-template

# 5-variant component ablation study
python training/scripts/run_ablation.py --source rgb --n-images 50

# Phase 7 production-readiness gate
python training/evaluation/evaluate_phase7.py
```

### Production Targets

| Metric | Target |
|---|---|
| Command accuracy (with GT) | ≥ 88% |
| STOP precision | ≥ 99% |
| P95 inference latency | < 100 ms (with ONNX) |
| Reliability | 0 crashes over 500-frame run |
| Lane detection rate (both lanes) | ≥ 40% on MNNIT test split |

---

## Training

### Self-Supervised Label Generation

RoadSage uses no manual command labels. Pseudo-labels are derived from geometry:

```
1. Pretrain UFLD v2 on TuSimple + CULane (public labeled data)
2. Run on MNNIT images → accept where lane_confidence > 0.85
3. Derive commands via geometric logic → pseudo-labels
4. Fine-tune lane detector + decision CNN on pseudo-labeled set
5. Repeat until coverage > 80%
```

```bash
# Generate pseudo-labels
python training/scripts/generate_pseudo_labels.py \
    --input data/mnnit/rgb/ --output data/mnnit/pseudo_labels/ --min-confidence 0.85

# Train decision CNN
python training/trainers/train_decision.py --epochs 50

# Export all models to ONNX
python training/scripts/export_onnx.py --all
```

---

## Edge Deployment

All models are exported to ONNX and run via ONNX Runtime — PyTorch and training dependencies are not needed at inference time.

```bash
# Build inference-only Docker image (~1.2 GB vs ~8 GB full image)
docker build -f Dockerfile.edge -t roadsage-edge .
```

Target: P95 < 100 ms end-to-end on Raspberry Pi 4 / Jetson Nano.

---

## Configuration

All runtime thresholds live in `configs/` — nothing is hardcoded in source.

```yaml
# configs/decision_engine.yaml (key thresholds)
geometric:
  offset_threshold:        0.3    # m — lateral drift triggers correction
  curve_threshold:         0.005  # m⁻¹ — curvature triggers turn
  strong_offset_threshold: 0.6
  strong_curve_threshold:  0.012

safety:
  min_confidence:          0.60   # below this → force STOP
  temporal_consistency_frames: 3

confidence_fusion:
  weight_lane:             0.40
  weight_geometric:        0.35
  weight_ml:               0.25
```

---

## Development Status

| Phase | Status | Deliverables |
|---|---|---|
| **Phase 1** — Foundation & Data Pipeline | ✅ Complete | 1,302 verified images, 4-gate quality filter, augmentation pipeline, Docker stack |
| **Phase 2** — BEV & Lane Detection | ✅ Complete | BEV calibrated on real MNNIT frames, UFLD v2 wrapper, lane geometry computer |
| **Phase 3** — Scene Understanding | ✅ Complete | NanoDet, MiDaS, surface classifier, SceneAnalyzer fusion; 34/34 tests pass |
| **Phase 4** — Decision Engine | ✅ Complete | Full 5-layer priority chain, safety gate, confidence fusion, ML fallback |
| **Phase 5** — System Integration & API | ✅ Complete | RoadSageEngine, GradCAM, FastAPI endpoints, WebSocket streaming |
| **Phase 6** — Dashboard & Monitoring | ✅ Complete | React dashboard (7 components), Grafana dashboard, Prometheus metrics |
| **Phase 7** — Evaluation & Production Gate | ✅ Complete | Evaluation scripts, ablation study, integration tests, Phase 7 validator |

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
