# RoadSage — Vision-Based Lane Understanding & Intelligent Driving Decision Engine

> Production-grade autonomous navigation system for campus roads (MNNIT Allahabad).  
> Predicts driving decisions from unlabeled road images using a hybrid geometric + ML pipeline.

---

## What It Does

Given a single forward-facing RGB road image, RoadSage outputs:

| Field | Type | Description |
|---|---|---|
| `command` | `FORWARD \| LEFT \| RIGHT \| STOP` | Driving decision |
| `confidence` | `float 0–1` | Decision certainty |
| `lane_offset` | `float (m)` | Lateral distance from lane center (negative = left) |
| `curvature` | `float (m⁻¹)` | Road curve severity |
| `hazard_flag` | `bool` | Immediate obstacle detected |
| `explanation` | `path` | GradCAM overlay image showing decision rationale |

---

## System Architecture

```
[Raw Image]
     │
     ▼
Preprocessing & Augmentation   ← CLAHE, denoise, perspective warp
     │
     ├────────────────────────────────────┐
     ▼                                    ▼
Lane Detection Engine          Scene Understanding
(UltraFast Lane Det v2)        (YOLOv8n + MiDaS depth)
     │                                    │
     └──────────────┬─────────────────────┘
                    ▼
         Geometric Analysis & Feature Fusion
         (lane offset, curvature, vanishing point)
                    │
                    ▼
           Decision Engine (Hybrid)
           ┌── Geometric rules (primary)
           └── ML fallback (MobileNetV3-Small)
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    Safety & Confidence    Explainability
    Gate (MC Dropout)      (GradCAM + lane viz)
          │
          ▼
    FastAPI REST + WebSocket
          │
          ▼
    React Real-Time Dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Lane Detection | UltraFast Lane Detection v2 (ResNet-50 backbone) |
| Object Detection | YOLOv8n |
| Depth Estimation | MiDaS v2.1 Small |
| ML Framework | PyTorch 2.x |
| Image Augmentation | albumentations |
| Geometric Processing | OpenCV 4.x |
| API Server | FastAPI + Uvicorn |
| Edge Inference | ONNX Runtime + TensorRT |
| Dashboard | React 18, Recharts, lucide-react, Vite |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + docker-compose |

---

## Project Structure

```
roadsage/
├── app/                        # Core Python package
│   ├── engine.py               # RoadSageEngine — main entry point
│   ├── preprocessing/          # CLAHE, augmentation, perspective warp
│   ├── lane_detection/         # UFLD v2 model, BEV transform, geometry
│   ├── scene_understanding/    # YOLOv8n, MiDaS, surface classifier
│   ├── decision/               # Geometric logic, ML fallback, safety gate, confidence fusion
│   ├── explainability/         # GradCAM, lane visualizer
│   └── utils/                  # Kalman filter, logger, metrics
│
├── api/                        # FastAPI service
│   ├── main.py
│   ├── routes/                 # /predict, /health, /batch
│   ├── websocket/              # Live streaming endpoint
│   └── middleware/             # Logging, rate limiting
│
├── dashboard/                  # React + Vite frontend
│   └── src/components/         # VideoFeed, DecisionPanel, LaneMetrics,
│                               # GradCamView, ConfidenceMeter, DecisionHistory, SystemHealth
│
├── training/
│   ├── trainers/               # train_lane.py, train_decision.py
│   ├── scripts/                # Pseudo-label generation, dataset filtering, ONNX export
│   └── evaluation/             # Lane + decision metrics, HTML report
│
├── configs/                    # YAML configs (production, development, lane, decision, augmentation)
├── data/                       # MNNIT raw/pseudo/verified + external datasets (TuSimple, CULane)
├── models/                     # Model weights (download via models/download_models.sh)
├── notebooks/                  # 5 Jupyter notebooks for exploration, demo, and evaluation
├── monitoring/                 # Prometheus config + Grafana dashboards
└── tests/                      # pytest suite (lane detection, decision engine, safety gate, API)
```

---

## Quickstart

### Prerequisites

- Docker & docker-compose
- Python 3.10+
- Node.js 18+ / pnpm (for local dashboard dev only)

### Run Everything (Docker)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Dashboard | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

### Run API Locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Run Dashboard Locally

```bash
cd dashboard
pnpm install
pnpm dev          # http://localhost:5173
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Single image → driving command + explanation |
| `GET` | `/api/v1/health` | System status, model load state, avg latency |
| `POST` | `/api/v1/batch` | Batch image predictions with full trace |
| `WS` | `/ws/live` | Real-time prediction stream |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "image=@frame.jpg"
```

**Example response:**

```json
{
  "command": "FORWARD",
  "confidence": 0.91,
  "lane_offset": 0.12,
  "curvature": 0.002,
  "hazard": false,
  "gradcam_path": "outputs/gradcam_042.jpg",
  "lane_viz_path": "outputs/lane_042.jpg",
  "latency_ms": 47.3
}
```

---

## Decision Logic

The engine follows a deterministic priority chain:

1. **Safety Gate** — hard STOP if obstacle < 2.0 m or any hazard flag
2. **Geometric Decision** — uses lane offset + curvature (primary path)
   - `|offset| > 0.3 m` → correct left/right
   - `|curvature| > 0.005 m⁻¹` → turn command
   - Otherwise → FORWARD
3. **Single-Lane Fallback** — one lane missing → steer toward center
4. **ML Fallback** — no lanes detected → MobileNetV3-Small classifier
5. **Confidence Gate** — `confidence < 0.60` → STOP (fail-safe)

Key thresholds are tunable in `configs/decision_engine.yaml`.

---

## Training Pipeline

```bash
# 1. Pretrain lane detector on public data (TuSimple + CULane)
python training/trainers/train_lane.py \
    --config configs/lane_detection.yaml \
    --dataset tusimple+culane --epochs 100

# 2. Generate pseudo-labels for MNNIT images
python training/scripts/generate_pseudo_labels.py \
    --model checkpoints/lane_best.pth \
    --input data/mnnit/raw/ \
    --output data/mnnit/pseudo_labels/ \
    --min_confidence 0.85

# 3. Fine-tune on MNNIT pseudo-labeled data
python training/trainers/train_lane.py \
    --config configs/lane_detection.yaml \
    --dataset mnnit_pseudo --epochs 30 --lr 1e-4 \
    --resume checkpoints/lane_best.pth

# 4. Train decision CNN on geometric pseudo-labels
python training/trainers/train_decision.py \
    --config configs/decision_engine.yaml \
    --dataset mnnit_with_commands --epochs 50

# 5. Export all models to ONNX
python training/scripts/export_onnx.py --all
```

### Download Pretrained Weights

```bash
bash models/download_models.sh
```

---

## Data Strategy

Since MNNIT campus images are **unlabeled**, the system uses a self-supervised pseudo-labeling loop:

1. Pretrain on public datasets (TuSimple, CULane)
2. Run inference on MNNIT images; keep predictions where `lane_confidence > 0.85`
3. Use geometric decision logic to auto-generate driving commands as pseudo-labels
4. Fine-tune on pseudo-labeled MNNIT data
5. Repeat for 2–3 iterations until ~85% of data is labeled
6. Human spot-check on 10% sample before final training

Data quality filters applied at ingestion: blur detection (Laplacian variance), brightness bounds, road pixel ratio, perceptual hash deduplication.

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Lane F1-Score | > 0.85 |
| Row-anchor accuracy | > 94% |
| Command accuracy (manual eval) | > 88% |
| STOP precision (safety-critical) | > 99% |
| End-to-end latency | < 100 ms |
| Uncertainty calibration (ECE) | < 0.05 |

Run evaluation:

```bash
python training/evaluation/evaluate_lane.py
python training/evaluation/evaluate_decision.py
python training/evaluation/generate_report.py   # outputs HTML report
```

---

## Tests

```bash
pytest tests/ -v
```

Test files:

| File | Coverage |
|---|---|
| `tests/test_lane_detection.py` | UFLD v2 inference, BEV transform, geometry |
| `tests/test_decision_engine.py` | Geometric logic, ML fallback, confidence fusion |
| `tests/test_safety_gate.py` | Hard stop, confidence gate, temporal consistency |
| `tests/test_api.py` | `/predict`, `/health`, `/batch` endpoints |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | MNNIT dataset EDA, quality filtering |
| `02_lane_detection_demo.ipynb` | UFLD v2 inference walkthrough |
| `03_pseudo_labeling.ipynb` | Self-training pipeline demo |
| `04_decision_logic_analysis.ipynb` | Decision engine behavior analysis |
| `05_model_evaluation.ipynb` | Final metrics + ablation study |

---

## Configuration

All tuneable parameters live in `configs/`:

| File | Controls |
|---|---|
| `production.yaml` | Model paths, latency targets, safety thresholds |
| `development.yaml` | Debug flags, visualization toggles |
| `lane_detection.yaml` | Backbone, anchor count, confidence threshold |
| `decision_engine.yaml` | Offset/curvature thresholds, MC Dropout samples |
| `augmentation.yaml` | Augmentation probabilities and parameters |

---

## Monitoring

Prometheus scrapes metrics from the API at `/metrics`. Import `monitoring/grafana/dashboards/roadsage.json` into Grafana to view:

- Inference latency (p50/p95/p99)
- Throughput (requests/sec)
- Confidence score distribution
- Command frequency breakdown
- Safety gate trigger rate

---

## Edge Deployment

For Raspberry Pi 4 / Jetson Nano:

```bash
pip install -r requirements-edge.txt    # lightweight deps
python training/scripts/export_onnx.py --all
# Run with ONNX Runtime — target < 100ms on edge
```

---

## Future Scope

- GPS + waypoint fusion for full campus route navigation
- SLAM-based HD map of MNNIT roads
- Night mode (low-light specialized model)
- Pedestrian trajectory prediction
- Multi-camera 360° awareness
- Federated learning across multiple vehicles

---

## Documentation

Full technical design, architecture decisions, data strategy, and module deep-dives: [`docs/road_sage.md`](docs/road_sage.md)
