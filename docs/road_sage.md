# 🛣️ RoadSage — Vision-Based Lane Understanding & Intelligent Driving Decision Engine
### Production-Grade Autonomous Navigation System for Campus Roads (MNNIT Allahabad)

> **Hackathon Edition** | Authored by: [Your Team Name]  
> **Target Environment:** MNNIT Campus Roads | **Task:** Predict driving decisions from unlabeled road images

---

## 📌 Table of Contents

1. [Project Vision & Scope](#1-project-vision--scope)
2. [Problem Statement (Formal)](#2-problem-statement-formal)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Module Breakdown](#4-module-breakdown)
   - 4.1 Data Pipeline
   - 4.2 Lane Detection Engine
   - 4.3 Scene Understanding Module
   - 4.4 Decision Engine
   - 4.5 Confidence & Safety Layer
   - 4.6 Explainability Layer
   - 4.7 API & Deployment Layer
   - 4.8 Real-Time Dashboard
5. [Technology Stack & Choices](#5-technology-stack--choices)
6. [Model Architecture Deep Dive](#6-model-architecture-deep-dive)
7. [Data Strategy (No Labels? No Problem)](#7-data-strategy-no-labels-no-problem)
8. [Training Pipeline](#8-training-pipeline)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Decision Logic: How Driving Commands Are Derived](#10-decision-logic-how-driving-commands-are-derived)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Advanced Features That Win Hackathons](#13-advanced-features-that-win-hackathons)
14. [Project File Structure](#14-project-file-structure)
15. [Roadmap & Milestones](#15-roadmap--milestones)
16. [Risk Analysis & Mitigation](#16-risk-analysis--mitigation)
17. [Engineering Decisions Log](#17-engineering-decisions-log)
18. [Future Scope (Post-Hackathon)](#18-future-scope-post-hackathon)

---

## 1. Project Vision & Scope

**RoadSage** is not just a lane detection system — it is an **end-to-end intelligent driving decision engine** that ingests raw, unlabeled road images from MNNIT campus and produces:

- **Driving Command:** `FORWARD | LEFT | RIGHT | STOP`
- **Lane Geometry:** Detected lane boundaries with curvature & offset
- **Scene Context:** Road surface quality, obstacles, road width estimate
- **Confidence Score:** Per-decision uncertainty quantification
- **Explainability Map:** Visual saliency showing *why* the model decided what it decided

The system is designed to be:
- **Zero-label at inference** — works directly on raw images
- **Edge-deployable** — runs on a Raspberry Pi 4 / Jetson Nano
- **Production-hardened** — with monitoring, fallback logic, and safety gating
- **Explainable** — not a black box; every decision is visually justified

---

## 2. Problem Statement (Formal)

### Input
- A single RGB road image captured from a forward-facing camera mounted on a vehicle navigating MNNIT campus roads.
- Images are **unlabeled** — no ground truth steering angle, no semantic segmentation masks.

### Output
```
{
  "command":     "FORWARD" | "LEFT" | "RIGHT" | "STOP",
  "confidence":  float (0.0 - 1.0),
  "lane_offset": float (meters, negative=left, positive=right of center),
  "curvature":   float (1/radius in m⁻¹),
  "hazard_flag": bool,
  "explanation": "path/to/grad_cam_overlay.jpg"
}
```

### Why This Is Hard
1. **No labels** → Cannot do supervised classification directly
2. **Campus-specific** → General autonomous driving datasets (KITTI, CityScapes) don't transfer well to narrow Indian campus roads with trees, bollards, and mixed traffic
3. **Variable lighting** → Morning haze (as seen in sample image), afternoon glare, shadows from dense trees
4. **Non-standard markings** → MNNIT roads have yellow-black kerb markings, white dashed center lines — different from highway conventions

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RoadSage System                                │
│                                                                         │
│  [Raw Image]                                                            │
│      │                                                                  │
│      ▼                                                                  │
│  ┌──────────────────┐                                                   │
│  │  Preprocessing   │  ← Denoise, CLAHE, Perspective Warp              │
│  │  & Augmentation  │                                                   │
│  └────────┬─────────┘                                                   │
│           │                                                             │
│      ┌────┴────────────────────────┐                                   │
│      ▼                             ▼                                    │
│  ┌──────────────┐         ┌────────────────────┐                       │
│  │ Lane         │         │ Scene Understanding│                        │
│  │ Detection    │         │ (Obstacle/Context) │                        │
│  │ Engine       │         │                    │                        │
│  │ (UltraFast   │         │ (YOLOv8 + DepthEst)│                       │
│  │  Lane Det v2)│         └────────┬───────────┘                       │
│  └──────┬───────┘                  │                                   │
│         │                          │                                   │
│         └──────────┬───────────────┘                                   │
│                    ▼                                                    │
│          ┌─────────────────────┐                                        │
│          │  Geometric Analysis │  ← Lane offset, curvature,            │
│          │  & Feature Fusion   │    vanishing point, road width         │
│          └──────────┬──────────┘                                        │
│                     │                                                   │
│                     ▼                                                   │
│          ┌─────────────────────┐                                        │
│          │  Decision Engine    │  ← Rule-based + ML hybrid             │
│          │  (Hybrid Logic)     │                                        │
│          └──────────┬──────────┘                                        │
│                     │                                                   │
│          ┌──────────┴──────────┐                                        │
│          ▼                     ▼                                        │
│  ┌──────────────┐    ┌─────────────────────┐                           │
│  │ Safety &     │    │ Explainability       │                           │
│  │ Confidence   │    │ (GradCAM + Lane Viz) │                           │
│  │ Gate         │    └─────────────────────┘                           │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────┐                      │
│  │         FastAPI REST Endpoint / Edge Runtime  │                      │
│  └──────────────────────────────────────────────┘                      │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────┐                      │
│  │    Real-Time Dashboard (React + WebSocket)   │                      │
│  └──────────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Breakdown

---

### 4.1 Data Pipeline

#### 4.1.1 Data Collection Strategy
Since images are **unlabeled**, we use a multi-source data strategy:

**Source A — MNNIT Campus Images (Primary)**
- Collect images by driving/walking the campus roads at different times of day
- Cover: main road, internal roads, roundabouts, T-junctions, tree-shaded zones
- Target: ~500–800 images minimum; more = better

**Source B — Public Dataset Transfer Learning**
- `TuSimple` dataset (highway lane detection) — used for model pretraining
- `CULane` dataset — diverse road types, includes curves and intersections
- `BDD100K` — includes Indian road-like scenarios
- These will NOT be used as final training data; only for **feature extractor pretraining**

**Source C — Synthetic Data (Augmentation Engine)**
- Use **CARLA Simulator** or **Blender** to generate synthetic campus-like road images
- Controllable lighting, time-of-day, road width, tree density
- Auto-generates ground truth labels for supervised pretraining

#### 4.1.2 Pseudo-Labeling Strategy (The Core Innovation)
Since our target data is unlabeled, we use a **self-supervised + pseudo-label pipeline:**

```
Step 1: Pretrain lane detector on TuSimple/CULane
Step 2: Run inference on MNNIT images → get lane predictions
Step 3: Filter high-confidence predictions (confidence > 0.85)
Step 4: Use these as pseudo-labels for fine-tuning
Step 5: Human-in-the-loop validation on 10% samples
Step 6: Iterate 2–3 cycles (self-training loop)
```

This is the same technique used by Tesla's Autopilot team for scaling to new road types.

#### 4.1.3 Augmentation Suite
Every image goes through a stochastic augmentation pipeline:

| Augmentation | Purpose |
|---|---|
| CLAHE (Contrast Limited Adaptive Histogram Equalization) | Fix morning haze / low contrast |
| Random brightness ±30% | Handle time-of-day variation |
| Gaussian blur + sharpen | Simulate camera shake |
| Horizontal flip (with lane label mirroring) | Double dataset size |
| Random shadow overlay | Simulate tree shadows on road |
| Perspective warp ±10° | Simulate camera mount angle variation |
| Salt & pepper noise | Simulate dust on lens |
| Random crop + resize | Simulate distance variation |
| Rain/fog simulation (albumentations) | Edge case robustness |

**Library:** `albumentations` — industry standard, 10x faster than torchvision transforms

---

### 4.2 Lane Detection Engine

#### Choice: UltraFast Lane Detection v2 (UFLD v2)

**Why not classical (Canny + Hough)?**
- Fails on curved roads, faded markings, shadows — all common on MNNIT roads
- Not robust to the yellow-black bollards being confused with lane lines

**Why not SegNet/DeepLab semantic segmentation?**
- Heavy, slow (>100ms on CPU)
- Overkill for lane-only task

**Why UFLD v2?**
- Treats lanes as a **row-anchor classification** problem — extremely fast
- 322 FPS on GPU, ~45 FPS on Jetson Nano
- Works well on curved lanes
- Backbone: ResNet-34 (light) or ResNet-101 (accurate) — we choose **ResNet-50 as sweet spot**
- State-of-the-art on TuSimple (96.06% accuracy) and CULane

**Lane Detection Outputs:**
```python
{
  "left_lane":   [(x1,y1), (x2,y2), ...],  # pixel coordinates
  "right_lane":  [(x1,y1), (x2,y2), ...],
  "center_lane": [(x1,y1), ...],             # if present
  "confidence":  [0.94, 0.91, 0.78]
}
```

#### Bird's Eye View (BEV) Transform
After detection, we apply an **Inverse Perspective Mapping (IPM):**
- Warp detected lanes to top-down view
- Fit polynomial (2nd degree) to each lane: `x = ay² + by + c`
- Compute:
  - **Lane offset** from center: how far left/right the vehicle is
  - **Road curvature** (1/radius): determines turn severity
  - **Vanishing point**: where lanes converge → tells if road curves ahead

```python
# Curvature formula at bottom of image
curvature = ((1 + (2*A*y_eval + B)**2)**1.5) / abs(2*A)
offset = (left_lane_x + right_lane_x) / 2 - image_center_x
```

---

### 4.3 Scene Understanding Module

Beyond lanes, the vehicle needs **spatial awareness.**

#### 4.3.1 Object Detection — YOLOv8n (Nano)
- Detects: persons, vehicles, cycles, auto-rickshaws, animals
- YOLOv8n chosen for: **<5ms inference**, runs on edge
- Fine-tuned on campus-specific classes (cyclist, pedestrian, cycle rickshaw)
- Output: bounding boxes + class + distance estimate

#### 4.3.2 Monocular Depth Estimation — MiDaS v2.1 Small
- Generates a relative depth map from a single image
- Used to estimate **how far ahead an obstacle is**
- Does NOT require stereo camera
- MiDaS Small: 30ms inference on CPU

**Fusion Logic:**
```
If YOLO detects obstacle AND MiDaS shows depth < threshold:
    → Flag as "immediate hazard"
    → Override decision to STOP
```

#### 4.3.3 Road Surface Quality Estimation
- Detect potholes, speed breakers, debris using a lightweight CNN classifier
- Classes: `clean | pothole | speed_breaker | waterlogged`
- Trained on ~200 MNNIT campus road patches + augmentation
- Affects: speed recommendation (future work), generates hazard alerts

---

### 4.4 Decision Engine

This is the **brain** of RoadSage. It is a **hybrid system** — not pure ML, not pure rules. Both.

#### Why Hybrid?
- Pure ML: black box, unpredictable edge cases, dangerous
- Pure rules: too brittle, can't handle all scenarios
- **Hybrid:** rules handle safety-critical cases; ML handles nuanced scenes

#### Decision Logic Flow

```
INPUT: lane_offset, curvature, vanishing_point_x, obstacle_detected, obstacle_distance, road_surface

STEP 1 — SAFETY GATE (Hard Rules)
  IF obstacle_distance < 2.0m OR hazard_flag == True:
      → STOP (with 100% confidence override)

STEP 2 — LANE-BASED GEOMETRIC DECISION
  IF both lanes detected:
      IF abs(offset) > OFFSET_THRESHOLD:
          → LEFT if offset > 0 (vehicle drifted right)
          → RIGHT if offset < 0 (vehicle drifted left)
      IF curvature > CURVE_THRESHOLD:
          → LEFT if curve bends left
          → RIGHT if curve bends right
      ELSE:
          → FORWARD

STEP 3 — SINGLE LANE / EDGE CASE
  IF only right lane detected:
      → Steer LEFT to return to center
  IF only left lane detected:
      → Steer RIGHT to return to center
  IF no lanes detected:
      → Activate ML fallback classifier

STEP 4 — ML FALLBACK CLASSIFIER
  A lightweight CNN (MobileNetV3-Small) trained on:
  - Visual features of MNNIT roads
  - Pseudo-labeled with geometric decision logic (Step 2 as teacher)
  
  Outputs: softmax over [FORWARD, LEFT, RIGHT, STOP]
  Used ONLY when geometric approach fails

STEP 5 — CONFIDENCE FUSION
  Final confidence = weighted combination of:
  - Lane detection confidence
  - Geometric decision clarity (how strong the signal is)
  - ML classifier softmax max value
  
  IF final_confidence < 0.60:
      → STOP (uncertainty-based safety)
```

#### Thresholds (Tunable via config.yaml)

| Parameter | Default Value | Description |
|---|---|---|
| `OFFSET_THRESHOLD` | 0.3m | Lane offset to trigger correction |
| `CURVE_THRESHOLD` | 0.005 m⁻¹ | Curvature to trigger turn command |
| `OBSTACLE_STOP_DIST` | 2.0m | Distance to trigger hard stop |
| `MIN_CONFIDENCE` | 0.60 | Below this → STOP |
| `LANE_CONF_THRESHOLD` | 0.75 | Minimum lane confidence to trust |

---

### 4.5 Confidence & Safety Layer

Every output carries **uncertainty quantification** using:

#### Monte Carlo Dropout (MC Dropout)
- At inference, keep dropout active
- Run the model **N=10 times** on the same image
- Compute mean prediction + variance
- High variance = model is uncertain → flag for safety

```python
def predict_with_uncertainty(model, image, n_samples=10):
    model.train()  # keep dropout active
    predictions = [model(image) for _ in range(n_samples)]
    mean = torch.stack(predictions).mean(0)
    std  = torch.stack(predictions).std(0)
    return mean, std
```

#### Safety Guarantees
1. **Confidence Gate:** If confidence < threshold → STOP (fail-safe)
2. **Temporal Consistency:** Command must persist for ≥3 consecutive frames before execution (prevents flickering)
3. **Command Smoothing:** Moving average over last 5 predictions for stability
4. **Hard Stop Override:** Any hazard detection → immediate STOP regardless of confidence

---

### 4.6 Explainability Layer

This is what separates RoadSage from every other team's project.

#### GradCAM Overlay
- For every decision, generate a **Grad-CAM heatmap** on the input image
- Shows: which pixels in the image caused the model to make that decision
- Tells the judge: "The model looked at the lane markings here, and the curve ahead"

#### Lane Visualization
- Draw detected lane lines on original image (green = confident, yellow = uncertain, red = missing)
- Show BEV (bird's eye view) lane map
- Annotate: offset value, curvature, decision command

#### Decision Trace Log
```json
{
  "frame_id": 42,
  "timestamp": "2024-03-15T10:23:41.112Z",
  "raw_image": "frame_042.jpg",
  "lane_offset_m": 0.18,
  "curvature_inv_m": 0.003,
  "lanes_detected": ["left", "right"],
  "obstacle": false,
  "decision_path": "geometric",
  "command": "FORWARD",
  "confidence": 0.91,
  "gradcam_overlay": "gradcam_042.jpg",
  "lane_viz": "lane_042.jpg"
}
```

---

### 4.7 API & Deployment Layer

#### FastAPI Backend

```
POST /api/v1/predict
  Input:  multipart/form-data image
  Output: JSON with command + confidence + explanation image URL

GET  /api/v1/health
  Output: system status, model load state, avg inference time

WS   /ws/live
  WebSocket: stream predictions in real-time (for demo)

POST /api/v1/batch
  Input:  list of image paths
  Output: batch predictions with full trace
```

#### Docker Deployment
```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Edge Runtime (for live demo)
- ONNX export of all models (lane detector, depth estimator, decision CNN)
- Runtime: ONNX Runtime with TensorRT optimization on Jetson Nano
- Target: **< 100ms end-to-end latency** on edge hardware

---

### 4.8 Real-Time Dashboard

A **React + TailwindCSS** frontend with:

- **Live Camera Feed** with lane overlay drawn in real-time
- **Decision Panel:** Big command display (FORWARD/LEFT/RIGHT/STOP) with color coding
  - 🟢 FORWARD, 🟡 LEFT, 🔵 RIGHT, 🔴 STOP
- **Confidence Meter:** Real-time gauge
- **Lane Metrics Panel:** Offset, curvature, detected lanes
- **GradCAM View:** Toggle to see what the model "sees"
- **Decision History Timeline:** Scrolling log of last 50 decisions
- **System Health:** FPS counter, model latency, GPU/CPU usage
- **Map View (Bonus):** GPS overlay on MNNIT campus map showing current vehicle position

---

## 5. Technology Stack & Choices

| Layer | Technology | Why |
|---|---|---|
| **Lane Detection** | UltraFast Lane Det v2 | Best speed-accuracy tradeoff; row-anchor approach handles curves |
| **Object Detection** | YOLOv8n | Fastest YOLO variant; good on campus-scale objects |
| **Depth Estimation** | MiDaS v2.1 Small | Monocular depth without stereo; lightweight |
| **ML Framework** | PyTorch 2.x | Flexible, best ecosystem for CV research |
| **Image Augmentation** | albumentations | 10x faster than torchvision transforms |
| **Geometric Processing** | OpenCV 4.x | Industry standard for IPM, lane polynomial fitting |
| **API Server** | FastAPI + Uvicorn | Async, fast, automatic OpenAPI docs |
| **Containerization** | Docker + docker-compose | Reproducible deployment |
| **Edge Inference** | ONNX Runtime + TensorRT | Model portability + GPU acceleration |
| **Dashboard Frontend** | React + TailwindCSS + Chart.js | Fast to build, visually clean |
| **Real-time Comms** | WebSocket (FastAPI) | Low latency for live demo |
| **Experiment Tracking** | MLflow | Track all training runs, metrics, model versions |
| **Config Management** | Hydra + OmegaConf | Clean config files, easy ablation |
| **Testing** | pytest + hypothesis | Unit + property-based testing |
| **CI/CD** | GitHub Actions | Auto test on every push |
| **Monitoring** | Prometheus + Grafana | Production inference monitoring |

---

## 6. Model Architecture Deep Dive

### Lane Detection: UFLD v2 Architecture

```
Input Image (800×288) 
    │
    ▼
[Backbone: ResNet-50 pretrained on ImageNet]
    │
    ├── Feature Pyramid Network (FPN) 
    │   → Multi-scale features: P2, P3, P4, P5
    │
    ▼
[Row Anchor Head]
    │ For each of 72 row anchors (uniformly spaced vertically)
    │ Predict: which x-grid-cell contains the lane point
    │ Output shape: [batch, num_lanes, num_row_anchors, num_grids+1]
    │ The +1 is for "no lane at this row"
    │
    ▼
[Lane Existence Head]
    │ Binary sigmoid: does this lane exist?
    │
    ▼
Post-Processing:
    → Convert grid predictions → pixel coordinates
    → Filter by existence confidence
    → Fit polynomial in BEV space
```

### Decision CNN: MobileNetV3-Small (Fallback)

```
Input: 224×224×3 image
    │
    ▼
MobileNetV3-Small backbone (pretrained ImageNet)
    │
    ▼
Global Average Pooling
    │
    ▼
Linear(576 → 128) + ReLU + Dropout(0.3)
    │
    ▼
Linear(128 → 4)  ← [FORWARD, LEFT, RIGHT, STOP]
    │
    ▼
Softmax → Confidence scores
```

**Trained using:** Pseudo-labels generated by the geometric decision engine on MNNIT images.

---

## 7. Data Strategy (No Labels? No Problem)

### The Self-Training Loop

```
Iteration 0:
  - Pretrain UFLD v2 on TuSimple + CULane (public labeled data)
  - Pretrain MobileNetV3 on synthetic CARLA data (auto-labeled)

Iteration 1:
  - Run pretrained models on MNNIT images
  - Keep predictions where lane_confidence > 0.85
  - Generate pseudo-labels for driving commands via geometric logic
  - Fine-tune both models on pseudo-labeled MNNIT data
  - ~60% MNNIT data gets pseudo-labels

Iteration 2:
  - Run Iter-1 models on remaining unlabeled MNNIT images
  - More images now exceed confidence threshold
  - ~85% data pseudo-labeled
  - Fine-tune again

Iteration 3:
  - Human review of 50 random samples (quality check)
  - Final fine-tune with verified + pseudo-labeled data
  - Lock models for production
```

### Data Quality Filters

Before any image enters training:
1. **Blur Detection:** Laplacian variance < 50 → reject (blurry image)
2. **Brightness Filter:** Mean pixel value < 30 or > 220 → reject (over/underexposed)
3. **Road Detection Pre-check:** Simple HSV-based road pixel ratio check; reject if < 20% road visible
4. **Duplicate Filter:** Perceptual hash (pHash) deduplication; cosine similarity > 0.98 → discard

---

## 8. Training Pipeline

### Directory Structure for Training

```
training/
├── configs/
│   ├── lane_detection.yaml
│   ├── decision_cnn.yaml
│   └── augmentation.yaml
├── datasets/
│   ├── tusimple/
│   ├── culane/
│   └── mnnit/
│       ├── raw/           ← original unlabeled images
│       ├── pseudo_labels/ ← generated labels
│       └── verified/      ← human-verified subset
├── models/
│   ├── lane_detector.py
│   ├── depth_estimator.py
│   └── decision_cnn.py
├── trainers/
│   ├── train_lane.py
│   └── train_decision.py
├── evaluation/
│   ├── metrics.py
│   └── visualize_predictions.py
└── scripts/
    ├── generate_pseudo_labels.py
    ├── filter_dataset.py
    └── export_onnx.py
```

### Training Commands

```bash
# Step 1: Pretrain lane detector on public data
python trainers/train_lane.py \
    --config configs/lane_detection.yaml \
    --dataset tusimple+culane \
    --epochs 100 \
    --backbone resnet50

# Step 2: Generate pseudo-labels for MNNIT data
python scripts/generate_pseudo_labels.py \
    --model checkpoints/lane_best.pth \
    --input data/mnnit/raw/ \
    --output data/mnnit/pseudo_labels/ \
    --min_confidence 0.85

# Step 3: Fine-tune on MNNIT
python trainers/train_lane.py \
    --config configs/lane_detection.yaml \
    --dataset mnnit_pseudo \
    --epochs 30 \
    --lr 1e-4 \
    --resume checkpoints/lane_best.pth

# Step 4: Train decision CNN on geometric pseudo-labels
python trainers/train_decision.py \
    --config configs/decision_cnn.yaml \
    --dataset mnnit_with_commands \
    --epochs 50

# Step 5: Export to ONNX
python scripts/export_onnx.py --all
```

### MLflow Tracking
- Every experiment auto-logged: hyperparams, loss curves, sample predictions
- Model registry: version-controlled model checkpoints
- Comparison dashboard: side-by-side experiment results

---

## 9. Inference Pipeline

### Single Image Inference

```python
from roadsage import RoadSageEngine

engine = RoadSageEngine(config="configs/production.yaml")

result = engine.predict("frame_042.jpg")

# result = {
#   "command": "FORWARD",
#   "confidence": 0.91,
#   "lane_offset": 0.12,
#   "curvature": 0.002,
#   "hazard": False,
#   "gradcam_path": "outputs/gradcam_042.jpg",
#   "lane_viz_path": "outputs/lane_042.jpg",
#   "latency_ms": 47.3
# }
```

### Batch Inference

```python
results = engine.predict_batch(
    image_dir="data/mnnit/test/",
    output_dir="outputs/",
    save_visualizations=True,
    num_workers=4
)
engine.generate_report(results, "outputs/evaluation_report.html")
```

### Streaming Inference (Live Camera)

```python
engine.start_stream(
    source=0,           # camera index or RTSP URL
    websocket_port=8765,
    target_fps=15
)
```

---

## 10. Decision Logic: How Driving Commands Are Derived

### Visual Illustration

```
         MNNIT Road — BEV Top-Down View

    Left  Center  Right
      │      │      │
      │      │      │
      │   [CAR]     │      → offset = +0.05m (nearly centered)
      │      │      │        curvature = 0.001 (nearly straight)
      │      │      │        → COMMAND: FORWARD
      │      │      │


      │      │      │
      │   [CAR]     │      → offset = +0.35m (drifted right)
      │             │        → COMMAND: LEFT
      │             │


      │             │
      │   [CAR]     │      → Only right lane detected
                    │        → COMMAND: LEFT (move toward center)
                    │


      │      │      │
      │   [CAR]  ▐█▌│      → Obstacle at 1.5m
      │      │      │        → COMMAND: STOP (hard override)
```

### Curvature → Turn Command Mapping

```
curvature > +0.005 m⁻¹  →  RIGHT TURN (road curves right)
curvature < -0.005 m⁻¹  →  LEFT TURN  (road curves left)
|curvature| ≤ 0.005     →  Based on offset (straight road)
```

---

## 11. Evaluation Metrics

Since we're working with pseudo-labels and a hackathon setting, we use a multi-level evaluation:

### Lane Detection Metrics
| Metric | Target |
|---|---|
| F1-Score (lane pixel) | > 0.85 |
| Accuracy (row anchor) | > 94% |
| False Positive Rate | < 5% |

### Decision Accuracy
- **Manual Evaluation:** Drive the MNNIT route, record 100 frames, manually annotate ground truth commands
- Compare model predictions vs manual annotations
- Target: **> 88% command accuracy**

### System-Level Metrics
| Metric | Target |
|---|---|
| End-to-End Latency | < 100ms |
| STOP precision (safety-critical) | > 99% |
| False STOP rate | < 8% |
| Uncertainty calibration (ECE) | < 0.05 |

### Ablation Study
Test with each module removed to demonstrate contribution:
- No scene understanding → decision quality drops X%
- No BEV transform → curvature estimation fails on curves
- No confidence gating → unsafe commands increase by X%

---

## 12. Deployment Architecture

### Local Demo Setup (Hackathon)

```
Laptop/Server
    ├── Docker container: RoadSage API (port 8000)
    ├── Docker container: React Dashboard (port 3000)
    └── USB Webcam OR pre-recorded video stream
```

### Edge Setup (Advanced Demo)

```
Raspberry Pi 4 / Jetson Nano
    ├── ONNX Runtime inference engine
    ├── Camera: Pi Camera Module v2
    ├── Output: HDMI display with lane overlay
    └── WiFi hotspot → phone can view dashboard
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./outputs:/outputs
    environment:
      - MODEL_PATH=/models/production
      
  dashboard:
    build: ./dashboard
    ports:
      - "3000:3000"
    depends_on:
      - api
      
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3001"
```

---

## 13. Advanced Features That Win Hackathons

### 🏆 Feature 1: Multi-Frame Temporal Consistency
- Buffer last 5 frames
- Run temporal smoothing on lane predictions using Kalman Filter
- Eliminates flickering lane detections caused by shadows/reflections
- **Why it matters:** Real-world driving requires stable, not frame-by-frame decisions

### 🏆 Feature 2: Uncertainty-Aware Decision Making
- Every decision has a confidence interval, not just a point estimate
- MC Dropout provides epistemic uncertainty
- "I'm not sure" is a valid and safe answer → vehicle slows/stops
- **Why it matters:** Safety-critical AI must know what it doesn't know

### 🏆 Feature 3: GradCAM Explainability (Live)
- Every prediction comes with a visual explanation
- Judges can see *why* the model said LEFT or STOP
- **Why it matters:** Black-box AI is a red flag in safety applications

### 🏆 Feature 4: Campus-Specific Fine-Tuning
- Specifically trained on MNNIT roads (tree shadows, yellow-black bollards, dust)
- Vastly outperforms any general-purpose system on this domain
- **Why it matters:** Domain adaptation is the real engineering challenge

### 🏆 Feature 5: ONNX Edge Deployment
- Models run on a Raspberry Pi with < 100ms latency
- Live demo with real camera, not pre-recorded video
- **Why it matters:** A working demo beats a slide deck every time

### 🏆 Feature 6: Self-Supervised Learning Pipeline
- No manual labeling required — the system labels itself
- Scalable: more MNNIT roads = more data = better model, zero human effort
- **Why it matters:** This is how real autonomous driving companies scale

### 🏆 Feature 7: Production Monitoring
- Prometheus metrics: inference latency, throughput, confidence distribution
- Grafana dashboard: real-time model health
- **Why it matters:** Shows you think beyond the hackathon, like a real engineer

### 🏆 Feature 8: REST API with Auto-Documentation
- FastAPI auto-generates OpenAPI (Swagger) docs
- Any team member or judge can call the API directly from browser
- **Why it matters:** Integration-ready; shows engineering maturity

---

## 14. Project File Structure

```
roadsage/
├── README.md
├── road_sage.md                  ← this document
├── docker-compose.yml
├── requirements.txt
├── requirements-edge.txt         ← lighter requirements for Pi/Jetson
│
├── configs/
│   ├── production.yaml
│   ├── development.yaml
│   ├── lane_detection.yaml
│   ├── decision_engine.yaml
│   └── augmentation.yaml
│
├── data/
│   ├── mnnit/
│   │   ├── raw/
│   │   ├── pseudo_labels/
│   │   └── verified/
│   └── external/
│       ├── tusimple/
│       └── culane/
│
├── roadsage/                     ← main Python package
│   ├── __init__.py
│   ├── engine.py                 ← main RoadSageEngine class
│   ├── preprocessing/
│   │   ├── augmentation.py
│   │   ├── image_quality.py
│   │   └── perspective.py
│   ├── lane_detection/
│   │   ├── ufld_model.py
│   │   ├── bev_transform.py
│   │   └── lane_geometry.py
│   ├── scene_understanding/
│   │   ├── object_detector.py
│   │   ├── depth_estimator.py
│   │   └── surface_classifier.py
│   ├── decision/
│   │   ├── geometric_logic.py
│   │   ├── ml_fallback.py
│   │   ├── safety_gate.py
│   │   └── confidence_fusion.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   └── visualizer.py
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       └── kalman_filter.py
│
├── training/
│   ├── trainers/
│   │   ├── train_lane.py
│   │   └── train_decision.py
│   ├── scripts/
│   │   ├── generate_pseudo_labels.py
│   │   ├── filter_dataset.py
│   │   ├── run_ablation.py
│   │   └── export_onnx.py
│   └── evaluation/
│       ├── evaluate_lane.py
│       ├── evaluate_decision.py
│       └── generate_report.py
│
├── api/
│   ├── Dockerfile
│   ├── main.py
│   ├── routes/
│   │   ├── predict.py
│   │   ├── health.py
│   │   └── batch.py
│   ├── websocket/
│   │   └── stream.py
│   └── middleware/
│       ├── logging.py
│       └── rate_limit.py
│
├── dashboard/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── VideoFeed.jsx
│           ├── DecisionPanel.jsx
│           ├── LaneMetrics.jsx
│           ├── GradCamView.jsx
│           ├── ConfidenceMeter.jsx
│           ├── DecisionHistory.jsx
│           └── SystemHealth.jsx
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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lane_detection_demo.ipynb
│   ├── 03_pseudo_labeling.ipynb
│   ├── 04_decision_logic_analysis.ipynb
│   └── 05_model_evaluation.ipynb
│
└── monitoring/
    ├── prometheus.yml
    └── grafana/
        └── dashboards/
            └── roadsage.json
```

---

## 15. Roadmap & Milestones

### Phase 1 — Foundation (Days 1–2)
- [ ] Set up project repository and Docker environment
- [ ] Collect MNNIT campus road images (target: 500+)
- [ ] Implement preprocessing pipeline (CLAHE, augmentation)
- [ ] Set up MLflow experiment tracking

### Phase 2 — Core Models (Days 3–4)
- [ ] Download and test UFLD v2 pretrained weights
- [ ] Implement BEV transform and lane geometry module
- [ ] Implement geometric decision logic
- [ ] Set up YOLOv8n for obstacle detection
- [ ] Integrate MiDaS depth estimator

### Phase 3 — Self-Training (Day 5)
- [ ] Run pseudo-label generation pipeline on MNNIT data
- [ ] Fine-tune lane detector on pseudo-labeled MNNIT data
- [ ] Train decision CNN on geometric pseudo-labels
- [ ] Implement confidence + safety gate

### Phase 4 — Explainability & API (Day 6)
- [ ] Implement GradCAM overlay
- [ ] Build FastAPI backend with all endpoints
- [ ] Implement WebSocket streaming
- [ ] Build React dashboard

### Phase 5 — Hardening & Demo Prep (Day 7)
- [ ] Export all models to ONNX
- [ ] Test on edge hardware (if available)
- [ ] Run full evaluation on held-out MNNIT images
- [ ] Set up Prometheus + Grafana monitoring
- [ ] Prepare demo video (30-second highlight)
- [ ] Write final evaluation report

---

## 16. Risk Analysis & Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Insufficient MNNIT images | Medium | High | Use synthetic data from CARLA + aggressive augmentation |
| Pseudo-labels are noisy | High | Medium | Confidence filtering + human spot-check on 10% |
| Lane detector fails on faded markings | High | High | Rely on edge detection (bollards) as fallback lanes |
| Edge hardware too slow | Medium | Low | ONNX + TensorRT; demo can run on laptop |
| No lane markings detected at all | Medium | High | Scene-level ML fallback + STOP safety gate |
| Tree shadows confuse lane detector | High | Medium | Shadow augmentation during training |
| Model overfits to specific lighting | Medium | High | Train across morning/afternoon/evening images |

---

## 17. Engineering Decisions Log

### Decision 1: Why NOT use end-to-end imitation learning (behavioral cloning)?
**Alternative:** Train a CNN to directly output command from image (like NVIDIA's DAVE-2)  
**Rejected because:** Requires paired (image, steering command) data — impossible without a vehicle + sensor setup. Also, no interpretability.  
**Chosen:** Geometric lane analysis → interpretable, no paired data needed.

### Decision 2: Why UltraFast Lane Detection over Semantic Segmentation?
**Alternative:** DeepLabV3+, SegFormer — pixel-level road/lane segmentation  
**Rejected because:** 150ms+ inference even on GPU; overkill for lane boundary extraction; harder to extract geometric parameters  
**Chosen:** UFLD v2 — 5ms GPU inference, directly outputs lane coordinates, easy polynomial fitting

### Decision 3: Why MiDaS over stereo depth?
**Alternative:** Stereo camera + disparity map (precise metric depth)  
**Rejected because:** Requires hardware modification; stereo calibration complexity; not available in typical competition setup  
**Chosen:** MiDaS monocular depth — works with a single camera, relative depth sufficient for stop/go decisions

### Decision 4: Why Hybrid decision engine over pure ML?
**Alternative:** End-to-end ML classifier for all decisions  
**Rejected because:** Requires thousands of labeled (image, command) pairs; black box; unsafe for demo  
**Chosen:** Geometric rules for normal cases (transparent, reliable) + ML fallback for edge cases

### Decision 5: Why albumentations over torchvision transforms?
**Alternative:** torchvision.transforms (standard PyTorch)  
**Rejected because:** ~10x slower for complex augmentations; less augmentation variety  
**Chosen:** albumentations — 10x faster (uses OpenCV under the hood), 70+ augmentation types, supports keypoint/lane coordinate transforms

### Decision 6: Why FastAPI over Flask?
**Alternative:** Flask (simpler, more familiar)  
**Rejected because:** Synchronous by default; slower; no automatic OpenAPI docs  
**Chosen:** FastAPI — async, 3x faster under load, auto-generates Swagger docs, type validation with Pydantic

### Decision 7: Why pseudo-labeling over full manual annotation?
**Alternative:** Manually label all 500+ images  
**Rejected because:** Time-consuming (days of work), error-prone, doesn't scale  
**Chosen:** Self-training with pseudo-labels — 80%+ of data labeled automatically, human reviews only border cases

---

## 18. Future Scope (Post-Hackathon)

1. **GPS Integration:** Fuse lane decisions with GPS waypoints for full campus navigation
2. **HD Map Building:** Use SLAM to build a precise 3D map of MNNIT campus roads
3. **Multi-Camera Setup:** Add rear and side cameras for 360° awareness
4. **Vehicle Integration:** Deploy on an actual RC car or golf cart for real autonomous navigation
5. **Night Mode:** Specialized model trained on night/low-light images
6. **Pedestrian Prediction:** Predict where pedestrians will walk (trajectory forecasting)
7. **V2X Communication:** Vehicle-to-infrastructure communication for smart traffic management
8. **Federated Learning:** Multiple vehicles share learned experiences without sharing raw data
9. **Reinforcement Learning Layer:** Fine-tune decisions based on simulated driving feedback
10. **ADAS Suite:** Full ADAS — lane departure warning, collision warning, speed sign recognition

---

## Appendix A: Key Papers to Reference in Presentation

1. **UltraFast Lane Detection v2** — Qinghao Feng et al., 2022
2. **MiDaS: Towards Robust Monocular Depth Estimation** — Ranftl et al., 2020
3. **YOLOv8** — Ultralytics, 2023
4. **Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method** — Lee, 2013
5. **GradCAM: Visual Explanations from Deep Networks** — Selvaraju et al., 2017
6. **End-to-End Learning for Self-Driving Cars (DAVE-2)** — Bojarski et al., NVIDIA, 2016

---

## Appendix B: Quick-Start Commands

```bash
# Clone and setup
git clone https://github.com/yourteam/roadsage
cd roadsage
pip install -r requirements.txt

# Download pretrained models
bash models/download_models.sh

# Run on a single image
python -m roadsage.engine --image path/to/road_image.jpg --output results/

# Start API server
uvicorn api.main:app --reload --port 8000

# Start full stack with Docker
docker-compose up --build

# Run training pipeline
python training/trainers/train_lane.py --config configs/lane_detection.yaml

# Run evaluation
python training/evaluation/evaluate_decision.py \
    --test_dir data/mnnit/verified/ \
    --model_path models/production/
```

---

*RoadSage — Seeing the road, understanding the path.*  
*Built for MNNIT Allahabad | Production-grade autonomous navigation*

---
**Document Version:** 1.0  
**Last Updated:** May 2026  
**Status:** Active Development