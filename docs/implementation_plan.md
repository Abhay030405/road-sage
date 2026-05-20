# RoadSage — Implementation Plan (7 Phases)

> This document defines the complete 7-phase execution plan for building RoadSage. Each phase has a clear objective, a detailed breakdown of tasks, the files affected, the engineering decisions to be made, exit criteria, and known risks. The phases are designed to be executed sequentially, with each phase leaving the system in a working and testable state.

---

## Phase 1 — Foundation, Environment, and Data Collection

**Duration estimate:** Days 1–2  
**Goal:** Establish a working development environment, set up the full project infrastructure, and collect the raw MNNIT campus image dataset. At the end of this phase, we have a reproducible environment and a clean, filtered dataset ready for pseudo-labeling.

---

### 1.1 Repository and Environment Setup

**Docker environment (primary)**

The entire stack runs in Docker from the start. This eliminates "works on my machine" problems and ensures every team member works against the same environment. The `docker-compose.yml` defines four services from day one: the API, the dashboard, Prometheus, and Grafana. Running `docker-compose up --build` from the root should produce a fully operational system by the end of Phase 1.

Tasks:
- Verify `docker-compose.yml` brings up all services cleanly
- Confirm API is reachable at `localhost:8000`, dashboard at `localhost:3000`
- Confirm Prometheus scrapes API at `/metrics` and Grafana dashboard loads

**Python environment (backend)**

While Docker is primary, a local Python virtual environment is needed for running training scripts and notebooks outside Docker. Create `backend/venv/`, install `requirements.txt`, and verify all imports resolve.

Tasks:
- `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Verify `import torch, cv2, albumentations, ultralytics, fastapi` all succeed
- Verify `pytest tests/` runs without import errors (even if tests fail due to missing models)

**Configuration validation**

All configs in `configs/` should be loadable and validated at startup. Any missing required key should raise a clear error, not a confusing `KeyError` mid-inference.

Tasks:
- Review `configs/production.yaml` and `configs/development.yaml` for completeness
- Ensure model paths, thresholds, and device settings are all present with correct types
- Add a `config_validator.py` utility that parses a config and raises descriptive errors for missing/invalid fields

---

### 1.2 Data Collection Strategy

**What to collect**

The goal is 500–800 images minimum. Diversity matters more than quantity. Cover:
- Main campus entrance road
- Internal narrow roads near academic blocks
- Roundabouts and T-junctions
- Tree-shaded sections (the hardest for lane detection)
- Morning (hazy), afternoon (bright), and evening (low angle light) conditions
- Wet road sections if available

**Collection tooling**

Write a simple data collection script (`training/scripts/collect_frames.py`) that:
- Captures from a webcam or reads from a video file
- Applies a minimum time delta between saved frames (prevents near-duplicate capture)
- Names frames with timestamp and condition tag (e.g., `mnnit_morning_0042.jpg`)
- Saves a sidecar JSON with capture metadata (time, GPS if available, camera parameters)

**Immediate quality checks during collection**

Do not save obviously bad frames. Run blur detection (Laplacian variance < 50 → reject) and brightness check inline during collection. This saves cleanup time later.

---

### 1.3 Data Quality Pipeline (`training/scripts/filter_dataset.py`)

After collection, every image passes four automatic quality gates:

1. **Blur** — Laplacian variance < 50 → reject
2. **Brightness** — Mean pixel < 30 or > 220 → reject
3. **Road Coverage** — HSV-based road pixel ratio < 20% → reject
4. **Deduplication** — Perceptual hash comparison (pHash); if cosine similarity > 0.98 with any existing image → reject

Log rejection statistics: total images, rejected by each filter, final count accepted. This log becomes important for reporting and for iterating on the filter thresholds if too many good images are being rejected.

**Files affected:**
- `training/scripts/filter_dataset.py`
- `data/mnnit/raw/` (input)
- `data/mnnit/verified/` (filtered output)

---

### 1.4 Augmentation Pipeline Validation (`app/preprocessing/augmentation.py`)

Before training begins, manually visually validate the augmentation pipeline:
- Load 5 sample MNNIT images
- Run through the full augmentation suite
- Display augmented versions side-by-side with originals
- Verify augmentations look realistic (shadows look like tree shadows, haze looks like haze)
- Adjust probabilities and strengths if any augmentation is too aggressive or unrealistic

Notebook `01_data_exploration.ipynb` is the right place to do this visual validation.

---

### Phase 1 Exit Criteria

- [ ] `docker-compose up --build` completes without errors
- [ ] API health endpoint returns 200 with all checks passing
- [ ] ≥ 500 images collected, quality-filtered, and stored in `data/mnnit/verified/`
- [ ] Augmentation pipeline visually validated on sample images
- [ ] `pytest tests/` runs without import errors

---

## Phase 2 — Core Perception: Lane Detection Engine

**Duration estimate:** Days 3–4  
**Goal:** Implement and validate the full lane detection pipeline, from raw image to lane geometry. At the end of this phase, given any MNNIT campus image, the system can output lane pixel coordinates, BEV-transformed lane curves, lane offset, and road curvature.

---

### 2.1 UltraFast Lane Detection v2 Integration (`app/lane_detection/ufld_model.py`)

**Model loading and inference**

UFLD v2 is available as a PyTorch model. The model must be:
- Loaded once at system startup (not per-request)
- Wrapped in a class with a clean `predict(image: np.ndarray) -> LaneDetectionResult` interface
- Supported for both GPU and CPU inference (auto-detect device)
- Wrapped in a try/except that returns a `LaneDetectionResult` with `no_lanes_detected=True` rather than crashing on model failure

The `LaneDetectionResult` dataclass should contain:
```python
@dataclass
class LaneDetectionResult:
    left_lane: List[Tuple[int, int]]    # pixel coordinates
    right_lane: List[Tuple[int, int]]
    center_lane: Optional[List[Tuple[int, int]]]
    confidence: List[float]             # per-lane confidence
    no_lanes_detected: bool
    inference_time_ms: float
```

**Pre/post-processing**

UFLD v2 expects a specific input size (800×288) and normalization. The preprocessing must:
- Resize the input image to model input dimensions
- Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Run the model
- Convert the raw row-anchor grid output back to pixel coordinates
- Apply a confidence threshold (default: 0.75, configurable in `configs/lane_detection.yaml`)

**Model download**

`models/download_models.sh` must download the UFLD v2 pretrained weights (TuSimple and/or CULane pretrained checkpoint). Verify the SHA256 hash of downloaded files to detect corrupt downloads.

---

### 2.2 Bird's Eye View Transform (`app/lane_detection/bev_transform.py`)

The IPM (Inverse Perspective Mapping) transform converts the front-facing camera view into a top-down view where lane geometry can be accurately computed.

**Calibration**

The perspective transform matrix requires 4 source points (in the original image) and 4 destination points (in the BEV image). These points are defined by camera mount geometry. For MNNIT images with a known camera position, calibrate these points empirically:
- Find an image with clearly visible straight lane markings
- Identify 4 lane boundary points that form a trapezoid in the original image
- Define their corresponding positions in the BEV (rectangle)
- Compute `M = cv2.getPerspectiveTransform(src_points, dst_points)`
- Store M in `configs/lane_detection.yaml` as a serialized matrix

The transform matrix must be stored in config (not hardcoded) so it can be updated when the camera mount changes.

**Polynomial fitting**

After BEV transform, fit a 2nd-degree polynomial to each detected lane:
```python
coeffs = np.polyfit(y_coords, x_coords, deg=2)  # x = ay² + by + c
```

From the polynomial coefficients, compute:
- **Lane offset** from image center: `offset = (left_x + right_x) / 2 - image_center_x`, convert to meters using pixels-per-meter calibration
- **Road curvature**: `R = ((1 + (2Ay + B)²)^1.5) / |2A|`, compute at the bottom of the image (closest to vehicle)

---

### 2.3 Lane Geometry Module (`app/lane_detection/lane_geometry.py`)

This module wraps the BEV polynomial results into a clean `LaneGeometry` dataclass:

```python
@dataclass
class LaneGeometry:
    offset_m: float             # lateral offset from center (negative = left)
    curvature_inv_m: float      # 1/radius; positive = right curve, negative = left
    vanishing_point_x: int      # x-pixel of lane convergence
    road_width_m: float         # estimated road width
    left_lane_detected: bool
    right_lane_detected: bool
```

The vanishing point is where the left and right polynomial fits converge. It indicates where the road is heading and is used as a supplementary signal in the decision engine.

---

### 2.4 Validation and Notebook (`notebooks/02_lane_detection_demo.ipynb`)

Systematically test the lane detection pipeline on:
- 10 easy cases (clear road, good lighting, visible markings)
- 10 medium cases (some shadows or partial occlusion)
- 5 hard cases (heavy shadows, faded markings, curves)

For each, display: original image, lane overlay, BEV view, computed offset and curvature. This notebook is essential for catching calibration errors before they propagate to the decision engine.

**Files affected:**
- `app/lane_detection/ufld_model.py`
- `app/lane_detection/bev_transform.py`
- `app/lane_detection/lane_geometry.py`
- `models/download_models.sh`
- `configs/lane_detection.yaml`
- `tests/test_lane_detection.py`
- `notebooks/02_lane_detection_demo.ipynb`

---

### Phase 2 Exit Criteria

- [ ] Lane detector runs on all MNNIT test images without crashing
- [ ] BEV transform produces geometrically correct top-down lane view
- [ ] Offset and curvature computed correctly for 3+ manually verified test cases
- [ ] `tests/test_lane_detection.py` passes all unit tests
- [ ] Notebook shows visual validation across easy/medium/hard cases

---

## Phase 3 — Scene Understanding: Obstacles and Depth

**Duration estimate:** Day 5 (morning)  
**Goal:** Implement obstacle detection (YOLOv8n) and monocular depth estimation (MiDaS), and fuse their outputs into a structured scene context that the decision engine can consume.

---

### 3.1 Object Detector (`app/scene_understanding/object_detector.py`)

**YOLOv8n integration**

Ultralytics provides a clean Python API for YOLOv8. The integration is straightforward but requires:
- Model loaded once at startup
- Inference on each frame with confidence threshold (0.5) and NMS IoU threshold (0.45)
- Filtering to only relevant classes: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7), animal classes
- Add campus-specific classes in fine-tuning (cycle rickshaw, auto-rickshaw, pedestrian with load)

Output `DetectionResult` should include bounding boxes in normalized coordinates, class names, confidence scores, and a list of high-priority obstacles (those above a confidence threshold and within the vehicle's path).

**Path-relevant filtering**

Not every detected object is a navigation hazard. A pedestrian on the sidewalk is not an immediate concern. Filter to objects in the "lane corridor" — roughly the center 40% of the image width. Objects outside this corridor are logged but don't trigger stops.

---

### 3.2 Depth Estimator (`app/scene_understanding/depth_estimator.py`)

**MiDaS integration**

MiDaS v2.1 Small is available via `torch.hub.load`. It produces a relative inverse depth map (higher values = closer to camera). The output is not in metric units and requires calibration.

**Depth-to-distance calibration**

Given a known object at a known distance (e.g., place a cone at 3 meters and capture an image), measure the MiDaS output value at that pixel location. This gives a calibration constant to convert MiDaS relative depth to approximate metric distance for objects of known size.

For the STOP decision, we use a depth threshold rather than an absolute distance. Through empirical calibration on MNNIT roads, determine the MiDaS value that corresponds to "object within 2.0 meters" and store it in `configs/decision_engine.yaml` as `obstacle_stop_depth_threshold`.

---

### 3.3 Scene Fusion Logic

The obstacle detector gives us bounding boxes and classes. The depth estimator gives us a depth map. The fusion logic extracts the minimum depth value within each bounding box to get the estimated distance to each detected object.

```python
def get_obstacle_distance(depth_map, bbox):
    roi = depth_map[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    return roi.max()   # max inverse depth = minimum metric distance
```

The scene understanding module outputs a `SceneContext` object:

```python
@dataclass
class SceneContext:
    obstacles: List[DetectedObject]     # all detected objects
    nearest_obstacle_depth: float       # depth value of closest path obstacle
    immediate_hazard: bool              # True if nearest_obstacle_depth > stop_threshold
    road_surface: SurfaceClass          # clean / pothole / speed_breaker / waterlogged
```

---

### 3.4 Road Surface Classifier (`app/scene_understanding/surface_classifier.py`)

A lightweight CNN classifier (MobileNetV2) trained on road surface patches.

Classes: `clean`, `pothole`, `speed_breaker`, `waterlogged`

Training data: 200 MNNIT road patch images labeled manually (this is one of the few places where manual labeling is tractable due to the small dataset size) plus augmentation.

In Phase 3, implement the inference interface. Training happens in Phase 4 alongside the decision CNN.

**Files affected:**
- `app/scene_understanding/object_detector.py`
- `app/scene_understanding/depth_estimator.py`
- `app/scene_understanding/surface_classifier.py`
- `tests/test_decision_engine.py` (scene context unit tests)

---

### Phase 3 Exit Criteria

- [ ] YOLOv8n detects pedestrians and vehicles in MNNIT test images
- [ ] MiDaS produces qualitatively correct depth maps (near objects brighter)
- [ ] Depth threshold calibrated empirically on at least 3 known-distance test cases
- [ ] Scene fusion correctly identifies nearest path obstacle
- [ ] `immediate_hazard` flag is correctly set on test images with close obstacles

---

## Phase 4 — Decision Engine and Self-Training Pipeline

**Duration estimate:** Days 5 (afternoon)–6  
**Goal:** Implement the complete hybrid decision engine, run the pseudo-labeling self-training loop, train the ML fallback model, and implement the confidence + safety gate. At the end of this phase, the system can make reliable driving decisions on MNNIT images.

---

### 4.1 Geometric Decision Logic (`app/decision/geometric_logic.py`)

Implement the priority chain exactly as designed:

```
1. Safety Gate check (delegate to safety_gate.py)
2. Dual-lane geometric decision
   - compute command from offset and curvature
3. Single-lane fallback
   - only one lane detected → steer toward center
4. Return None → trigger ML fallback
```

All threshold values (offset threshold, curvature threshold, etc.) must be read from `configs/decision_engine.yaml` and never hardcoded in source. This enables tuning without code changes.

**Unit testability requirement**

`geometric_logic.py` must be purely functional — given a `LaneGeometry` and `SceneContext`, return a `DecisionResult`. No I/O, no model loading, no side effects. This makes it trivially unit-testable.

```python
def compute_geometric_decision(
    geometry: LaneGeometry,
    scene: SceneContext,
    config: DecisionConfig
) -> Optional[DecisionResult]:
    ...
```

---

### 4.2 Pseudo-Label Generation (`training/scripts/generate_pseudo_labels.py`)

This is one of the most important scripts in the project. For each MNNIT image:

1. Run the pretrained lane detector → get `LaneDetectionResult`
2. If `lane_confidence < 0.85` → skip (do not generate pseudo-label)
3. Compute `LaneGeometry` from BEV transform
4. Run `compute_geometric_decision(geometry, ...)` → get `DecisionResult`
5. If decision was made via geometric path (not fallback) → save pseudo-label
6. Save as: `{"image": "path.jpg", "command": "FORWARD", "confidence": 0.91, "source": "geometric"}`

The pseudo-label JSONL file goes to `data/mnnit/pseudo_labels/`.

Log statistics: images processed, images with confidence < threshold (skipped), pseudo-labels generated, command distribution.

---

### 4.3 ML Fallback Training (`training/trainers/train_decision.py`)

Train MobileNetV3-Small on the pseudo-labeled dataset:

- Input: 224×224 RGB images
- Target: 4-class softmax (FORWARD=0, LEFT=1, RIGHT=2, STOP=3)
- Loss: CrossEntropyLoss with label smoothing (0.1) to account for pseudo-label noise
- Optimizer: AdamW, lr=1e-3, weight decay=1e-4
- LR schedule: CosineAnnealingLR
- Data split: 80% train, 10% val, 10% test (stratified by command class)
- Dropout: 0.3 in final classifier layers (required for MC Dropout inference)
- Training: 50 epochs, early stopping if val loss doesn't improve for 10 epochs

**Class imbalance handling**

The pseudo-label distribution will be heavily skewed toward FORWARD. Use weighted sampling (WeightedRandomSampler) to ensure balanced batches. Log the class distribution before and after weighting.

---

### 4.4 Confidence Fusion (`app/decision/confidence_fusion.py`)

After obtaining predictions from the geometric logic and/or ML fallback, the confidence fusion module computes the final confidence score:

```python
final_confidence = (
    w_lane * lane_detection_confidence +
    w_geo  * geometric_signal_strength +
    w_ml   * ml_softmax_max_value
) / (w_lane + w_geo + w_ml)
```

Weights are configurable. `geometric_signal_strength` is a derived metric — how far the offset/curvature is from the decision boundary (larger margin = stronger signal).

If `final_confidence < MIN_CONFIDENCE` (default: 0.60), the command is overridden to `STOP`.

---

### 4.5 Safety Gate (`app/decision/safety_gate.py`)

A pure deterministic module. Given a `SceneContext`, returns whether the safety gate is triggered:

```python
def evaluate_safety(
    scene: SceneContext,
    last_n_commands: List[str],
    config: SafetyConfig
) -> SafetyGateResult:
    ...
```

Safety conditions checked:
1. `scene.immediate_hazard == True` → STOP
2. `scene.nearest_obstacle_depth > stop_threshold` → STOP  
3. `final_confidence < MIN_CONFIDENCE` → STOP
4. Command would change more than 2 steps in one frame (e.g., FORWARD → STOP without transition) → only allow if hazard is detected

The safety gate never blocks a STOP command. It only adds STOP.

---

### 4.6 Self-Training Iteration 2

After training the initial ML fallback model, run a second round of pseudo-label generation using the improved lane detector (now fine-tuned on MNNIT pseudo-labels from iteration 1). This should bring coverage from ~60% to ~80–85%.

Re-train the decision CNN with the expanded dataset.

**Files affected:**
- `app/decision/geometric_logic.py`
- `app/decision/ml_fallback.py`
- `app/decision/safety_gate.py`
- `app/decision/confidence_fusion.py`
- `training/scripts/generate_pseudo_labels.py`
- `training/trainers/train_decision.py`
- `data/mnnit/pseudo_labels/`
- `tests/test_decision_engine.py`
- `tests/test_safety_gate.py`
- `notebooks/03_pseudo_labeling.ipynb`
- `notebooks/04_decision_logic_analysis.ipynb`

---

### Phase 4 Exit Criteria

- [ ] Pseudo-label pipeline generates labels for ≥ 60% of MNNIT images in iteration 1
- [ ] ML fallback model trained, achieves > 80% val accuracy on pseudo-labeled test set
- [ ] Geometric decision logic unit tested with ≥ 20 test cases covering all branches
- [ ] Safety gate tested: STOP is always triggered for obstacle < threshold
- [ ] `tests/test_decision_engine.py` and `tests/test_safety_gate.py` all passing
- [ ] End-to-end inference on 10 MNNIT test images produces sensible commands

---

## Phase 5 — Full System Integration, Explainability, and API

**Duration estimate:** Day 6  
**Goal:** Integrate all modules into the `RoadSageEngine` class, implement GradCAM and lane visualization, build the complete FastAPI backend with all endpoints and WebSocket streaming, and verify end-to-end inference works correctly.

---

### 5.1 RoadSageEngine (`app/engine.py`)

The `RoadSageEngine` class is the single entry point for inference. It:
- Loads all models at initialization (not per-request)
- Orchestrates the full pipeline in order: preprocess → lane detect → scene understand → decide → explain
- Returns a structured `PredictionResult`
- Measures and logs latency for each component
- Handles all errors gracefully — never propagates a raw exception to the API layer

```python
class RoadSageEngine:
    def __init__(self, config_path: str): ...
    def predict(self, image: np.ndarray) -> PredictionResult: ...
    def predict_batch(self, image_dir: str, ...) -> List[PredictionResult]: ...
    def start_stream(self, source: int, websocket_port: int, ...): ...
```

Model loading happens once in `__init__`. The engine reads five env vars at startup: `DEVICE` (cpu|cuda), `UFLD_MODEL`, `DETECTOR_MODEL`, `DEPTH_MODEL`, `FALLBACK_MODEL`. Switching CPU/GPU mode requires only changing env vars — no code change. Subsequent calls to `predict()` reuse loaded models. This is critical for API performance.

---

### 5.2 GradCAM Implementation (`app/explainability/gradcam.py`)

GradCAM is computed against the final convolutional layer of the UFLD v2 backbone (or the decision CNN when using ML fallback). Implementation:

1. Register a forward hook on the target layer to capture activation maps
2. Register a backward hook to capture gradients
3. Forward pass → get prediction
4. Backward pass with respect to the predicted class logit
5. Compute channel-wise mean of gradients → weights
6. Weighted sum of activation maps → raw CAM
7. Apply ReLU (only keep positive influences)
8. Resize to original image dimensions
9. Normalize to [0, 1] and apply colormap (jet)
10. Overlay on original image with transparency alpha=0.4

The GradCAM generation is wrapped in its own thread/async context so that it does not block the prediction pipeline in streaming mode. In streaming mode, GradCAM is generated for every 5th frame only (to maintain throughput).

---

### 5.3 Lane Visualizer (`app/explainability/visualizer.py`)

Renders:
- Detected lane lines on original image (color-coded by confidence: green/yellow/red)
- Lane center fill (translucent green polygon between left and right lane)
- Offset indicator (arrow showing which direction to correct)
- BEV minimap (small top-right inset showing lanes from above)
- Decision command overlay (large text bottom-center)
- Confidence bar (left edge of image)

All visualization parameters (font size, colors, opacity) are configurable in `configs/production.yaml`.

---

### 5.4 FastAPI Routes

**`POST /api/v1/predict`** (`api/routes/predict.py`)  
Accepts multipart/form-data image. Runs `engine.predict()`. Returns JSON PredictionResult + optional base64-encoded GradCAM overlay if `include_viz=true` query param.

**`GET /api/v1/health`** (`api/routes/health.py`)  
Runs a warmup inference, returns model status, average latency (rolling window of last 100 inferences), memory usage, and config hash.

**`POST /api/v1/batch`** (`api/routes/batch.py`)  
Accepts a list of base64-encoded images or a ZIP file. Returns batch predictions. Processes in parallel using `asyncio.gather` with a concurrency limit (default: 4).

**`WS /ws/live`** (`api/websocket/stream.py`)  
Client sends frames as binary WebSocket messages. Server responds with JSON prediction results. Handles connection lifecycle, graceful disconnection, and rate limiting (max 30 FPS input, downsampled to 15 FPS processed).

---

### 5.5 Middleware

**Logging middleware** (`api/middleware/logging.py`)  
Structured JSON logging for every request: endpoint, latency, status code, decision command (if predict endpoint). No raw image data logged (privacy + storage).

**Rate limiting** (`api/middleware/rate_limit.py`)  
SlowAPI integration. Limits per endpoint per IP. Returns 429 with `Retry-After` header.

---

### Phase 5 Exit Criteria

- [ ] `engine.predict(image)` runs end-to-end without error on 20 test images
- [ ] GradCAM overlays look correct (highlighted regions correspond to lane markings / obstacles)
- [ ] All 4 API endpoints functional: `/predict`, `/health`, `/batch`, `/ws/live`
- [ ] `tests/test_api.py` all passing
- [ ] WebSocket streaming works with a test client sending frames at 15 FPS

---

## Phase 6 — Real-Time Dashboard

**Duration estimate:** Day 6 (evening)–Day 7 (morning)  
**Goal:** Build the React dashboard with all 7 UI components, WebSocket integration for live data, and a polished visual design suitable for a demo. The dashboard must be informative, real-time, and immediately understandable by a non-technical judge.

---

### 6.1 Architecture Overview

The dashboard is a React 18 single-page application built with Vite. All data flows from the backend via WebSocket (`/ws/live`) or REST polling (`/api/v1/health`). The state management approach is simple: a single top-level `useWebSocket` custom hook provides the latest `PredictionResult` to all components via React Context. No Redux, no Zustand — the data model is simple enough for context + local state.

---

### 6.2 Component Specifications

**`VideoFeed.jsx`**  
Displays the live camera frame with lane overlay drawn on top. The lane overlay (pixel coordinates from the server response) is rendered on a `<canvas>` element layered over the `<img>` tag. Overlay includes: colored lane lines, center corridor fill, offset arrow. Toggleable GradCAM mode that swaps the overlay to the GradCAM heatmap.

**`DecisionPanel.jsx`**  
The most prominent UI element. Displays the current command (FORWARD / LEFT / RIGHT / STOP) in large bold text with color coding:
- FORWARD → green background
- LEFT → amber background with left arrow
- RIGHT → blue background with right arrow
- STOP → red background with pulse animation

Also shows the current confidence percentage and a brief description of which decision path was used (Geometric / ML Fallback / Safety Gate).

**`ConfidenceMeter.jsx`**  
Radial gauge (using Recharts `RadialBarChart`) showing confidence from 0–100%. Color transitions from red (0–60%) to amber (60–80%) to green (80–100%). Includes a threshold marker line at 60% (the safety stop threshold).

**`LaneMetrics.jsx`**  
Numerical readout panel showing:
- Lanes detected (left ✓/✗, center ✓/✗, right ✓/✗)
- Lateral offset (meters, with direction indicator)
- Road curvature (m⁻¹, with "straight / mild curve / sharp curve" label)
- Road surface classification

**`GradCamView.jsx`**  
Side-by-side display of original frame and GradCAM heatmap overlay. Toggle switch to turn on/off. Slider to control overlay opacity. Label showing which layer the GradCAM was computed from.

**`DecisionHistory.jsx`**  
Scrolling timeline of the last 50 decisions. Each entry shows: timestamp, command, confidence, decision path. Color-coded rows matching `DecisionPanel` colors. Click on a row to see the corresponding GradCAM in `GradCamView`. Useful for identifying patterns (e.g., always triggers STOP at a specific road section).

**`SystemHealth.jsx`**  
Live metrics panel showing:
- FPS (frames processed per second by backend)
- P50/P95 inference latency (ms)
- CPU/GPU utilization (from `/health` endpoint)
- Model status indicators (lane detector, object detector, depth estimator — green/red)
- WebSocket connection status with auto-reconnect indicator

---

### 6.3 WebSocket Integration

Custom hook `useWebSocket` handles:
- Connection and automatic reconnection (exponential backoff: 1s, 2s, 4s, 8s, max 30s)
- Message parsing and validation
- Connection state management (connecting / connected / disconnected / error)
- Dispatching latest `PredictionResult` to context

The hook renders a banner notification when the connection is lost, and removes it automatically on reconnection.

---

### 6.4 Styling and UX

Dark theme (dark background, colored accents) appropriate for automotive/monitoring dashboards. Tailwind CSS utility classes throughout. Grid layout: VideoFeed takes 60% width on left; DecisionPanel + ConfidenceMeter stack on right; LaneMetrics, GradCamView, DecisionHistory, and SystemHealth in a 4-column row below.

All components handle the "loading" state (before first WebSocket message arrives) with skeleton placeholders, not blank panels.

---

### Phase 6 Exit Criteria

- [ ] Dashboard loads in < 2 seconds at `localhost:3000`
- [ ] All 7 components render correctly with live WebSocket data
- [ ] VideoFeed shows lane overlay (not just raw image)
- [ ] DecisionPanel updates within 100ms of backend producing a new prediction
- [ ] WebSocket auto-reconnects after disconnection
- [ ] `pnpm build` completes without TypeScript or ESLint errors

---

## Phase 7 — Hardening, Evaluation, and Production Readiness

**Duration estimate:** Day 7  
**Goal:** Run the complete evaluation suite, fix any remaining issues, export models to ONNX for edge deployment, ensure full test coverage, validate monitoring, and prepare the system for demo. At the end of this phase, the system meets all production-readiness criteria defined in the build plan.

---

### 7.1 Full Evaluation Run

**Lane detection evaluation** (`training/evaluation/evaluate_lane.py`)  
Run the lane detector on the held-out MNNIT test set (the 10% kept out from training). Compute:
- F1-score on lane pixels
- Row-anchor accuracy
- False positive rate
- Per-image visualization of failures

Target: F1 > 0.85, accuracy > 94%.

**Decision accuracy evaluation** (`training/evaluation/evaluate_decision.py`)  
Manually drive/walk 5 campus road sections and annotate ground truth commands for 100 representative frames. Run RoadSage on these frames. Compute:
- Overall command accuracy
- Per-class accuracy (is STOP always correct when it triggers?)
- Confusion matrix

Target: overall accuracy > 88%, STOP precision > 99%.

**Latency profiling**  
Run 500 inference cycles, record per-component latency. Ensure P95 < 100ms. If any component exceeds its budget, optimize before final demo.

**Uncertainty calibration**  
Run MC Dropout on 200 test images with N=10 passes. Plot calibration curve (predicted confidence vs actual accuracy). Compute ECE (Expected Calibration Error). Target: ECE < 0.05. If ECE is too high, add temperature scaling.

---

### 7.2 Ablation Study (`training/scripts/run_ablation.py`)

Run the system with each module disabled to quantify its contribution:
- No scene understanding → measure increase in false FORWARD (missed obstacles)
- No BEV transform (use raw pixel offset) → measure curvature estimation error
- No confidence gate → count commands that would have been unsafe
- No temporal smoothing → count command flicker events

Results go into `training/evaluation/generate_report.py` output.

---

### 7.3 ONNX Export (`training/scripts/export_onnx.py`)

Export both CPU and GPU variants of all four models:
- `models/lane_detector_resnet18.onnx` (CPU) / `lane_detector_resnet50.onnx` (GPU)
- `models/object_detector_nanodet.onnx` (CPU) / `object_detector_yolov8n.onnx` (GPU)
- `models/depth_estimator_midas.onnx` (CPU) / `depth_estimator_dav2.onnx` (GPU)
- `models/fallback_cnn_mobilenet.onnx` (CPU) / `fallback_cnn_efficientlite.onnx` (GPU)

Verification steps after export:
1. Run ONNX model on 5 test images, compare output numerically to PyTorch model output (max absolute diff < 1e-5)
2. Measure ONNX inference latency vs PyTorch inference latency
3. Verify ONNX models run correctly on CPU (not just GPU)

---

### 7.4 Docker Hardening

- Review all Docker images for unnecessary packages
- Ensure no secrets are baked into Docker images (use environment variables or mounted secrets)
- Verify `docker-compose up --build` starts all services cleanly from a fresh clone
- Test that the full stack works end-to-end in Docker: send an image to the API, verify the dashboard updates

**Edge Docker image**  
Build and test a separate `Dockerfile.edge` using `requirements-edge.txt`. Verify it runs on ARM64 (if Raspberry Pi / Jetson is available for testing).

---

### 7.5 Final Test Suite Pass

- Run `pytest tests/ -v --tb=short` — all tests must pass
- Run `ruff check .` from repo root — zero lint errors
- Run `cd dashboard && pnpm lint` — zero ESLint errors
- Run `cd dashboard && npx tsc --noEmit` — zero type errors

---

### 7.6 Monitoring Validation

- Verify Prometheus is scraping all configured metrics at the correct interval
- Import `monitoring/grafana/dashboards/roadsage.json` into Grafana and verify all panels load with data
- Run a 5-minute inference loop and verify all metrics populate correctly in Grafana
- Test that a spike in `safety_gate_triggers_total` is visible in the dashboard

---

### 7.7 Demo Preparation

**Demo script**  
Prepare a 30-second demo video showing:
1. System startup (`docker-compose up`, services healthy)
2. Pre-recorded MNNIT road video fed to `/ws/live`
3. Dashboard showing live lane overlay, FORWARD → LEFT → FORWARD transitions
4. A close-obstacle scenario triggering STOP with red panel
5. GradCAM view showing the model "looking at" lane markings

**Fallback demo**  
If live demo environment fails (network, hardware), have a pre-recorded demo video on-device.

**API demo**  
Show Swagger UI at `/docs` with a live `/predict` call from the browser.

---

### Phase 7 Exit Criteria

- [ ] Lane detection F1 > 0.85 on held-out test set
- [ ] Overall command accuracy > 88% on manually annotated frames
- [ ] STOP precision > 99%
- [ ] P95 inference latency < 100ms
- [ ] ECE < 0.05 (uncertainty calibration)
- [ ] ONNX models exported and numerically verified
- [ ] All pytest tests passing
- [ ] Zero ruff lint errors, zero ESLint errors
- [ ] Full Docker stack starts cleanly from fresh clone
- [ ] Prometheus + Grafana monitoring operational with live data
- [ ] Demo script rehearsed and working

---

## Summary Table

| Phase | Focus | Key Output | Duration |
|---|---|---|---|
| 1 | Foundation & Data | Filtered MNNIT dataset, Docker stack running | Days 1–2 |
| 2 | Lane Detection | UFLD v2 + BEV + geometry pipeline | Days 3–4 |
| 3 | Scene Understanding | NanoDet/YOLOv8n + MiDaS/DepthAny + obstacle fusion | Day 5 AM |
| 4 | Decision Engine & Self-Training | Trained ML fallback, full decision pipeline | Days 5 PM–6 |
| 5 | Integration & API | RoadSageEngine, FastAPI all endpoints | Day 6 |
| 6 | Dashboard | React real-time UI with 7 components | Days 6 evening–7 AM |
| 7 | Hardening & Evaluation | Metrics validated, ONNX exported, demo ready | Day 7 |

---

## Critical Path

The most time-sensitive dependency chain is:

```
Data Collection (Phase 1) 
  → Lane Detection Validation (Phase 2)
    → Pseudo-Label Generation (Phase 4)
      → ML Fallback Training (Phase 4)
        → End-to-End Engine (Phase 5)
          → Full Evaluation (Phase 7)
```

Everything else (dashboard, monitoring, ONNX export) can be parallelized or deferred without blocking the core ML pipeline. If any phase runs over schedule, cut scope from Phase 6 (reduce dashboard components) and Phase 7 (skip ablation study) before touching the core ML pipeline phases.
