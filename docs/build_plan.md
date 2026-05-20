# RoadSage — Senior ML Engineer Build Plan

> This document describes, from a senior ML engineering perspective, the complete technical strategy for building RoadSage: a vision-based lane understanding and intelligent driving decision engine for MNNIT campus roads. It covers architecture decisions, data strategy, model choices, engineering trade-offs, and production readiness considerations.

---

## 1. Problem Framing — Starting With First Principles

Before writing a single line of code, a senior ML engineer asks: **What is the actual problem, and what is the minimum viable system that solves it reliably?**

### 1.1 The Core Constraint

The dataset is **unlabeled**. This is not a minor inconvenience — it completely rules out the most obvious approach (supervised classification). You cannot train a model to output `LEFT/RIGHT/FORWARD/STOP` if you have no examples of which command corresponds to which image.

This is the central engineering challenge. Every design decision flows from it.

### 1.2 The Insight That Unlocks Everything

Driving decisions are not arbitrary. They are **deterministic functions of physical geometry**:

- If you are drifting right of the lane center, you must steer left.
- If the road curves left, you must turn left.
- If something is 1.5 meters ahead, you must stop.

This means we do not need human-labeled commands. We can **derive commands from physics** if we can accurately measure lane geometry and obstacles. The problem reduces to: *accurately perceive the road geometry from an image*, then *apply deterministic rules*.

This insight is the foundation of the entire system.

### 1.3 Where ML Is Actually Needed

Given the above, ML is required in three places:

1. **Lane Detection** — Extracting lane boundaries from raw pixels (this is genuinely hard; classical methods fail on curved, shadowed, or faded lanes)
2. **Obstacle Detection** — Identifying objects and estimating their distance
3. **Decision Fallback** — When geometric perception fails (no lanes visible), a CNN trained on pseudo-labels from the geometric system acts as a backup

Everything else — the decision logic, the safety gate, the confidence system — is deterministic engineering, not ML. This separation is intentional and important for safety and debuggability.

---

## 2. Architecture Philosophy

### 2.1 Defense in Depth

A production autonomous system cannot rely on a single model being correct. RoadSage uses **layered fallbacks**:

```
Layer 1: Geometric decision (most reliable, fully interpretable)
Layer 2: ML fallback CNN (when geometry fails)
Layer 3: Safety gate (hard override — always active)
Layer 4: Confidence gate (uncertainty-based fail-safe)
Layer 5: Temporal consistency (anti-flicker smoothing)
```

Each layer is independently testable. If the lane detector is wrong, the safety gate still catches an imminent obstacle. If the obstacle detector misses something, the confidence gate still triggers a STOP if uncertainty is high.

This is not over-engineering — it is the minimum responsible design for any system that makes real-world navigation decisions.

### 2.2 Interpretability as a First-Class Requirement

Every decision produced by RoadSage must be explainable. This is non-negotiable for three reasons:

1. **Debugging** — When the system makes a wrong decision, you need to know why
2. **Trust** — Judges, evaluators, and users need to see that the system is reasoning correctly
3. **Safety** — Black-box systems are intrinsically dangerous in safety-critical applications

This drives the inclusion of GradCAM overlays, decision trace logs, lane visualization, and the hybrid (rule-based + ML) decision engine rather than a pure end-to-end neural approach.

### 2.3 Modular, Replaceable Components

Each module — lane detector, depth estimator, object detector, decision engine — communicates through well-defined interfaces. This means:

- Any module can be upgraded independently (e.g., swap MiDaS for a better depth model) without touching the rest of the system
- Each module can be unit-tested in isolation
- Performance bottlenecks can be profiled and fixed without architectural changes
- The system can be partially deployed (e.g., lane detection only) and extended incrementally

---

## 3. Data Strategy — The Most Important Engineering Decision

### 3.1 Why Data Strategy Matters More Than Model Choice

Given two systems — one with a state-of-the-art model on poor data, and one with a modest model on high-quality domain-specific data — the second will always win on the target domain. This is one of the most consistent findings in applied ML. For MNNIT campus roads, no publicly available model has been trained on anything close to our target distribution.

The data strategy is therefore the single highest-leverage investment in the project.

### 3.2 The Self-Training Loop

We use an iterative pseudo-labeling approach based on the same principles used by Tesla Autopilot and academic work on semi-supervised learning:

**Iteration 0 (Cold Start)**  
Pretrain the lane detector on TuSimple and CULane — large public datasets with ground truth lane annotations. This gives us a model that understands "what lanes look like" in general.

**Iteration 1 (Domain Adaptation)**  
Run the pretrained model on MNNIT images. Accept predictions only where `lane_confidence > 0.85`. For accepted predictions, use the geometric decision engine to derive driving commands as pseudo-labels. Fine-tune both the lane detector and decision CNN on this data. Expected coverage: ~60% of MNNIT images.

**Iteration 2 (Refinement)**  
The improved model now achieves confidence above threshold on more images. Re-run pseudo-label generation. Expected coverage: ~80–85%. Fine-tune again with the expanded dataset.

**Iteration 3 (Human-in-the-Loop Validation)**  
Randomly sample ~50 images and manually verify that pseudo-labels are correct. Fix systematic errors. Final fine-tune on the full pseudo-labeled set.

**Why this is sound:**  
The geometric decision engine — which derives commands from lane geometry — is a high-precision teacher. It only makes predictions when it has strong geometric evidence. The ML model learns to reproduce these decisions from visual features, making it useful in cases where the geometry is ambiguous. This is knowledge distillation from a rule-based system to a neural network.

### 3.3 Data Quality Filtering

Garbage data destroys models. Every image must pass four automatic quality gates before entering training:

| Filter | Mechanism | Threshold | Rejects |
|---|---|---|---|
| Blur Detection | Laplacian variance | < 50 | Camera shake, motion blur |
| Brightness | Mean pixel value | < 30 or > 220 | Under/overexposed frames |
| Road Coverage | HSV-based road pixel ratio | < 20% | Non-road images |
| Deduplication | Perceptual hash similarity | > 0.98 cosine similarity | Near-duplicate frames |

These filters are cheap to run and prevent the training set from being polluted with unusable examples.

### 3.4 Augmentation Strategy — Solving the Distribution Problem

The MNNIT road environment has specific visual characteristics that standard pretrained models are not optimized for:

- **Morning haze** in most sample images (common in Allahabad mornings)
- **Tree-cast shadows** that cross lane lines and can be confused for lane markings
- **Yellow-black kerb bollards** which look superficially similar to lane markings in some lighting
- **Faded center lines** on older road sections
- **Variable road width** — some campus roads are narrow

Augmentation is the primary tool for making the model robust to all of these, without needing to collect images in every possible condition. The augmentation suite includes:

- `CLAHE` — Specifically targets haze and low contrast. Equalizes local contrast to recover faded markings
- `RandomShadow` (albumentations) — Simulates tree shadow patterns
- `RandomBrightness/Contrast` — Handles morning → afternoon → evening lighting shifts
- `GaussianBlur + Sharpen` — Camera shake and focus variation
- `RandomRain/Fog` — Edge case weather robustness
- `HorizontalFlip with lane mirroring` — Doubles dataset size without any collection effort
- `PerspectiveTransform` — Simulates camera mount angle variation

**Library choice: albumentations over torchvision**  
albumentations is ~10x faster for complex augmentations because it operates on NumPy arrays using OpenCV internally. For a dataset of 500–1000 images with 10-15 augmentations per training step, this speed difference matters significantly during training.

---

## 4. Model Selection — Rationale for Every Choice

### 4.1 Lane Detection: UltraFast Lane Detection v2

**What was considered:**
- Classical (Canny edge detection + Hough transform)
- Semantic segmentation (DeepLabV3+, SegFormer)
- Instance segmentation (Mask R-CNN)
- Row-anchor classification (UFLD v2)

**Why UFLD v2 wins for this problem:**

UFLD v2 reformulates lane detection as a **row-anchor classification problem**. Instead of predicting a dense pixel mask, it asks: "at each horizontal row of the image, which x-position column does this lane pass through?" This is computationally much simpler than full segmentation.

Results:
- 322 FPS on GPU (vs ~8 FPS for DeepLabV3+)
- 96.06% accuracy on TuSimple benchmark
- Directly outputs lane pixel coordinates (no post-processing needed to extract geometry)
- Handles curves naturally (row-anchor formulation is not biased toward straight lines)

Classical methods (Canny + Hough) fail specifically on the problems we face: curved lanes, shadows crossing lane lines, faded markings, and bollards that look like lane edges. They also require extensive manual tuning and do not generalize to domain shift.

**Backbone choice: ResNet-18 (CPU default) / ResNet-50 (GPU swap)**

- `ufldv2_resnet18` � lighter backbone for CPU and edge deployment (~45 FPS on Jetson Nano). Set via `UFLD_MODEL=ufldv2_resnet18`.
- `ufldv2_resnet50` � higher accuracy on complex curves and low-contrast markings. Set via `UFLD_MODEL=ufldv2_resnet50` when CUDA is available.

Both variants share the same row-anchor head and produce identical output schemas. The model loader reads `UFLD_MODEL` from the environment at startup � no code change needed to switch modes.

### 4.2 Object Detection: YOLOv8n

YOLOv8n (nano) provides < 5ms inference latency on CPU, which is necessary for real-time operation. For the obstacle detection use case — we need to know "is there an object close enough to stop?" — we do not need high-precision bounding box regression. YOLOv8n's accuracy is more than sufficient.

The nano variant is also deployable on edge hardware (Jetson Nano) without modification.

### 4.3 Depth Estimation: MiDaS v2.1 Small

Stereo depth would provide metric depth estimates, but requires hardware modification (dual cameras) and stereo calibration. For our application, **relative depth** is sufficient. We don't need to know exactly 2.47 meters — we need to know whether the obstacle is "close enough to stop." MiDaS provides this reliably from a single camera.

MiDaS Small runs at ~30ms on CPU, acceptable for our latency budget.

### 4.4 Decision CNN Fallback: MobileNetV3-Small

This model is used only when the geometric pipeline has insufficient geometric evidence (no lanes detected). It is trained entirely on **pseudo-labels generated by the geometric engine**, making it a student model that approximates the teacher's decisions from visual features alone.

MobileNetV3-Small was chosen for:
- Very low latency (< 5ms on CPU)
- Good accuracy-to-parameter ratio for 4-class classification
- Standard pretrained weights on ImageNet available

### 4.5 Uncertainty Quantification: Monte Carlo Dropout

MC Dropout is the simplest reliable uncertainty quantification technique for neural networks. By keeping dropout active at inference and running N forward passes, we get a distribution over predictions. High variance = high uncertainty = safety trigger.

This is not a research curiosity — it is a practical safety mechanism. If the model has never seen conditions like the current frame (out-of-distribution input), MC Dropout will produce inconsistent predictions across passes, flagging the uncertainty before a potentially dangerous decision is made.

---

## 5. Decision Engine Design — Correctness by Construction

The decision engine is the most safety-critical component. Its design must prioritize **correctness and predictability** over performance.

### 5.1 Why Not End-to-End?

An end-to-end model (image → command directly) would require thousands of labeled (image, command) pairs. We have none. It would also be a black box — impossible to debug, certify, or trust in a safety context.

### 5.2 The Hybrid Approach

The decision engine is **rule-based where physics is clear, ML-based only when it must be.** This is the right engineering trade-off because:

- Rules are **auditable** — every decision can be traced back to a specific geometric measurement
- Rules are **correct by construction** for the cases they handle
- ML handles the **residual uncertainty** that rules cannot cover
- The combination is more robust than either alone

### 5.3 The Priority Chain

The engine enforces a strict priority ordering. Higher-priority decisions always override lower-priority ones:

```
Priority 1 (SAFETY GATE):     Hard obstacle stop — no exceptions
Priority 2 (GEOMETRIC):       Lane-based decision — high confidence path
Priority 3 (SINGLE-LANE):     One lane fallback — medium confidence
Priority 4 (ML FALLBACK):     CNN prediction — lower confidence
Priority 5 (CONFIDENCE GATE): STOP on uncertainty — final safety net
```

This ordering means that a safety-critical condition (obstacle at 1.5m) will always produce a STOP regardless of what the lane detector or ML model says.

### 5.4 Temporal Consistency

A single frame is noisy. Tree shadows, JPEG artifacts, motion blur can all cause a brief incorrect lane detection. The decision engine implements two mechanisms to prevent this from causing erratic behavior:

1. **Persistence Filter** — A command must appear in ≥ 3 consecutive frames before being executed. This gives the system 3 frames (~200ms at 15fps) to "confirm" a decision.
2. **Moving Average Smoothing** — Confidence and offset values are smoothed over the last 5 frames using a weighted average (more recent frames weighted higher).

Both mechanisms come with a critical exception: **STOP is never smoothed or delayed**. A safety-critical command executes immediately.

---

## 6. API Design — Production-Grade From Day One

### 6.1 Endpoint Design Principles

The API must support three distinct use cases:
- **Single image inference** — for demos and real-time evaluation
- **Batch processing** — for running the full evaluation suite
- **WebSocket streaming** — for the real-time dashboard demo

Each use case has different latency and throughput requirements and should be handled separately rather than forcing all three through a single endpoint.

### 6.2 Async Architecture

FastAPI with Uvicorn (ASGI) is used throughout. All inference calls are wrapped in `asyncio.run_in_executor` to avoid blocking the event loop during model inference (which is CPU/GPU bound). This allows the API to handle concurrent requests without stalling.

### 6.3 Rate Limiting

SlowAPI middleware is configured to prevent abuse:
- `/api/v1/predict` — 30 requests/minute per IP
- `/api/v1/batch` — 5 requests/minute per IP
- `/ws/live` — connection-based throttling

### 6.4 Health Check Design

The `/health` endpoint does more than return `{"status": "ok"}`. It:
- Runs a dummy inference on a blank image to verify model is loaded
- Checks memory usage
- Returns average inference latency from a rolling window
- Returns model version and configuration hash

This makes the health endpoint useful for monitoring and for automated deployment verification.

---

## 7. Explainability — Non-Negotiable Engineering

### 7.1 GradCAM Implementation

GradCAM is computed with respect to the final convolutional layer of the lane detector backbone. The resulting heatmap shows which spatial regions of the image most influenced the lane prediction. This is overlaid on the original image and returned with every prediction response.

For the decision CNN fallback, GradCAM is computed with respect to the predicted command class. This shows whether the model is "looking at" the road ahead (correct) or at irrelevant image regions (potential failure mode).

### 7.2 Decision Trace Logging

Every prediction generates a structured JSON log entry containing:
- Raw image path
- All intermediate measurements (offset, curvature, obstacle distance, lane confidence)
- Which decision path was taken (geometric / single-lane fallback / ML fallback)
- Final command and confidence
- Paths to generated visualizations

These logs are the primary tool for debugging and model improvement. They allow post-hoc analysis of exactly why the system made each decision on a given frame.

### 7.3 Lane Visualization Convention

Lane lines are color-coded by confidence:
- **Green** — both lanes detected, confidence > 0.85
- **Yellow** — one or both lanes detected with moderate confidence (0.65–0.85)
- **Red** — lane detected but low confidence (< 0.65) or fallback active

This gives an immediate visual signal of system health without reading numbers.

---

## 8. Monitoring and Observability

### 8.1 Prometheus Metrics

The following metrics are exposed at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `roadsage_inference_latency_seconds` | Histogram | End-to-end prediction latency |
| `roadsage_command_total` | Counter | Predictions by command type |
| `roadsage_confidence_histogram` | Histogram | Distribution of confidence scores |
| `roadsage_safety_gate_triggers_total` | Counter | How often the safety gate fires |
| `roadsage_lane_detection_failures_total` | Counter | Frames with no lanes detected |
| `roadsage_ml_fallback_activations_total` | Counter | How often the ML fallback is used |

### 8.2 Why These Metrics Matter

The `safety_gate_triggers` and `ml_fallback_activations` counters are leading indicators of system health. A sudden spike in safety gate triggers could indicate a new obstacle type the system isn't handling. A spike in ML fallback activations could indicate deteriorating lane detection (e.g., entering a section of road with faded markings).

Grafana alerts can be configured to notify when these metrics cross thresholds.

---

## 9. Edge Deployment Strategy

### 9.1 ONNX Export

All four neural models (lane detector, depth estimator, object detector, decision CNN) are exported to ONNX after training � both the CPU and GPU variant of each model. ONNX provides:
- Hardware-agnostic format
- TensorRT optimization on Jetson Nano
- Consistent inference behavior across platforms

### 9.2 Latency Budget

Target: < 100ms end-to-end on edge hardware (Raspberry Pi 4 or Jetson Nano)

| Component | Target Latency |
|---|---|
| Preprocessing | < 5ms |
| Lane Detection (UFLD v2, ONNX) | < 15ms |
| Object Detection (YOLOv8n, ONNX) | < 10ms |
| Depth Estimation (MiDaS Small, ONNX) | < 30ms |
| Decision Engine | < 2ms |
| GradCAM | < 20ms |
| Response serialization | < 5ms |
| **Total** | **< 87ms** |

GradCAM generation is the most expensive non-model step. On edge deployments where latency is critical, GradCAM generation is lazy (generated only on explicit request, not on every frame).

### 9.3 `requirements-edge.txt`

A separate requirements file strips out all training-only dependencies (PyTorch, albumentations, MLflow, etc.) and keeps only the inference dependencies (ONNX Runtime, OpenCV, FastAPI). This reduces the Docker image from ~8GB to ~1.2GB.

---

## 10. Testing Philosophy

### 10.1 What Gets Tested

Tests are divided by concern:

- **Unit tests** — individual functions: BEV transform matrix correctness, offset calculation, curvature formula, safety gate logic
- **Integration tests** — module pipelines: lane detector + BEV + geometry produces correct offset for known test images
- **End-to-end tests** — API tests with real HTTP requests using `httpx` + `TestClient`
- **Property-based tests** — hypothesis tests for decision engine: e.g., for any `offset > OFFSET_THRESHOLD`, the command must always be a correction command

### 10.2 What Is Explicitly Not Tested Here

Model accuracy (lane F1, command accuracy) is evaluated through the training evaluation scripts (`evaluate_lane.py`, `evaluate_decision.py`), not through pytest. These require actual model weights and test datasets, making them integration-level evaluation rather than unit tests.

---

## 11. Risk Register and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Lane detector fails on faded MNNIT markings | High | High | Bollard edge detection as secondary lane proxy; aggressive augmentation on faded markings; human review of pseudo-label quality |
| Tree shadows trigger false lane detections | High | High | Shadow augmentation during training; confidence thresholding; temporal consistency filter |
| Obstacle detector misses unusual objects (cycle rickshaws, animals) | High | Medium | Fine-tune active detector (`nanodet_plus_m` on CPU / `yolov8n` on GPU) on campus-specific classes; conservative obstacle distance threshold |
| Depth model scale incorrect for stop/go decisions | Medium | Medium | Calibrate depth threshold separately for `midas_small` (CPU) and `depth_anything_v2` (GPU); store as separate keys in `configs/decision_engine.yaml` |
| Pseudo-labels systematically wrong for specific road sections | High | Medium | Human-in-the-loop validation on 10% sample; per-section confidence analysis |
| MC Dropout underestimates uncertainty on OOD inputs | Medium | Low | Calibrate ECE < 0.05 on held-out set; add temperature scaling if needed |
| Edge hardware too slow for real-time | Medium | Medium | ONNX + TensorRT; selective GradCAM; profile and cut slowest components |

---

## 12. What "Done" Looks Like

The system is production-ready when all of the following are true:

1. **Correctness** — Command accuracy > 88% on manually annotated held-out MNNIT frames
2. **Safety** — STOP precision > 99% on safety-critical test cases
3. **Latency** — P95 inference latency < 100ms
4. **Reliability** — Zero crashes over a 500-frame continuous run
5. **Interpretability** — Every prediction has a correct GradCAM overlay and decision trace log
6. **Monitoring** — Prometheus metrics populated; Grafana dashboard operational
7. **Tests** — All pytest tests passing; no test failures in CI

These are not aspirational targets — they are the minimum bar for a system that is operating in a real environment and making safety-relevant decisions.
