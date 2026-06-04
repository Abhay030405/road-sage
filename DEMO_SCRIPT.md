# RoadSage — Demo Script (30 seconds)

## Pre-Demo Checklist
- [ ] `docker-compose up --build` ran successfully
- [ ] `localhost:8000/docs` opens in browser
- [ ] `localhost:3000` opens in browser (dashboard)
- [ ] `rgb/` folder has images
- [ ] TestDemo button visible on dashboard (bottom-right)

---

## 30-Second Live Demo Flow

### Step 1 — Show System Health (5 seconds)
Open `localhost:3000`. Point out:
- Header: **"RoadSage — MNNIT Campus Vision Navigation"**
- **SystemHealth** panel: model status indicators, uptime
- Connection status: **CONNECTED** (green)

### Step 2 — Start Simulation (5 seconds)
Click **"Start Simulation"** button (bottom-right).  
Dashboard comes alive:
- **DecisionPanel** cycles: `FORWARD → LEFT → FORWARD → STOP` (red pulse)
- **ConfidenceMeter** gauge moves in real-time
- **DecisionHistory** fills up with color-coded rows
- **LaneMetrics** shows offset and curvature values changing

### Step 3 — Show STOP Hazard Detection (5 seconds)
When `STOP` appears (red panel, pulsing):  
Point out: *"Safety gate triggered — obstacle detected in path"*  
Show hazard reason in the red strip below the command.

### Step 4 — Show API (10 seconds)
Open `localhost:8000/docs` in browser.  
Expand **POST /api/v1/predict**.  
Click **"Try it out"** → upload any `rgb_image_*.png` → **Execute**.  
Show the JSON response: `command`, `confidence`, `decision_path`, `latency_ms`.

### Step 5 — Show GradCAM (5 seconds)
Click a `STOP` row in **DecisionHistory**.  
**GradCamView** updates to show that frame.  
*"This heatmap shows exactly which pixels caused the STOP decision."*

---

## Fallback Plan
If backend is down:
- Show **TestDemo** simulation (no backend needed)
- Show pre-recorded `outputs/` visualizations
- Show `notebooks/04_decision_logic_analysis.ipynb` for decision engine walkthrough

---

## Key Talking Points
- **"1,302 real MNNIT campus images — not a generic dataset"**
- **"5-layer defense in depth — STOP is never delayed"**
- **"Self-supervised pseudo-labeling — no manual annotation needed"**
- **"Every decision is explainable via GradCAM"**
- **"CPU-only, ONNX-optimized, runs on Raspberry Pi"**
