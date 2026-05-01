import { useState, useEffect } from "react";
import VideoFeed from "./components/VideoFeed";
import DecisionPanel from "./components/DecisionPanel";
import LaneMetrics from "./components/LaneMetrics";
import GradCamView from "./components/GradCamView";
import ConfidenceMeter from "./components/ConfidenceMeter";
import DecisionHistory from "./components/DecisionHistory";
import SystemHealth from "./components/SystemHealth";

const MOCK_STATE = {
  decision: "LANE_KEEP",
  confidence: 0.91,
  lane_offset: 0.12,
  curvature: 0.003,
  speed_kmh: 62,
  fps: 28,
  detections: [
    { id: 1, label: "car",   confidence: 0.95, bbox: [120, 200, 240, 320] },
    { id: 2, label: "truck", confidence: 0.87, bbox: [400, 180, 560, 340] },
  ],
  surface: "asphalt_dry",
  depth_available: true,
  system: {
    cpu: 43,
    gpu: 71,
    ram: 58,
    inference_ms: 34,
    model_version: "ufld-v2-r34",
    uptime_s: 3720,
  },
  history: [],
};

export default function App() {
  const [state, setState] = useState(MOCK_STATE);
  const [wsStatus, setWsStatus] = useState("disconnected");

  useEffect(() => {
    let ws;
    function connect() {
      ws = new WebSocket(`ws://${globalThis.location.hostname}:8000/ws/stream`);
      ws.onopen  = () => setWsStatus("connected");
      ws.onclose = () => { setWsStatus("disconnected"); setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          setState((prev) => ({
            ...data,
            history: [
              { ts: Date.now(), decision: data.decision, confidence: data.confidence },
              ...(prev.history || []).slice(0, 49),
            ],
          }));
        } catch { /* ignore malformed frames */ }
      };
    }
    connect();
    return () => ws?.close();
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", padding: "20px 24px", gap: "20px", maxWidth: "1600px", margin: "0 auto" }}>

      {/* ── Top bar ─────────────────────────────────────────── */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingBottom: "16px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <div style={{
            width: 38, height: 38, borderRadius: 10,
            background: "linear-gradient(135deg, var(--accent-blue), var(--accent-indigo))",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "18px", boxShadow: "0 0 24px rgba(79,142,247,0.25)",
          }}>🛣️</div>
          <div>
            <div style={{ fontSize: "16px", fontWeight: 800, letterSpacing: "-0.3px", color: "var(--text-primary)" }}>
              RoadSage
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", letterSpacing: "0.04em" }}>
              Autonomous Driving Intelligence
            </div>
          </div>
          <span className={`badge ${wsStatus === "connected" ? "badge-green badge-live" : "badge-red"}`} style={{ marginLeft: 8 }}>
            {wsStatus === "connected" ? "Live" : "Disconnected"}
          </span>
        </div>
        <SystemHealth system={state.system} />
      </header>

      {/* ── Main grid ───────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 340px", gap: "18px", flex: 1 }}>

        {/* Column 1 — Camera */}
        <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          <VideoFeed fps={state.fps} detections={state.detections} surface={state.surface} />
          <GradCamView />
        </div>

        {/* Column 2 — Lane / Scene */}
        <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          <LaneMetrics laneOffset={state.lane_offset} curvature={state.curvature} speedKmh={state.speed_kmh} />
        </div>

        {/* Column 3 — Decision / Telemetry */}
        <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          <DecisionPanel decision={state.decision} confidence={state.confidence} />
          <ConfidenceMeter confidence={state.confidence} />
          <DecisionHistory history={state.history} />
        </div>
      </div>
    </div>
  );
}

