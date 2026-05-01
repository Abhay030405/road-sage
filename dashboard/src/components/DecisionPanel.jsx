const DECISION_META = {
  LANE_KEEP:         { label: "Lane Keep",     badge: "badge-green",  icon: "→", glow: "rgba(52,211,153,0.18)" },
  LANE_CHANGE_LEFT:  { label: "Lane Change ←", badge: "badge-yellow", icon: "←", glow: "rgba(251,191,36,0.18)" },
  LANE_CHANGE_RIGHT: { label: "Lane Change →", badge: "badge-yellow", icon: "→", glow: "rgba(251,191,36,0.18)" },
  SLOW_DOWN:         { label: "Slow Down",     badge: "badge-red",    icon: "⚠", glow: "rgba(248,113,113,0.18)" },
  STOP:              { label: "STOP",          badge: "badge-red",    icon: "■", glow: "rgba(248,113,113,0.3)"  },
};

const BORDER_COLOR = {
  "badge-green":  "var(--accent-green)",
  "badge-yellow": "var(--accent-yellow)",
  "badge-red":    "var(--accent-red)",
};

export function DecisionPanel({ decision = "LANE_KEEP", confidence = 0 }) {
  const meta = DECISION_META[decision] ?? { label: decision, badge: "badge-blue", icon: "?", glow: "transparent" };
  const borderColor = BORDER_COLOR[meta.badge] ?? "var(--accent-blue)";

  return (
    <div className="card">
      <div className="section-title">Current Decision</div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "16px",
          padding: "18px",
          background: meta.glow,
          borderRadius: "var(--r-sm)",
          border: `1.5px solid ${borderColor}`,
          transition: "all 0.3s ease",
        }}
      >
        <div
          style={{
            width: 52, height: 52,
            borderRadius: 12,
            background: `color-mix(in srgb, ${borderColor} 15%, transparent)`,
            border: `1px solid color-mix(in srgb, ${borderColor} 40%, transparent)`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "24px",
            flexShrink: 0,
          }}
        >
          {meta.icon}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: "17px", fontWeight: 700, color: borderColor, letterSpacing: "-0.2px" }}>
            {meta.label}
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "3px" }}>
            {(confidence * 100).toFixed(1)}% confidence
          </div>
        </div>
        <span className={`badge ${meta.badge}`}>Active</span>
      </div>
    </div>
  );
}

export default DecisionPanel;
