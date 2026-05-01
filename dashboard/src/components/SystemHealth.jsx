function barColor(v) {
  if (v >= 85) return "var(--accent-red)";
  if (v >= 60) return "var(--accent-yellow)";
  return "var(--accent-green)";
}

function ResourceBar({ label, value }) {
  const color = barColor(value);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <span style={{ width: "32px", fontSize: "11px", fontWeight: 600, color: "var(--text-muted)", letterSpacing: "0.04em" }}>
        {label}
      </span>
      <div className="progress-track" style={{ flex: 1 }}>
        <div className="progress-fill" style={{ width: `${value}%`, background: color }} />
      </div>
      <span style={{ width: "32px", textAlign: "right", fontSize: "12px", fontWeight: 600, color }}>
        {value}%
      </span>
    </div>
  );
}

export function SystemHealth({ system = {} }) {
  const { cpu = 0, gpu = 0, ram = 0, inference_ms = 0, model_version = "—", uptime_s = 0 } = system;
  const uptime = `${Math.floor(uptime_s / 3600)}h ${Math.floor((uptime_s % 3600) / 60)}m`;
  let inferenceColor;
  if (inference_ms > 60)      inferenceColor = "var(--accent-red)";
  else if (inference_ms > 40) inferenceColor = "var(--accent-yellow)";
  else                        inferenceColor = "var(--accent-green)";

  return (
    <div className="card" style={{ minWidth: "280px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" strokeWidth="2">
            <rect x="4" y="4" width="16" height="16" rx="2" />
            <rect x="9" y="9" width="6" height="6" />
            <path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" />
          </svg>
          <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-secondary)" }}>System</span>
        </div>
        <span className="badge badge-indigo" style={{ fontSize: "10px" }}>{model_version}</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "14px" }}>
        <ResourceBar label="CPU" value={cpu} />
        <ResourceBar label="GPU" value={gpu} />
        <ResourceBar label="RAM" value={ram} />
      </div>

      <div className="divider" />

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px" }}>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Inference </span>
          <span style={{ fontWeight: 700, color: inferenceColor }}>{inference_ms} ms</span>
        </div>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Uptime </span>
          <span style={{ fontWeight: 700, color: "var(--text-primary)" }}>{uptime}</span>
        </div>
      </div>
    </div>
  );
}

export default SystemHealth;
