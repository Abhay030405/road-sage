const DECISION_COLORS = {
  LANE_KEEP:         "badge-green",
  LANE_CHANGE_LEFT:  "badge-yellow",
  LANE_CHANGE_RIGHT: "badge-yellow",
  SLOW_DOWN:         "badge-red",
  STOP:              "badge-red",
};

export function DecisionHistory({ history = [] }) {
  return (
    <div className="card" style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div className="section-title">
        Decision History
        {history.length > 0 && (
          <span className="badge badge-blue" style={{ marginLeft: "auto" }}>{history.length}</span>
        )}
      </div>

      {history.length === 0 ? (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", padding: "24px 0", color: "var(--text-muted)" }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span style={{ fontSize: "12px" }}>No decisions yet</span>
        </div>
      ) : (
        <ul style={{ listStyle: "none", display: "flex", flexDirection: "column", gap: "5px", maxHeight: "240px", overflowY: "auto", paddingRight: "2px" }}>
          {history.map((entry, i) => (
            <li
              key={entry.ts}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "7px 10px",
                background: i === 0 ? "var(--bg-elevated)" : "var(--bg-card-2)",
                borderRadius: "8px",
                border: "1px solid var(--border)",
                transition: "opacity 0.3s",
                opacity: i === 0 ? 1 : Math.max(0.45, 1 - i * 0.04),
              }}
            >
              <span className={`badge ${DECISION_COLORS[entry.decision] ?? "badge-blue"}`} style={{ fontSize: "10px" }}>
                {entry.decision.replaceAll("_", " ")}
              </span>
              <span style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-secondary)" }}>
                {(entry.confidence * 100).toFixed(0)}%
              </span>
              <span style={{ fontSize: "10px", color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
                {new Date(entry.ts).toLocaleTimeString()}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default DecisionHistory;
