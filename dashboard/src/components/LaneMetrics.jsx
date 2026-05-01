import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { useState, useEffect } from "react";

export function LaneMetrics({ laneOffset = 0, curvature = 0, speedKmh = 0 }) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    setHistory((prev) => [
      ...prev.slice(-49),
      { offset: Number.parseFloat(laneOffset.toFixed(3)), curvature: Number.parseFloat(curvature.toFixed(4)) },
    ]);
  }, [laneOffset, curvature]);

  let offsetStatus;
  if (Math.abs(laneOffset) > 0.3)       offsetStatus = "badge-red";
  else if (Math.abs(laneOffset) > 0.15) offsetStatus = "badge-yellow";
  else                                   offsetStatus = "badge-green";

  const tiles = [
    { label: "Lane Offset", value: `${laneOffset >= 0 ? "+" : ""}${laneOffset.toFixed(3)}m`, status: offsetStatus },
    { label: "Curvature",   value: `${curvature.toFixed(4)}`, sub: "1/m" },
    { label: "Speed",       value: speedKmh, sub: "km/h" },
  ];

  return (
    <div className="card">
      <div className="section-title">Lane Metrics</div>

      <div style={{ display: "flex", gap: "10px", marginBottom: "18px" }}>
        {tiles.map((t) => (
          <div key={t.label} className="stat-tile">
            <div className="stat-label">{t.label}</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "4px" }}>
              <span className="stat-value">{t.value}</span>
              {t.sub && <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>{t.sub}</span>}
            </div>
            {t.status && <span className={`badge ${t.status}`} style={{ marginTop: "6px", fontSize: "10px" }}>offset</span>}
          </div>
        ))}
      </div>

      <div style={{ marginBottom: "6px" }}>
        <span style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 600 }}>Lane Offset History</span>
      </div>
      <ResponsiveContainer width="100%" height={100}>
        <AreaChart data={history} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="offsetGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="var(--accent-blue)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="t" hide />
          <YAxis domain={[-0.5, 0.5]} hide />
          <Tooltip
            formatter={(v) => [`${v.toFixed(3)} m`, "offset"]}
            contentStyle={{ background: "var(--bg-elevated)", border: "1px solid var(--border-strong)", borderRadius: "8px", fontSize: "12px" }}
            labelStyle={{ display: "none" }}
          />
          <ReferenceLine y={0} stroke="var(--border-strong)" strokeDasharray="3 3" />
          <Area type="monotone" dataKey="offset" stroke="var(--accent-blue)" fill="url(#offsetGrad)" strokeWidth={1.8} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export default LaneMetrics;
