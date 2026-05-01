import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from "recharts";

function getColor(c) {
  if (c >= 0.8) return "var(--accent-green)";
  if (c >= 0.5) return "var(--accent-yellow)";
  return "var(--accent-red)";
}

function getLabel(c) {
  if (c >= 0.85) return { text: "High",   badge: "badge-green"  };
  if (c >= 0.6)  return { text: "Medium", badge: "badge-yellow" };
  return           { text: "Low",    badge: "badge-red"    };
}

export function ConfidenceMeter({ confidence = 0 }) {
  const pct   = Math.round(confidence * 100);
  const color = getColor(confidence);
  const meta  = getLabel(confidence);
  const data  = [{ value: pct }];

  return (
    <div className="card">
      <div className="section-title">Decision Confidence</div>

      <div style={{ position: "relative", height: 150 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%" cy="55%"
            innerRadius="62%" outerRadius="88%"
            startAngle={210} endAngle={-30}
            data={data}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
            <RadialBar
              dataKey="value"
              cornerRadius={8}
              fill={color}
              background={{ fill: "var(--bg-elevated)" }}
            />
          </RadialBarChart>
        </ResponsiveContainer>

        <div
          style={{
            position: "absolute",
            top: "54%", left: "50%",
            transform: "translate(-50%, -50%)",
            textAlign: "center",
            pointerEvents: "none",
          }}
        >
          <div style={{ fontSize: "28px", fontWeight: 800, color, letterSpacing: "-1px", lineHeight: 1 }}>
            {pct}<span style={{ fontSize: "14px" }}>%</span>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "4px" }}>
        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Fusion score</span>
        <span className={`badge ${meta.badge}`}>{meta.text}</span>
      </div>

      {/* Segmented ticks */}
      <div style={{ display: "flex", gap: "3px", marginTop: "10px" }}>
        {Array.from({ length: 20 }, (_, i) => (
          <div
            key={`tick-${i}`}
            style={{
              flex: 1,
              height: "4px",
              borderRadius: "2px",
              background: i < Math.round(pct / 5) ? color : "var(--bg-elevated)",
              transition: "background 0.4s",
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default ConfidenceMeter;
