export function VideoFeed({ fps = 0, detections = [], surface = "unknown" }) {
  const surfaceLabel = surface.replaceAll("_", " ");

  return (
    <div className="card">
      <div className="section-title">
        Camera Feed
        <div style={{ marginLeft: "auto", display: "flex", gap: "6px" }}>
          <span className="badge badge-cyan">{fps} fps</span>
          <span className="badge badge-indigo">{surfaceLabel}</span>
        </div>
      </div>

      {/* Video frame */}
      <div
        style={{
          width: "100%",
          aspectRatio: "16/9",
          background: "linear-gradient(160deg, #07101f 0%, #0a1628 100%)",
          borderRadius: "var(--r-sm)",
          border: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Grid overlay for visual depth */}
        <svg width="100%" height="100%" style={{ position: "absolute", inset: 0, opacity: 0.06 }}>
          <defs>
            <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
              <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#4f8ef7" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", color: "var(--text-muted)", zIndex: 1 }}>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M15 10l4.553-2.276A1 1 0 0121 8.723v6.554a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
          </svg>
          <span style={{ fontSize: "12px" }}>Awaiting stream…</span>
        </div>

        {/* Detection overlays */}
        {detections.map((d) => (
          <div
            key={d.id}
            style={{
              position: "absolute",
              left: `${(d.bbox[0] / 640) * 100}%`,
              top:  `${(d.bbox[1] / 360) * 100}%`,
              width: `${((d.bbox[2] - d.bbox[0]) / 640) * 100}%`,
              height: `${((d.bbox[3] - d.bbox[1]) / 360) * 100}%`,
              border: "1.5px solid var(--accent-cyan)",
              borderRadius: "4px",
              boxShadow: "0 0 8px rgba(34,211,238,0.3)",
            }}
          >
            <span
              style={{
                background: "rgba(34,211,238,0.85)",
                color: "#000",
                fontSize: "9px",
                fontWeight: 700,
                padding: "1px 5px",
                borderRadius: "3px",
                position: "absolute",
                top: "-17px",
                left: 0,
                whiteSpace: "nowrap",
              }}
            >
              {d.label} {(d.confidence * 100).toFixed(0)}%
            </span>
          </div>
        ))}

        {/* Detection count badge */}
        {detections.length > 0 && (
          <div
            style={{
              position: "absolute",
              bottom: "10px",
              right: "10px",
              background: "rgba(6,11,24,0.8)",
              border: "1px solid var(--border-strong)",
              borderRadius: "6px",
              padding: "4px 8px",
              fontSize: "11px",
              color: "var(--accent-cyan)",
              fontWeight: 600,
            }}
          >
            {detections.length} object{detections.length === 1 ? "" : "s"}
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoFeed;
