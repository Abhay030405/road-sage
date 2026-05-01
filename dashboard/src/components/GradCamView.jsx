export function GradCamView({ imageUrl = null }) {
  return (
    <div className="card">
      <div className="section-title">
        Grad-CAM Explainability
        <span className="badge badge-indigo" style={{ marginLeft: "auto" }}>Last frame</span>
      </div>

      <div
        style={{
          width: "100%",
          aspectRatio: "16/9",
          background: "linear-gradient(160deg, #07101f, #0d1a2e)",
          borderRadius: "var(--r-sm)",
          border: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "10px",
          color: "var(--text-muted)",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {imageUrl ? (
          <img src={imageUrl} alt="Grad-CAM heatmap" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <>
            {/* Placeholder heatmap gradient */}
            <svg width="100%" height="100%" style={{ position: "absolute", inset: 0, opacity: 0.12 }}>
              <defs>
                <radialGradient id="heat1" cx="60%" cy="50%" r="40%">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity="1" />
                  <stop offset="60%" stopColor="#fbbf24" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="heat2" cx="35%" cy="55%" r="25%">
                  <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="transparent" stopOpacity="0" />
                </radialGradient>
              </defs>
              <rect width="100%" height="100%" fill="url(#heat1)" />
              <rect width="100%" height="100%" fill="url(#heat2)" />
            </svg>

            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ position: "relative" }}>
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
            </svg>
            <span style={{ fontSize: "12px", position: "relative" }}>Grad-CAM overlay not available</span>
          </>
        )}
      </div>

      <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "10px", lineHeight: "1.6" }}>
        Highlights the image regions that most influenced the lane decision.
      </p>
    </div>
  );
}

export default GradCamView;
