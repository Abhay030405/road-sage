import { useState, useEffect } from 'react'
import { useRoadSage } from '../context/RoadSageContext'
import { API_URL, HEALTH_POLL_INTERVAL_MS } from '../constants'

function StatusDot({ ok, label }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-rs-green' : 'bg-rs-red'}`} />
      <span className="text-xs text-rs-muted">{label}</span>
    </div>
  )
}

function MetricBox({ label, value, color = 'text-rs-text' }) {
  return (
    <div className="bg-rs-bg rounded p-2 text-center">
      <div className={`text-lg font-mono font-bold ${color}`}>{value}</div>
      <div className="text-xs text-rs-muted mt-0.5">{label}</div>
    </div>
  )
}

function SystemHealth() {
  const { sessionFps, history } = useRoadSage()
  const [healthData, setHealthData] = useState(null)
  const [pollError, setPollError] = useState(false)

  useEffect(() => {
    const poll = () => {
      fetch(`${API_URL}/api/v1/health`)
        .then(r => r.json())
        .then(data => { setHealthData(data); setPollError(false) })
        .catch(() => setPollError(true))
    }
    poll()
    const id = setInterval(poll, HEALTH_POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  const recentLatencies = history
    .slice(0, 20)
    .map(h => h.latency_ms?.total)
    .filter(Boolean)
  const avgLatency = recentLatencies.length
    ? recentLatencies.reduce((a, b) => a + b, 0) / recentLatencies.length
    : 0
  const p95Latency = recentLatencies.length
    ? [...recentLatencies].sort((a, b) => a - b)[Math.floor(recentLatencies.length * 0.95)]
    : 0

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border p-4 flex flex-col gap-3">

      <div className="flex items-center justify-between">
        <span className="text-xs text-rs-muted font-medium">System Health</span>
        <span className={`text-xs ${pollError ? 'text-rs-red' : 'text-rs-green'}`}>
          {pollError ? '● API offline' : '● API online'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <MetricBox
          label="Session FPS"
          value={sessionFps.toFixed(1)}
          color={sessionFps > 5 ? 'text-rs-green' : 'text-rs-amber'}
        />
        <MetricBox
          label="Avg Latency"
          value={avgLatency ? `${avgLatency.toFixed(0)}ms` : '--'}
          color={avgLatency < 100 ? 'text-rs-green' : 'text-rs-amber'}
        />
        <MetricBox
          label="P95 Latency"
          value={p95Latency ? `${p95Latency.toFixed(0)}ms` : '--'}
          color={p95Latency < 200 ? 'text-rs-green' : 'text-rs-red'}
        />
        <MetricBox
          label="Frames"
          value={healthData?.frames_processed ?? history.length}
        />
      </div>

      <div className="space-y-1.5">
        <div className="text-xs text-rs-muted mb-1">Model Status</div>
        <StatusDot
          ok={healthData?.models?.lane_detector ?? false}
          label="Lane Detector (UFLD v2)"
        />
        <StatusDot
          ok={healthData?.models?.scene_analyzer ?? false}
          label="Scene Analyzer (NanoDet + MiDaS)"
        />
        <StatusDot
          ok={healthData?.models?.ml_fallback ?? false}
          label="Decision CNN (MobileNetV3)"
        />
      </div>

      {healthData && (
        <div className="text-xs text-rs-muted border-t border-rs-border pt-2">
          Uptime: {Math.floor(healthData.uptime_seconds / 60)}m{' '}
          {Math.floor(healthData.uptime_seconds % 60)}s
        </div>
      )}

    </div>
  )
}

export default SystemHealth
