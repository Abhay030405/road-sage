'use client'

import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'
import { Cpu, Activity, Wifi, WifiOff } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'
import { API_URL, HEALTH_POLL_INTERVAL_MS } from '../constants'

function useHealthPolling() {
  const [healthData, setHealthData] = useState(null)
  const [pollError,  setPollError]  = useState(false)

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

  return { healthData, pollError }
}

function StatCell({ label, value, color = 'text-rs-text' }) {
  return (
    <div className="bg-black rounded px-2.5 py-1.5 text-center">
      <div className={`text-sm font-mono font-bold tabular-nums leading-none ${color}`}>{value}</div>
      <div className="text-[9px] tracking-widest uppercase text-rs-muted mt-1">{label}</div>
    </div>
  )
}
StatCell.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  color: PropTypes.string,
}

function ModelRow({ label, ok }) {
  return (
    <div className="flex items-center gap-2 py-1.5 border-b border-rs-border last:border-0">
      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${ok ? 'bg-rs-green' : 'bg-rs-red'}`} />
      <span className="text-xs text-rs-muted flex-1 truncate">{label}</span>
      <span className={`text-[11px] font-bold tracking-widest ${ok ? 'text-rs-green' : 'text-rs-red'}`}>
        {ok ? 'OK' : 'ERR'}
      </span>
    </div>
  )
}
ModelRow.propTypes = {
  label: PropTypes.string.isRequired,
  ok:    PropTypes.bool.isRequired,
}

export function SystemStats() {
  const { sessionFps, history } = useRoadSage()
  const { healthData, pollError } = useHealthPolling()

  const recentLatencies = history.slice(0, 20).map(h => h.latency_ms?.total).filter(Boolean)
  const avgLatency = recentLatencies.length
    ? recentLatencies.reduce((a, b) => a + b, 0) / recentLatencies.length : 0
  const p95Latency = recentLatencies.length
    ? [...recentLatencies].sort((a, b) => a - b)[Math.floor(recentLatencies.length * 0.95)] : 0

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden h-full flex flex-col">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />

      <div className="flex items-center justify-between px-4 py-2 border-b border-rs-border">
        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-rs-red" strokeWidth={1.5} />
          <span className="text-xs font-semibold tracking-[0.2em] uppercase text-rs-muted">System</span>
        </div>
        <div className="flex items-center gap-1.5">
          {pollError
            ? <WifiOff className="w-3 h-3 text-rs-red" />
            : <Wifi    className="w-3 h-3 text-rs-green" />
          }
          <span className={`text-[11px] font-bold tracking-widest ${pollError ? 'text-rs-red' : 'text-rs-green'}`}>
            {pollError ? 'OFFLINE' : 'ONLINE'}
          </span>
          {healthData && (
            <span className="text-[11px] font-mono text-rs-muted ml-1">
              {Math.floor(healthData.uptime_seconds / 60)}m{Math.floor(healthData.uptime_seconds % 60)}s
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-1.5 p-2.5 flex-1 content-center">
        <StatCell label="FPS"     value={sessionFps.toFixed(1)}
          color={sessionFps > 5 ? 'text-rs-green' : 'text-rs-amber'} />
        <StatCell label="Avg Lat" value={avgLatency ? `${avgLatency.toFixed(0)}ms` : '--'}
          color={avgLatency < 100 ? 'text-rs-green' : 'text-rs-amber'} />
        <StatCell label="P95"     value={p95Latency ? `${p95Latency.toFixed(0)}ms` : '--'}
          color={p95Latency < 200 ? 'text-rs-green' : 'text-rs-red'} />
        <StatCell label="Frames"  value={healthData?.frames_processed ?? history.length} />
      </div>
    </div>
  )
}

export function ModelStatus() {
  const { healthData, pollError } = useHealthPolling()

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden h-full flex flex-col">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />

      <div className="flex items-center justify-between px-4 py-2 border-b border-rs-border">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-rs-red" strokeWidth={1.5} />
          <span className="text-xs font-semibold tracking-[0.2em] uppercase text-rs-muted">Models</span>
        </div>
        <div className="flex items-center gap-1">
          <span className={`w-1.5 h-1.5 rounded-full ${pollError ? 'bg-rs-red' : 'bg-rs-green animate-pulse'}`} />
        </div>
      </div>

      <div className="px-4 py-2 flex-1 flex flex-col justify-center">
        <ModelRow label="Lane Detector · UFLD v2"  ok={healthData?.models?.lane_detector  ?? false} />
        <ModelRow label="Scene · NanoDet + MiDaS"  ok={healthData?.models?.scene_analyzer ?? false} />
        <ModelRow label="Decision · MobileNetV3"   ok={healthData?.models?.ml_fallback    ?? false} />
      </div>
    </div>
  )
}

function SystemHealth() {
  return null
}

export default SystemHealth
