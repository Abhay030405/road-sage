'use client'

import { GitBranch } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'
import { formatOffset, formatCurvature } from '../utils/helpers'

function LaneBar({ label, detected }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="w-1.5 h-10 rounded-full transition-colors duration-300"
        style={{ background: detected ? '#3d8b5f' : '#1e1e1e' }}
      />
      <span className="text-[11px] font-bold tracking-widest uppercase"
        style={{ color: detected ? '#3d8b5f' : '#3a3a3a' }}>
        {label}
      </span>
    </div>
  )
}

function Row({ label, value, color = 'text-rs-text' }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-rs-border last:border-0">
      <span className="text-[10px] tracking-wider uppercase text-rs-muted">{label}</span>
      <span className={`text-xs font-mono font-semibold tabular-nums ${color}`}>{value}</span>
    </div>
  )
}

function LaneMetrics() {
  const { latestResult } = useRoadSage()

  if (!latestResult) {
    return (
      <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden">
        <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />
        <div className="flex items-center gap-2 px-4 py-3 border-b border-rs-border">
          <GitBranch className="w-4 h-4 text-rs-red" strokeWidth={1.5} />
          <span className="text-xs font-semibold tracking-[0.2em] uppercase text-rs-muted">Lane Metrics</span>
        </div>
        <div className="p-4 space-y-2">
          {[1, 2, 3, 4].map(i => <div key={i} className="h-5 bg-rs-border rounded animate-pulse" />)}
        </div>
      </div>
    )
  }

  const offsetAbs = Math.abs(latestResult?.lane_offset_m || 0)

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden h-full flex flex-col">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />

      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-rs-border">
        <GitBranch className="w-4 h-4 text-rs-red" strokeWidth={1.5} />
        <span className="text-xs font-semibold tracking-[0.2em] uppercase text-rs-muted">Lane Metrics</span>
      </div>

      {/* Body: lane bars left, metrics right */}
      <div className="flex flex-1">

        {/* Left — lane diagram */}
        <div className="flex items-center justify-center gap-4 px-4 py-3 bg-black border-r border-rs-border flex-shrink-0 self-stretch">
          <LaneBar label="L" detected={latestResult?.left_lane_detected} />
          <LaneBar label="C" detected={!!latestResult?.center_lane_points?.length} />
          <LaneBar label="R" detected={latestResult?.right_lane_detected} />
        </div>

        {/* Right — metrics */}
        <div className="flex-1 px-3 py-0.5">
          <Row
            label="Offset"
            value={formatOffset(latestResult.lane_offset_m)}
            color={offsetAbs > 0.3 ? 'text-rs-amber' : 'text-rs-green'}
          />
          <Row
            label="Curvature"
            value={formatCurvature(latestResult.curvature_inv_m)}
            color="text-rs-text"
          />
          <Row
            label="Surface"
            value={latestResult?.surface_class || '--'}
            color={['pothole', 'waterlogged'].includes(latestResult?.surface_class) ? 'text-rs-red' : 'text-rs-green'}
          />
          <Row
            label="Obstacle"
            value={latestResult?.nearest_obstacle_class || 'None'}
            color={latestResult?.nearest_obstacle_class ? 'text-rs-amber' : 'text-rs-muted'}
          />
        </div>

      </div>
    </div>
  )
}

export default LaneMetrics
