import { useRoadSage } from '../context/RoadSageContext'
import { formatOffset, formatCurvature } from '../utils/helpers'

function MetricRow({ label, value, valueColor = 'text-rs-text' }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-rs-border last:border-0">
      <span className="text-xs text-rs-muted">{label}</span>
      <span className={`text-sm font-mono font-medium ${valueColor}`}>{value}</span>
    </div>
  )
}

function LaneIndicator({ label, detected }) {
  return (
    <div className="flex items-center gap-1">
      <span className={`text-xs ${detected ? 'text-rs-green' : 'text-rs-red'}`}>
        {detected ? '✓' : '✗'}
      </span>
      <span className="text-xs text-rs-muted">{label}</span>
    </div>
  )
}

function LaneMetrics() {
  const { latestResult } = useRoadSage()

  if (!latestResult) {
    return (
      <div className="bg-rs-panel rounded-lg border border-rs-border p-4">
        <div className="text-xs text-rs-muted mb-3 font-medium">Lane Metrics</div>
        <div className="space-y-3">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-8 bg-rs-border rounded animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border p-4">
      <div className="text-xs text-rs-muted mb-3 font-medium">Lane Metrics</div>

      <div className="flex justify-around mb-3 py-2 bg-rs-bg rounded border border-rs-border">
        <LaneIndicator label="Left" detected={latestResult?.left_lane_detected} />
        <LaneIndicator label="Center" detected={!!latestResult?.center_lane_points?.length} />
        <LaneIndicator label="Right" detected={latestResult?.right_lane_detected} />
      </div>

      <div>
        <MetricRow
          label="Lateral Offset"
          value={formatOffset(latestResult.lane_offset_m)}
          valueColor={
            Math.abs(latestResult?.lane_offset_m || 0) > 0.3
              ? 'text-rs-amber' : 'text-rs-green'
          }
        />
        <MetricRow
          label="Curvature"
          value={formatCurvature(latestResult.curvature_inv_m)}
        />
        <MetricRow
          label="Surface"
          value={latestResult?.surface_class || '--'}
          valueColor={
            ['pothole', 'waterlogged'].includes(latestResult?.surface_class)
              ? 'text-rs-red' : 'text-rs-green'
          }
        />
        <MetricRow
          label="Nearest Obstacle"
          value={latestResult?.nearest_obstacle_class || 'None'}
          valueColor={latestResult?.nearest_obstacle_class ? 'text-rs-amber' : 'text-rs-muted'}
        />
      </div>
    </div>
  )
}

export default LaneMetrics
