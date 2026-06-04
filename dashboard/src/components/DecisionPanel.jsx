import { useRoadSage } from '../context/RoadSageContext'
import { COMMAND_COLORS, COMMAND_ICONS, DECISION_PATH_LABELS } from '../constants'
import { formatConfidence } from '../utils/helpers'

function DecisionPanel() {
  const { latestResult } = useRoadSage()

  if (!latestResult) {
    return (
      <div className="bg-rs-panel rounded-lg border border-rs-border p-4 h-40
                      animate-pulse flex items-center justify-center">
        <div className="w-24 h-12 bg-rs-border rounded" />
      </div>
    )
  }

  const command = latestResult?.command || 'STOP'
  const colors = COMMAND_COLORS[command] ?? COMMAND_COLORS.STOP
  const isStop = command === 'STOP'

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden">

      <div className={`${colors.bg} flex items-center justify-center gap-3 py-6
                      ${isStop ? 'animate-pulse' : ''}`}>
        <span className="text-5xl font-black text-white">
          {COMMAND_ICONS[command] ?? '?'}
        </span>
        <span className="text-4xl font-black text-white tracking-wider">
          {command}
        </span>
      </div>

      <div className="px-4 py-3 flex justify-between items-center">
        <div>
          <div className="text-xs text-rs-muted mb-0.5">Confidence</div>
          <div className="text-lg font-bold" style={{ color: colors.hex }}>
            {formatConfidence(latestResult?.confidence || 0)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-rs-muted mb-0.5">Decision Path</div>
          <div className="text-sm text-rs-text">
            {DECISION_PATH_LABELS[latestResult?.decision_path] || '--'}
          </div>
        </div>
      </div>

      {latestResult?.hazard_detected && (
        <div className="bg-rs-red bg-opacity-20 border-t border-rs-red px-4 py-2
                        flex items-center gap-2">
          <span className="text-rs-red text-xs font-medium">⚠ HAZARD:</span>
          <span className="text-rs-red text-xs">
            {latestResult.hazard_reason || 'obstacle detected'}
          </span>
        </div>
      )}

    </div>
  )
}

export default DecisionPanel
