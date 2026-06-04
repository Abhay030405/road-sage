import { useRoadSage } from '../context/RoadSageContext'
import { COMMAND_COLORS, COMMAND_ICONS, DECISION_PATH_LABELS } from '../constants'
import { formatConfidence, timeAgo } from '../utils/helpers'

function DecisionHistory() {
  const { history, setSelectedHistoryItem, selectedHistoryItem } = useRoadSage()

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border p-4 flex flex-col">

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1">
          <span className="text-xs text-rs-muted font-medium">Decision History</span>
          <span className="text-xs text-rs-border">({history.length})</span>
        </div>
        {selectedHistoryItem && (
          <button
            onClick={() => setSelectedHistoryItem(null)}
            className="text-xs text-rs-amber hover:text-rs-text"
          >
            Clear selection
          </button>
        )}
      </div>

      <div
        className="flex-1 overflow-y-auto space-y-1 max-h-48"
        style={{ scrollbarWidth: 'thin' }}
      >
        {history.length === 0 ? (
          <div className="text-center text-rs-muted text-xs py-4">
            No decisions yet — waiting for stream...
          </div>
        ) : (
          history.map((item, idx) => {
            const colors = COMMAND_COLORS[item.command] ?? COMMAND_COLORS.STOP
            const isSelected = selectedHistoryItem?.frameId === item.frameId
            return (
              <button
                key={item.frameId ?? idx}
                type="button"
                onClick={() => setSelectedHistoryItem(isSelected ? null : item)}
                className={`w-full text-left flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer
                  border transition-colors
                  ${isSelected
                    ? 'border-rs-amber bg-rs-amber bg-opacity-10'
                    : 'border-transparent hover:border-rs-border hover:bg-rs-bg'
                  }`}
              >
                <span className={`${colors.bg} text-white text-xs font-bold
                                  px-1.5 py-0.5 rounded min-w-16 text-center`}>
                  {COMMAND_ICONS[item.command]} {item.command}
                </span>

                <div className="flex-1 min-w-0">
                  <div className="flex justify-between text-xs">
                    <span className="text-rs-muted truncate">
                      {DECISION_PATH_LABELS[item.decision_path] || item.decision_path}
                    </span>
                    <span style={{ color: colors.hex }}>
                      {formatConfidence(item.confidence)}
                    </span>
                  </div>
                  <div className="text-xs text-rs-muted opacity-60 mt-0.5">
                    {timeAgo(item.receivedAt)}
                  </div>
                </div>
              </button>
            )
          })
        )}
      </div>

    </div>
  )
}

export default DecisionHistory
