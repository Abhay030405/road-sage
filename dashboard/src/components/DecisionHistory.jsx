'use client'

import { List } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'
import { COMMAND_COLORS, COMMAND_ICONS, DECISION_PATH_LABELS } from '../constants'
import { formatConfidence, timeAgo } from '../utils/helpers'

function DecisionHistory() {
  const { history, setSelectedHistoryItem, selectedHistoryItem } = useRoadSage()

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden flex flex-col h-full">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-rs-border">
        <div className="flex items-center gap-2">
          <List className="w-4 h-4 text-rs-red" strokeWidth={1.5} />
          <span className="text-xs font-semibold tracking-[0.2em] uppercase text-rs-muted">Event Log</span>
          <span className="text-[10px] font-mono text-rs-border">{history.length}</span>
        </div>
        {selectedHistoryItem && (
          <button
            onClick={() => setSelectedHistoryItem(null)}
            className="text-[10px] tracking-widest uppercase text-rs-amber"
          >
            Clear
          </button>
        )}
      </div>

      {/* Log */}
      <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: 'thin' }}>
        {history.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <span className="text-xs tracking-[0.25em] uppercase text-rs-border">Awaiting events...</span>
          </div>
        ) : (
          <div className="divide-y divide-rs-border">
            {history.map((item, idx) => {
              const colors = COMMAND_COLORS[item.command] ?? COMMAND_COLORS.STOP
              const isSelected = selectedHistoryItem?.frameId === item.frameId
              return (
                <button
                  key={item.frameId ?? idx}
                  type="button"
                  onClick={() => setSelectedHistoryItem(isSelected ? null : item)}
                  className={`w-full text-left flex items-center gap-2.5 px-4 py-2.5 transition-colors
                    ${isSelected ? 'bg-rs-bg' : 'hover:bg-rs-bg'}`}
                  style={isSelected ? { borderLeft: `2px solid ${colors.hex}` } : { borderLeft: '2px solid transparent' }}
                >
                  <span
                    className="text-[10px] font-bold tracking-widest px-1.5 py-0.5 rounded flex-shrink-0"
                    style={{ color: colors.hex, background: `${colors.hex}12`, border: `1px solid ${colors.hex}25` }}
                  >
                    {COMMAND_ICONS[item.command]} {item.command}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-mono text-rs-muted truncate">
                        {DECISION_PATH_LABELS[item.decision_path] || item.decision_path}
                      </span>
                      <span className="text-[11px] font-mono font-bold tabular-nums flex-shrink-0 ml-1" style={{ color: colors.hex }}>
                        {formatConfidence(item.confidence)}
                      </span>
                    </div>
                    <div className="text-[10px] font-mono text-rs-border mt-0.5">{timeAgo(item.receivedAt)}</div>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

export default DecisionHistory
