'use client'

import { ArrowUp, ArrowLeft, ArrowRight, OctagonX } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'
import { COMMAND_COLORS, DECISION_PATH_LABELS } from '../constants'
import { formatConfidence } from '../utils/helpers'

const COMMAND_LUCIDE = {
  FORWARD: ArrowUp,
  LEFT:    ArrowLeft,
  RIGHT:   ArrowRight,
  STOP:    OctagonX,
}

function DecisionPanel() {
  const { latestResult } = useRoadSage()

  if (!latestResult) {
    return (
      <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden">
        <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />
        <div className="p-5 h-36 flex items-center justify-center animate-pulse">
          <div className="w-32 h-10 bg-rs-border rounded" />
        </div>
      </div>
    )
  }

  const command = latestResult?.command || 'STOP'
  const colors  = COMMAND_COLORS[command] ?? COMMAND_COLORS.STOP
  const Icon    = COMMAND_LUCIDE[command] ?? OctagonX
  const isStop  = command === 'STOP'

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden">

      {/* Top accent — command color */}
      <div className="h-px" style={{ background: `linear-gradient(to right, transparent, ${colors.hex}, transparent)` }} />

      {/* Command area */}
      <div
        className="relative flex items-center gap-5 px-5 py-5"
        style={{ background: `radial-gradient(ellipse at 25% 50%, ${colors.hex}10 0%, transparent 65%)` }}
      >
        {/* Icon with glow */}
        <div className="relative flex-shrink-0">
          <div className="absolute inset-0 rounded-full blur-2xl opacity-25" style={{ background: colors.hex }} />
          <div
            className={`relative w-14 h-14 rounded-full flex items-center justify-center ${isStop ? 'animate-pulse' : ''}`}
            style={{ border: `1px solid ${colors.hex}35`, background: `${colors.hex}0d` }}
          >
            <Icon className="w-7 h-7" style={{ color: colors.hex }} strokeWidth={1.5} />
          </div>
        </div>

        {/* Command + path */}
        <div className="flex-1">
          <div className="text-[9px] tracking-[0.3em] uppercase text-rs-muted mb-1">Command</div>
          <div className="text-2xl font-black tracking-[0.12em] leading-none" style={{ color: colors.hex }}>
            {command}
          </div>
          <div className="text-[9px] tracking-widest uppercase text-rs-muted mt-2">
            {DECISION_PATH_LABELS[latestResult?.decision_path] || '--'}
          </div>
        </div>

        {/* Confidence */}
        <div className="text-right flex-shrink-0">
          <div className="text-[9px] tracking-[0.2em] uppercase text-rs-muted mb-1">Confidence</div>
          <div className="text-2xl font-mono font-bold tabular-nums" style={{ color: colors.hex }}>
            {formatConfidence(latestResult?.confidence || 0)}
          </div>
        </div>
      </div>

      {/* Hazard bar */}
      {latestResult?.hazard_detected && (
        <div
          className="flex items-center gap-2.5 px-5 py-2 border-t"
          style={{ borderColor: '#c0172b25', background: '#c0172b08' }}
        >
          <span className="text-[9px] font-bold tracking-[0.25em] uppercase text-rs-red">⚠ Hazard</span>
          <span className="text-[9px] text-rs-muted">{latestResult.hazard_reason || 'obstacle detected'}</span>
        </div>
      )}

      {/* Bottom accent */}
      <div className="h-px" style={{ background: `linear-gradient(to right, transparent, ${colors.hex}25, transparent)` }} />
    </div>
  )
}

export default DecisionPanel
