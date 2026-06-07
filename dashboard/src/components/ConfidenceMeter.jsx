'use client'

import { Activity } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'

// Speedometer geometry
const CX = 150, CY = 128, R = 90
const START = 225          // degrees from top, clockwise (7:30 o'clock)
const SWEEP = 270          // total degrees of sweep

function xy(deg, r = R) {
  const rad = ((deg - 90) * Math.PI) / 180
  return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)]
}

function arc(startDeg, sweepDeg, r = R) {
  if (sweepDeg <= 0) return ''
  const s = Math.min(sweepDeg, 359.9)
  const [x1, y1] = xy(startDeg, r)
  const [x2, y2] = xy(startDeg + s, r)
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${s > 180 ? 1 : 0} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`
}

function getColor(pct) {
  if (pct >= 80) return '#3d8b5f'
  if (pct >= 60) return '#c47d15'
  return '#c0172b'
}

function ConfidenceMeter() {
  const { latestResult } = useRoadSage()
  const pct         = Math.round((latestResult?.confidence ?? 0) * 100)
  const color       = getColor(pct)
  const valueSweep  = (pct / 100) * SWEEP
  const needleDeg   = START + valueSweep
  const [nx, ny]    = xy(needleDeg, R - 8)
  const [tx, ty]    = xy(needleDeg + 180, 18)

  let status, label
  if (pct >= 80)      { status = 'OPTIMAL';  label = 'HIGH CONFIDENCE' }
  else if (pct >= 60) { status = 'CAUTION';  label = 'MODERATE' }
  else                { status = 'CRITICAL'; label = 'LOW — STOPPING' }

  // Tick marks: every 10% = 27°, major every 25%
  const ticks = Array.from({ length: 11 }, (_, i) => {
    const deg    = START + (i / 10) * SWEEP
    const major  = i % 5 === 0
    const [ox, oy] = xy(deg, R + 6)
    const [ix, iy] = xy(deg, R - (major ? 16 : 8))
    return { ox, oy, ix, iy, major, val: i * 10 }
  })

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden h-full flex flex-col">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-rs-border">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-rs-red" strokeWidth={1.5} />
          <span className="text-[10px] font-semibold tracking-[0.2em] uppercase text-rs-muted">Confidence</span>
        </div>
        <span className="text-[9px] font-bold tracking-widest" style={{ color }}>{status}</span>
      </div>

      {/* Speedometer */}
      <div className="flex flex-1 justify-center items-center py-1">
      <svg viewBox="45 18 210 190" width="140" height="126" style={{ display: 'block' }}>
        <defs>
          <filter id="glow-c" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-needle" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ── Background track ── */}
        <path d={arc(START, SWEEP)} fill="none" stroke="#1c1c1c" strokeWidth="14" strokeLinecap="round" />

        {/* ── Colored value arc ── */}
        {valueSweep > 0 && (
          <path
            d={arc(START, valueSweep)}
            fill="none"
            stroke={color}
            strokeWidth="14"
            strokeLinecap="round"
            filter="url(#glow-c)"
          />
        )}


        {/* ── Tick marks ── */}
        {ticks.map((t) => (
          <line key={t.val}
            x1={t.ox.toFixed(2)} y1={t.oy.toFixed(2)}
            x2={t.ix.toFixed(2)} y2={t.iy.toFixed(2)}
            stroke={t.val <= pct ? color : '#2e2e2e'}
            strokeWidth={t.major ? 1.8 : 0.9}
            strokeLinecap="round"
          />
        ))}

        {/* ── End labels: 0 and 100 ── */}
        {(() => {
          const [x0, y0] = xy(START, R + 24)
          const [x1, y1] = xy(START + SWEEP, R + 24)
          return (
            <>
              <text x={x0.toFixed(2)} y={y0.toFixed(2)} textAnchor="middle"
                fill="#3a3a3a" fontSize="8" fontFamily="monospace">0</text>
              <text x={x1.toFixed(2)} y={y1.toFixed(2)} textAnchor="middle"
                fill="#3a3a3a" fontSize="8" fontFamily="monospace">100</text>
            </>
          )
        })()}


        {/* ── Needle ── */}
        <line
          x1={tx.toFixed(2)} y1={ty.toFixed(2)}
          x2={nx.toFixed(2)} y2={ny.toFixed(2)}
          stroke={color} strokeWidth="2.2" strokeLinecap="round"
          filter="url(#glow-needle)"
        />

        {/* ── Pivot cap ── */}
        <circle cx={CX} cy={CY} r="7" fill="#111111" stroke="#2a2a2a" strokeWidth="1" />
        <circle cx={CX} cy={CY} r="3.5" fill={color} filter="url(#glow-needle)" />

        {/* ── Center value ── */}
        <text x={CX} y={CY + 36} textAnchor="middle"
          fill={color} fontSize="26" fontFamily="monospace" fontWeight="900">
          {pct}%
        </text>
        <text x={CX} y={CY + 50} textAnchor="middle"
          fill="#3a3a3a" fontSize="7" fontFamily="monospace" letterSpacing="3">
          SIGNAL
        </text>
      </svg>
      </div>

      {/* Status label */}
      <div className="text-center text-[9px] font-semibold tracking-[0.2em] uppercase pb-3 -mt-1" style={{ color }}>
        {label}
      </div>
    </div>
  )
}

export default ConfidenceMeter
