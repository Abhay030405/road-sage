'use client'

import { useEffect, useState } from 'react'
import PropTypes from 'prop-types'
import { Gauge, Wifi, WifiOff, Clock, ArrowUp, ArrowLeft, ArrowRight, OctagonX } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'
import useWebSocket from '../hooks/useWebSocket'
import VideoFeed from '../components/VideoFeed'
import LaneMetrics from '../components/LaneMetrics'
import GradCamView from '../components/GradCamView'
import DecisionHistory from '../components/DecisionHistory'
import { SystemStats, ModelStatus } from '../components/SystemHealth'

import TestDemo from '../TestDemo'
import ConfidenceMeter from '../components/ConfidenceMeter'
import { COMMAND_COLORS, DECISION_PATH_LABELS } from '../constants'

const RELOAD_AFTER_ATTEMPTS = 5
const RELOAD_COUNTDOWN_S    = 15

const CMD_ICON = {
  FORWARD: ArrowUp,
  LEFT:    ArrowLeft,
  RIGHT:   ArrowRight,
  STOP:    OctagonX,
}

// ─── Inline stat cards ────────────────────────────────────────────────────────

function CommandCard() {
  const { latestResult } = useRoadSage()
  const command = latestResult?.command ?? 'STOP'
  const colors  = COMMAND_COLORS[command] ?? COMMAND_COLORS.STOP
  const Icon    = CMD_ICON[command] ?? OctagonX

  return (
    <div
      className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden flex items-center gap-4 px-4 py-2.5"
      style={{ borderLeft: `3px solid ${colors.hex}` }}
    >
      <div
        className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
        style={{ background: `${colors.hex}18`, border: `1px solid ${colors.hex}35` }}
      >
        <Icon className="w-5 h-5" style={{ color: colors.hex }} strokeWidth={2} />
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-[10px] tracking-[0.25em] uppercase text-rs-muted mb-0.5">Command</div>
        <div className="text-lg font-bold tracking-wider leading-none" style={{ color: colors.hex }}>
          {command}
        </div>
        {latestResult?.decision_path && (
          <div className="text-[10px] font-mono text-rs-muted mt-1 truncate">
            {DECISION_PATH_LABELS[latestResult.decision_path] || latestResult.decision_path}
          </div>
        )}
      </div>

    </div>
  )
}


// ─── Live clock hook ──────────────────────────────────────────────────────────

function useLiveClock() {
  const [time, setTime] = useState('')
  useEffect(() => {
    const tick = () => setTime(new Date().toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])
  return time
}

// ─── Header ───────────────────────────────────────────────────────────────────

function Header({ connectionState, sessionFps }) {
  const time        = useLiveClock()
  const isConnected = connectionState === 'connected'

  return (
    <header className="relative bg-rs-panel border-b border-rs-border">
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-rs-red to-transparent opacity-60" />

      <div className="px-8 py-4 flex items-center justify-between">

        {/* ── Brand ── */}
        <div className="flex items-center gap-4">
          <svg viewBox="0 0 46 46" width="46" height="46" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ flexShrink: 0 }}>
            <rect width="46" height="46" rx="11" fill="#0a0a0a"/>
            <rect x="0.75" y="0.75" width="44.5" height="44.5" rx="10.25" stroke="#c0172b" strokeWidth="1" strokeOpacity="0.7"/>
            <line x1="9"  y1="39" x2="23" y2="12" stroke="#c0172b" strokeWidth="2.2" strokeLinecap="round"/>
            <line x1="37" y1="39" x2="23" y2="12" stroke="#c0172b" strokeWidth="2.2" strokeLinecap="round"/>
            <line x1="7"  y1="29" x2="39" y2="29" stroke="#2e2e2e" strokeWidth="0.9" strokeLinecap="round"/>
            <line x1="23" y1="23" x2="23" y2="27" stroke="#484848" strokeWidth="1.6" strokeLinecap="round"/>
            <line x1="23" y1="30" x2="23" y2="34" stroke="#353535" strokeWidth="1.6" strokeLinecap="round"/>
            <line x1="23" y1="36" x2="23" y2="39" stroke="#242424" strokeWidth="1.6" strokeLinecap="round"/>
            <circle cx="23" cy="12" r="5"   fill="#c0172b" fillOpacity="0.12"/>
            <circle cx="23" cy="12" r="2.5" fill="#c0172b" fillOpacity="0.35"/>
            <circle cx="23" cy="12" r="1.2" fill="#c0172b"/>
            <line x1="5"  y1="19" x2="11" y2="19" stroke="#3a3a3a" strokeWidth="1" strokeLinecap="round"/>
            <line x1="5"  y1="23" x2="10" y2="23" stroke="#2a2a2a" strokeWidth="1" strokeLinecap="round"/>
            <line x1="41" y1="19" x2="35" y2="19" stroke="#3a3a3a" strokeWidth="1" strokeLinecap="round"/>
            <line x1="41" y1="23" x2="36" y2="23" stroke="#2a2a2a" strokeWidth="1" strokeLinecap="round"/>
          </svg>
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold tracking-widest uppercase text-rs-text">
                Road<span className="text-rs-red">Sage</span>
              </span>
              <span className="text-xs font-mono text-rs-muted border border-rs-border px-1.5 py-0.5 rounded">v1.0</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <div className="h-px w-4 bg-gradient-to-r from-transparent to-rs-red opacity-70" />
              <span className="text-[10px] font-semibold tracking-[0.3em] uppercase text-rs-muted">MNNIT Allahabad</span>
              <span className="text-rs-red opacity-60 text-[8px]">◆</span>
              <span className="text-[10px] font-semibold tracking-[0.3em] uppercase text-rs-red opacity-75">Vision Navigation</span>
              <div className="h-px w-4 bg-gradient-to-l from-transparent to-rs-red opacity-70" />
            </div>
          </div>
        </div>

        {/* ── Centre ── */}
        <div className="absolute left-1/2 -translate-x-1/2 flex flex-col items-center gap-1">
          <span className="text-xs tracking-[0.25em] uppercase text-rs-muted font-medium">Autonomous Driving System</span>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-rs-red opacity-60" />
            <span className="w-8 h-px bg-rs-border" />
            <span className="text-xs font-mono text-rs-muted">MNNIT-CAMPUS-01</span>
            <span className="w-8 h-px bg-rs-border" />
            <span className="w-1.5 h-1.5 rounded-full bg-rs-red opacity-60" />
          </div>
        </div>

        {/* ── Right stats ── */}
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2 text-rs-muted">
            <Clock className="w-3.5 h-3.5" strokeWidth={1.5} />
            <span className="font-mono text-sm text-rs-text tabular-nums">{time}</span>
          </div>
          <div className="w-px h-8 bg-rs-border" />
          <div className="flex items-center gap-2">
            <Gauge className="w-4 h-4 text-rs-muted" strokeWidth={1.5} />
            <div>
              <div className="text-xs text-rs-muted uppercase tracking-wider leading-none mb-0.5">FPS</div>
              <div className="text-sm font-mono font-bold text-rs-text tabular-nums leading-none">{sessionFps.toFixed(1)}</div>
            </div>
          </div>
          <div className="w-px h-8 bg-rs-border" />
          <div className="flex items-center gap-2.5">
            {isConnected
              ? <Wifi    className="w-4 h-4 text-rs-green" strokeWidth={1.5} />
              : <WifiOff className="w-4 h-4 text-rs-red"   strokeWidth={1.5} />
            }
            <div>
              <div className="text-xs text-rs-muted uppercase tracking-wider leading-none mb-0.5">Backend</div>
              <div className={`text-sm font-bold leading-none ${isConnected ? 'text-rs-green' : 'text-rs-red'}`}>
                {connectionState.toUpperCase()}
              </div>
            </div>
            {isConnected && <span className="w-2 h-2 rounded-full bg-rs-green animate-pulse" />}
          </div>
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />
    </header>
  )
}
Header.propTypes = {
  connectionState: PropTypes.string.isRequired,
  sessionFps:      PropTypes.number.isRequired,
}

// ─── Reload banner ────────────────────────────────────────────────────────────

function ConnectionBanner({ reloadCountdown, onCancelReload }) {
  if (reloadCountdown === null) return null
  return (
    <div className="fixed top-0 left-0 right-0 z-50 py-2 px-4 flex items-center justify-center gap-4
                    text-sm font-medium text-white bg-rs-red">
      <span>Still disconnected — reloading page in {reloadCountdown}s...</span>
      <div className="flex gap-2">
        <button onClick={() => globalThis.location.reload()}
          className="bg-white text-rs-red text-xs font-bold px-2 py-0.5 rounded">
          Reload Now
        </button>
        <button onClick={onCancelReload}
          className="border border-white text-white text-xs px-2 py-0.5 rounded">
          Cancel
        </button>
      </div>
    </div>
  )
}
ConnectionBanner.propTypes = {
  reloadCountdown: PropTypes.number,
  onCancelReload:  PropTypes.func.isRequired,
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { addResult, sessionFps, frameId } = useRoadSage()
  const { connectionState, reconnectAttempt, sendFrame } = useWebSocket(addResult)
  const [reloadCountdown, setReloadCountdown] = useState(null)

  useEffect(() => {
    if (reconnectAttempt >= RELOAD_AFTER_ATTEMPTS && reloadCountdown === null)
      setReloadCountdown(RELOAD_COUNTDOWN_S)
  }, [reconnectAttempt])

  useEffect(() => {
    if (connectionState === 'connected') setReloadCountdown(null)
  }, [connectionState])

  useEffect(() => {
    if (reloadCountdown === null) return
    if (reloadCountdown === 0) { globalThis.location.reload(); return }
    const t = setTimeout(() => setReloadCountdown(c => c - 1), 1000)
    return () => clearTimeout(t)
  }, [reloadCountdown])

  return (
    <div className="min-h-screen bg-rs-bg text-rs-text flex flex-col">

      <Header connectionState={connectionState} sessionFps={sessionFps} frameId={frameId} />

      <ConnectionBanner
        reloadCountdown={reloadCountdown}
        onCancelReload={() => setReloadCountdown(null)}
      />

      {/* Top cards row */}
      <div className="px-4 pt-2 pb-1 grid grid-cols-5 gap-3 w-full h-[178px]">
        <SystemStats />
        <DecisionHistory />
        <ConfidenceMeter />
        <LaneMetrics />
        <ModelStatus />
      </div>

      <main className="flex-1 p-4 cockpit-grid grid gap-5" style={{ gridTemplateRows: 'auto auto' }}>

        {/* Vision System */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <div className="w-0.5 h-3 bg-rs-red rounded-full" />
            <span className="text-[8px] tracking-[0.4em] uppercase text-rs-muted">Vision System</span>
            <div className="flex-1 h-px bg-rs-border opacity-40" />
          </div>
          <div className="grid gap-4" style={{ gridTemplateColumns: '3fr 2fr' }}>
            <VideoFeed />
            <div className="flex flex-col gap-4 h-full">
              <CommandCard />
              <div className="flex-1 min-h-0">
                <GradCamView />
              </div>
            </div>
          </div>
        </div>

      </main>

      <footer className="relative border-t border-rs-border bg-rs-panel">
        {/* Top accent line */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-rs-red to-transparent opacity-50" />

        <div className="px-8 py-3 flex items-center justify-between gap-6">

          {/* ── Left: Brand ── */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-px h-5 bg-rs-red opacity-70" />
              <div>
                <div className="text-[10px] font-bold tracking-[0.3em] uppercase text-rs-text">
                  Road<span className="text-rs-red">Sage</span>
                  <span className="ml-1.5 text-[8px] font-mono border border-rs-border px-1 py-px rounded text-rs-muted">v1.0</span>
                </div>
                <div className="text-[8px] tracking-[0.25em] uppercase text-rs-muted mt-px">MNNIT Allahabad · ECE Dept.</div>
              </div>
            </div>
          </div>

          {/* ── Centre: Status strip ── */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rs-green animate-pulse" />
              <span className="text-[9px] tracking-[0.2em] uppercase text-rs-muted">Neural Pipeline</span>
            </div>
            <div className="w-px h-3 bg-rs-border" />
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rs-amber" />
              <span className="text-[9px] tracking-[0.2em] uppercase text-rs-muted">Safety Gate</span>
            </div>
            <div className="w-px h-3 bg-rs-border" />
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-rs-green animate-pulse" />
              <span className="text-[9px] tracking-[0.2em] uppercase text-rs-muted">WebSocket</span>
            </div>
            <div className="w-px h-3 bg-rs-border" />
            <span className="text-[9px] font-mono tracking-widest text-rs-red uppercase">● LIVE</span>
          </div>

          {/* ── Right: Tech stack ── */}
          <div className="flex items-center gap-2">
            {['Next.js', 'FastAPI', 'PyTorch', 'OpenCV'].map(tech => (
              <span key={tech}
                className="text-[8px] font-mono tracking-wider px-1.5 py-0.5 rounded border border-rs-border text-rs-muted">
                {tech}
              </span>
            ))}
            <div className="w-px h-3 bg-rs-border ml-1" />
            <span className="text-[9px] tracking-[0.2em] uppercase text-rs-muted">
              Vision-Based Driving Engine
            </span>
          </div>

        </div>

        {/* Bottom scan line */}
        <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent opacity-30" />
      </footer>

      <TestDemo sendFrame={sendFrame} />
    </div>
  )
}
