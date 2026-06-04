import { RoadSageProvider, useRoadSage } from './context/RoadSageContext'
import TestDemo from './TestDemo'
import useWebSocket from './hooks/useWebSocket'
import VideoFeed from './components/VideoFeed'
import DecisionPanel from './components/DecisionPanel'
import ConfidenceMeter from './components/ConfidenceMeter'
import LaneMetrics from './components/LaneMetrics'
import GradCamView from './components/GradCamView'
import DecisionHistory from './components/DecisionHistory'
import SystemHealth from './components/SystemHealth'

function ConnectionBanner({ connectionState, reconnectAttempt }) {
  if (connectionState === 'connected') return null

  const isRed = connectionState === 'disconnected' || connectionState === 'error'

  let message
  if (connectionState === 'connecting') {
    message = 'Connecting to RoadSage backend...'
  } else if (connectionState === 'disconnected') {
    message = `Disconnected — reconnecting (attempt ${reconnectAttempt})...`
  } else {
    message = 'Connection error — retrying...'
  }

  return (
    <div className={`fixed top-0 left-0 right-0 z-50 py-2 px-4 text-center text-sm font-medium text-white
                     ${isRed ? 'bg-rs-red' : 'bg-rs-amber'}`}>
      {message}
    </div>
  )
}

function DashboardContent() {
  const { addResult } = useRoadSage()
  const { connectionState, reconnectAttempt } = useWebSocket(addResult)

  return (
    <div className="min-h-screen bg-rs-bg text-rs-text flex flex-col">

      <header className="border-b border-rs-border px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-rs-green">RoadSage</span>
          <span className="text-xs text-rs-muted bg-rs-panel px-2 py-1 rounded">
            MNNIT Campus — Vision Navigation
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-rs-muted">
          <span className={connectionState === 'connected' ? 'text-rs-green' : 'text-rs-red'}>
            ● {connectionState.toUpperCase()}
          </span>
        </div>
      </header>

      <ConnectionBanner connectionState={connectionState} reconnectAttempt={reconnectAttempt} />

      <main className="flex-1 p-4 grid gap-4" style={{ gridTemplateRows: 'auto auto' }}>

        <div className="grid gap-4" style={{ gridTemplateColumns: '60% 1fr' }}>
          <VideoFeed />
          <div className="flex flex-col gap-4">
            <DecisionPanel />
            <ConfidenceMeter />
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <LaneMetrics />
          <GradCamView />
          <DecisionHistory />
          <SystemHealth />
        </div>

      </main>

      <footer className="border-t border-rs-border px-6 py-2 text-xs text-rs-muted flex justify-between">
        <span>RoadSage v1.0 — MNNIT Allahabad</span>
        <span>Vision-Based Intelligent Driving Decision Engine</span>
      </footer>

      <TestDemo />
    </div>
  )
}

function App() {
  return (
    <RoadSageProvider>
      <DashboardContent />
    </RoadSageProvider>
  )
}

export default App
