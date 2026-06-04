import { useState, useEffect } from 'react'
import { useRoadSage } from './context/RoadSageContext'

const FAKE_SEQUENCE = [
  {
    command: 'FORWARD', confidence: 0.91, decision_path: 'geometric',
    lane_offset_m: 0.05, curvature_inv_m: 0.001,
    left_lane_detected: true, right_lane_detected: true,
    hazard_detected: false, surface_class: 'clean',
    latency_ms: { lane: 28, scene: 25, decision: 4, total: 57 },
  },
  {
    command: 'LEFT', confidence: 0.83, decision_path: 'geometric',
    lane_offset_m: 0.38, curvature_inv_m: 0.002,
    left_lane_detected: true, right_lane_detected: true,
    hazard_detected: false, surface_class: 'clean',
    latency_ms: { lane: 31, scene: 24, decision: 3, total: 58 },
  },
  {
    command: 'FORWARD', confidence: 0.88, decision_path: 'geometric',
    lane_offset_m: 0.12, curvature_inv_m: 0.006,
    left_lane_detected: true, right_lane_detected: true,
    hazard_detected: false, surface_class: 'clean',
    latency_ms: { lane: 29, scene: 27, decision: 4, total: 60 },
  },
  {
    command: 'STOP', confidence: 1.0, decision_path: 'safety_gate',
    lane_offset_m: 0.0, curvature_inv_m: 0.0,
    left_lane_detected: true, right_lane_detected: true,
    hazard_detected: true, hazard_reason: 'person detected in path',
    surface_class: 'clean',
    latency_ms: { lane: 30, scene: 26, decision: 2, total: 58 },
  },
]

function TestDemo() {
  const { addResult } = useRoadSage()
  const [running, setRunning] = useState(false)
  const [idx, setIdx] = useState(0)

  useEffect(() => {
    if (!running) return
    const interval = setInterval(() => {
      const item = FAKE_SEQUENCE[idx % FAKE_SEQUENCE.length]
      addResult(
        { ...item, frame_id: idx, timestamp: new Date().toISOString() },
        idx,
        14.5
      )
      setIdx(i => i + 1)
    }, 800)
    return () => clearInterval(interval)
  }, [running, idx, addResult])

  return (
    <div className="fixed bottom-4 right-4 bg-rs-panel border border-rs-border
                    rounded-lg p-3 flex items-center gap-3 z-50">
      <span className="text-xs text-rs-muted">Demo Mode</span>
      <button
        type="button"
        onClick={() => setRunning(!running)}
        className={`text-xs px-3 py-1.5 rounded font-medium
          ${running ? 'bg-rs-red text-white' : 'bg-rs-green text-white'}`}
      >
        {running ? 'Stop' : 'Start'} Simulation
      </button>
    </div>
  )
}

export default TestDemo
