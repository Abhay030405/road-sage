import { useState } from 'react'
import { useRoadSage } from '../context/RoadSageContext'

function getFrameContent(enabled, gradcamSrc, laneVizSrc, opacity) {
  if (enabled && gradcamSrc) {
    return (
      <img
        src={gradcamSrc}
        alt="GradCAM overlay"
        className="w-full h-full object-cover"
        style={{ opacity }}
      />
    )
  }
  if (enabled && laneVizSrc) {
    return (
      <img
        src={laneVizSrc}
        alt="Lane visualization"
        className="w-full h-full object-cover"
      />
    )
  }
  const message = enabled ? 'Waiting for frame...' : 'GradCAM disabled'
  return (
    <div className="absolute inset-0 flex items-center justify-center
                    text-rs-muted text-xs text-center p-2">
      {message}
    </div>
  )
}

function GradCamView() {
  const { latestResult, selectedHistoryItem } = useRoadSage()
  const [opacity, setOpacity] = useState(0.6)
  const [enabled, setEnabled] = useState(true)

  const displayResult = selectedHistoryItem || latestResult
  const gradcamSrc = displayResult?.gradcam_base64
    ? `data:image/jpeg;base64,${displayResult.gradcam_base64}`
    : null
  const laneVizSrc = displayResult?.lane_viz_base64
    ? `data:image/jpeg;base64,${displayResult.lane_viz_base64}`
    : null

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border p-4 flex flex-col">

      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-rs-muted font-medium">GradCAM View</span>
        <button
          onClick={() => setEnabled(!enabled)}
          className={`text-xs px-2 py-0.5 rounded border
            ${enabled ? 'border-rs-amber text-rs-amber' : 'border-rs-border text-rs-muted'}`}
        >
          {enabled ? 'ON' : 'OFF'}
        </button>
      </div>

      <div className="flex-1 relative bg-rs-bg rounded overflow-hidden min-h-24">
        {getFrameContent(enabled, gradcamSrc, laneVizSrc, opacity)}
      </div>

      {enabled && (
        <div className="mt-2">
          <div className="flex justify-between text-xs text-rs-muted mb-1">
            <span>Overlay opacity</span>
            <span>{Math.round(opacity * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={opacity}
            onChange={e => setOpacity(Number.parseFloat(e.target.value))}
            className="w-full accent-amber-500"
          />
        </div>
      )}

      <div className="mt-2 text-xs text-rs-muted">
        Layer: <span className="text-rs-text font-mono">features.12</span>
        {selectedHistoryItem && (
          <span className="ml-2 text-rs-amber">(historical frame)</span>
        )}
      </div>

    </div>
  )
}

export default GradCamView
