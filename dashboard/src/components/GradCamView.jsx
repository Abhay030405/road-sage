'use client'

import { useState } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'

function GradCamView() {
  const { latestResult, selectedHistoryItem } = useRoadSage()
  const [opacity, setOpacity] = useState(0.6)
  const [enabled, setEnabled] = useState(true)

  const displayResult = selectedHistoryItem || latestResult
  const gradcamSrc = displayResult?.gradcam_base64
    ? `data:image/jpeg;base64,${displayResult.gradcam_base64}` : null
  const laneVizSrc = displayResult?.lane_viz_base64
    ? `data:image/jpeg;base64,${displayResult.lane_viz_base64}` : null

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden flex flex-col h-full">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-border to-transparent" />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-rs-border">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-rs-red" strokeWidth={1.5} />
          <span className="text-xs font-semibold tracking-[0.2em] uppercase text-rs-muted">GradCAM</span>
          {selectedHistoryItem && (
            <span className="text-[10px] tracking-widest uppercase text-rs-amber">Historical</span>
          )}
        </div>
        <button
          onClick={() => setEnabled(!enabled)}
          className={`flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded border tracking-widest uppercase font-medium transition-colors
            ${enabled ? 'border-rs-amber text-rs-amber' : 'border-rs-border text-rs-muted'}`}
        >
          {enabled ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
          {enabled ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Image */}
      <div className="flex-1 relative bg-black overflow-hidden min-h-24">
        {enabled && gradcamSrc && (
          <img src={gradcamSrc} alt="GradCAM" className="w-full h-full object-cover" style={{ opacity }} />
        )}
        {enabled && !gradcamSrc && laneVizSrc && (
          <img src={laneVizSrc} alt="Lane viz" className="w-full h-full object-cover" />
        )}
        {(!enabled || (!gradcamSrc && !laneVizSrc)) && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs tracking-[0.25em] uppercase text-rs-border">
              {enabled ? 'Awaiting Frame' : 'Disabled'}
            </span>
          </div>
        )}
      </div>

      {/* Opacity slider */}
      {enabled && (
        <div className="px-4 py-2.5 border-t border-rs-border">
          <div className="flex justify-between text-[11px] font-mono text-rs-muted mb-2">
            <span>OPACITY</span>
            <span>{Math.round(opacity * 100)}%</span>
          </div>
          <input
            type="range" min="0" max="1" step="0.1" value={opacity}
            onChange={e => setOpacity(Number.parseFloat(e.target.value))}
            className="w-full h-px accent-amber-600 cursor-pointer"
          />
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 border-t border-rs-border">
        <span className="text-[11px] font-mono text-rs-muted">
          LAYER <span className="text-rs-text font-semibold">features.12</span>
        </span>
      </div>
    </div>
  )
}

export default GradCamView
