'use client'

import { useRef, useEffect, useState } from 'react'
import { Camera, Layers } from 'lucide-react'
import { useRoadSage } from '../context/RoadSageContext'

function drawLaneLines(ctx, points, color, lineWidth = 3) {
  if (!points || points.length === 0) return
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1])
  ctx.stroke()
}

function HudCorner({ pos }) {
  const map = {
    tl: 'top-3 left-3 border-t border-l',
    tr: 'top-3 right-3 border-t border-r',
    bl: 'bottom-3 left-3 border-b border-l',
    br: 'bottom-3 right-3 border-b border-r',
  }
  return <div className={`absolute w-5 h-5 border-rs-red opacity-50 ${map[pos]}`} />
}

function VideoFeed() {
  const { latestResult } = useRoadSage()
  const canvasRef = useRef(null)
  const [showOverlay, setShowOverlay] = useState(true)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    if (latestResult?.lane_viz_base64 && showOverlay) {
      const img = new Image()
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        if (latestResult.left_lane_points)
          drawLaneLines(ctx, latestResult.left_lane_points, latestResult.left_lane_detected ? '#3d8b5f' : '#c0172b')
        if (latestResult.right_lane_points)
          drawLaneLines(ctx, latestResult.right_lane_points, latestResult.right_lane_detected ? '#3d8b5f' : '#c0172b')
      }
      img.src = `data:image/jpeg;base64,${latestResult.lane_viz_base64}`
    } else {
      ctx.fillStyle = '#000000'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#2a2a2a'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('NO SIGNAL', canvas.width / 2, canvas.height / 2)
    }
  }, [latestResult, showOverlay])

  const bothLanes = latestResult?.left_lane_detected && latestResult?.right_lane_detected

  return (
    <div className="bg-black rounded-lg border border-rs-border overflow-hidden flex flex-col h-full">
      <div className="h-px bg-gradient-to-r from-transparent via-rs-red to-transparent opacity-40" />

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-rs-border bg-rs-panel">
        <div className="flex items-center gap-2.5">
          <Camera className="w-3.5 h-3.5 text-rs-red" strokeWidth={1.5} />
          <span className="text-[10px] font-semibold tracking-[0.25em] uppercase text-rs-muted">Forward Camera</span>
          {latestResult && (
            <div className="flex items-center gap-1.5 ml-1">
              <span className="w-1.5 h-1.5 rounded-full bg-rs-red animate-pulse" />
              <span className="text-[9px] font-bold tracking-widest text-rs-red">LIVE</span>
            </div>
          )}
        </div>
        <div className="flex gap-1.5">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`flex items-center gap-1 text-[9px] px-2 py-1 rounded border tracking-widest uppercase font-medium transition-colors
              ${showOverlay ? 'border-rs-green text-rs-text' : 'border-rs-border text-rs-muted'}`}
            style={showOverlay ? { backgroundColor: 'rgba(61,139,95,0.15)' } : {}}
          >
            <Layers className="w-3 h-3" /> Lane
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative bg-black">
        <canvas ref={canvasRef} width={640} height={480} className="w-full h-full object-contain" />

        <HudCorner pos="tl" />
        <HudCorner pos="tr" />
        <HudCorner pos="bl" />
        <HudCorner pos="br" />

        {latestResult && (
          <>
            <div className="absolute top-4 left-8 space-y-0.5">
              <div className="text-[9px] font-mono text-rs-muted">
                FRM <span className="text-rs-text">#{String(latestResult.frame_id || 0).padStart(5, '0')}</span>
              </div>
              <div className="text-[9px] font-mono text-rs-muted">
                CAM <span className="text-rs-text">FRONT-01</span>
              </div>
            </div>
            <div className="absolute top-4 right-8 text-right space-y-0.5">
              <div className="text-[9px] font-mono text-rs-muted">
                LAT <span className={latestResult.latency_ms?.total < 100 ? 'text-rs-green' : 'text-rs-amber'}>
                  {latestResult.latency_ms?.total?.toFixed(0) || '--'}ms
                </span>
              </div>
              <div className="text-[9px] font-mono text-rs-muted">
                {new Date(latestResult.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </>
        )}

        {!latestResult && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div className="w-8 h-8 border border-rs-border rounded-full animate-spin border-t-rs-red" />
            <span className="text-[9px] tracking-[0.3em] uppercase text-rs-muted">Awaiting Signal</span>
            <span className="text-[8px] font-mono text-rs-border">/ws/live</span>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="px-4 py-1.5 border-t border-rs-border bg-rs-panel flex justify-between items-center">
        {latestResult ? (
          <>
            <span className="text-[9px] font-mono text-rs-muted">
              FRAME <span className="text-rs-text">#{latestResult.frame_id || 0}</span>
            </span>
            <span className="text-[9px] font-mono text-rs-muted">
              LANES <span className={bothLanes ? 'text-rs-green' : 'text-rs-amber'}>
                {bothLanes ? 'BOTH DETECTED' : 'PARTIAL'}
              </span>
            </span>
            <span className="text-[9px] font-mono text-rs-muted">
              {new Date(latestResult.timestamp).toLocaleTimeString()}
            </span>
          </>
        ) : (
          <span className="text-[9px] font-mono text-rs-border tracking-widest">NO SIGNAL</span>
        )}
      </div>
    </div>
  )
}

export default VideoFeed
