import { useRef, useEffect, useState } from 'react'
import { useRoadSage } from '../context/RoadSageContext'

function drawLaneLines(ctx, points, color, lineWidth = 3) {
  if (!points || points.length === 0) return
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1])
  }
  ctx.stroke()
}

function VideoFeed() {
  const { latestResult } = useRoadSage()
  const canvasRef = useRef(null)
  const [showGradCam, setShowGradCam] = useState(false)
  const [showOverlay, setShowOverlay] = useState(true)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    if (latestResult?.lane_viz_base64 && showOverlay) {
      const img = new Image()
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        if (latestResult.left_lane_points) {
          const color = latestResult.left_lane_detected ? '#22c55e' : '#ef4444'
          drawLaneLines(ctx, latestResult.left_lane_points, color)
        }
        if (latestResult.right_lane_points) {
          const color = latestResult.right_lane_detected ? '#22c55e' : '#ef4444'
          drawLaneLines(ctx, latestResult.right_lane_points, color)
        }
      }
      img.src = `data:image/jpeg;base64,${latestResult.lane_viz_base64}`
    } else if (latestResult?.gradcam_base64 && showGradCam) {
      const img = new Image()
      img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      img.src = `data:image/jpeg;base64,${latestResult.gradcam_base64}`
    } else {
      ctx.fillStyle = '#111118'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#64748b'
      ctx.font = '16px system-ui'
      ctx.textAlign = 'center'
      ctx.fillText('Waiting for frames...', canvas.width / 2, canvas.height / 2)
    }
  }, [latestResult, showOverlay, showGradCam])

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border overflow-hidden flex flex-col h-full">

      <div className="flex items-center justify-between px-4 py-2 border-b border-rs-border">
        <span className="text-sm font-medium text-rs-text">Live Camera Feed</span>
        <div className="flex gap-2">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`text-xs px-2 py-1 rounded border
              ${showOverlay ? 'border-rs-green text-rs-green' : 'border-rs-border text-rs-muted'}`}
          >
            Lane Overlay
          </button>
          <button
            onClick={() => setShowGradCam(!showGradCam)}
            className={`text-xs px-2 py-1 rounded border
              ${showGradCam ? 'border-rs-amber text-rs-amber' : 'border-rs-border text-rs-muted'}`}
          >
            GradCAM
          </button>
        </div>
      </div>

      <div className="flex-1 relative bg-black flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="w-full h-full object-contain"
        />
        {!latestResult && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-rs-muted gap-2">
            <div className="w-12 h-12 border-2 border-rs-border rounded-full animate-spin border-t-rs-green" />
            <span className="text-sm">Waiting for video stream...</span>
            <span className="text-xs">Connect via WebSocket /ws/live</span>
          </div>
        )}
      </div>

      {latestResult && (
        <div className="px-4 py-1.5 border-t border-rs-border flex justify-between text-xs text-rs-muted">
          <span>Frame #{latestResult.frame_id || 0}</span>
          <span>{latestResult.latency_ms?.total?.toFixed(0) || '--'}ms</span>
          <span>{new Date(latestResult.timestamp).toLocaleTimeString()}</span>
        </div>
      )}

    </div>
  )
}

export default VideoFeed
