'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import PropTypes from 'prop-types'
import { Video, Images, Camera, Square } from 'lucide-react'
import { useRoadSage } from './context/RoadSageContext'

// ─── Demo data ────────────────────────────────────────────────────────────────

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
    command: 'STOP', confidence: 1, decision_path: 'safety_gate',
    lane_offset_m: 0, curvature_inv_m: 0,
    left_lane_detected: true, right_lane_detected: true,
    hazard_detected: true, hazard_reason: 'person detected in path',
    surface_class: 'clean',
    latency_ms: { lane: 30, scene: 26, decision: 2, total: 58 },
  },
]

// ─── Constants ────────────────────────────────────────────────────────────────

const FRAME_W    = 640
const FRAME_H    = 480
const TARGET_FPS = 15

function loadImage(file) {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.src = URL.createObjectURL(file)
  })
}

// ─── Component ────────────────────────────────────────────────────────────────

function TestDemo({ sendFrame }) {
  const { addResult } = useRoadSage()

  // Demo mode
  const [demoRunning, setDemoRunning] = useState(false)
  const [demoIdx, setDemoIdx]         = useState(0)

  // Real mode
  const [streaming, setStreaming] = useState(false)
  const [label,     setLabel]     = useState('')

  const canvasRef      = useRef(null)
  const videoRef       = useRef(null)
  const timerRef       = useRef(null)
  const mediaRef       = useRef(null)
  const imgIdxRef      = useRef(0)
  const imgsRef        = useRef([])
  const videoInputRef  = useRef(null)
  const imagesInputRef = useRef(null)

  // ── Demo ticker ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!demoRunning) return
    const id = setInterval(() => {
      const item = FAKE_SEQUENCE[demoIdx % FAKE_SEQUENCE.length]
      addResult({ ...item, frame_id: demoIdx, timestamp: new Date().toISOString() }, demoIdx, 14.5)
      setDemoIdx(i => i + 1)
    }, 800)
    return () => clearInterval(id)
  }, [demoRunning, demoIdx, addResult])

  // ── Real-mode helpers ────────────────────────────────────────────────────
  const sendBlob = useCallback((blob) => {
    if (blob) sendFrame(blob)
  }, [sendFrame])

  const stopReal = useCallback(() => {
    clearInterval(timerRef.current)
    timerRef.current = null
    if (mediaRef.current) {
      mediaRef.current.getTracks().forEach(t => t.stop())
      mediaRef.current = null
    }
    const v = videoRef.current
    if (v) { v.pause(); v.src = ''; v.srcObject = null }
    imgsRef.current   = []
    imgIdxRef.current = 0
    setStreaming(false)
    setLabel('')
  }, [])

  const startPump = useCallback((drawFn) => {
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')
    setStreaming(true)
    timerRef.current = setInterval(() => {
      try {
        drawFn(ctx)
        canvas.toBlob(sendBlob, 'image/jpeg', 0.85)
      } catch { /* frame dropped */ }
    }, 1000 / TARGET_FPS)
  }, [sendBlob])

  const handleVideoFile = useCallback(async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    stopReal()
    setLabel(file.name)
    const v = videoRef.current
    v.src = URL.createObjectURL(file)
    v.loop = true
    v.muted = true
    await v.play()
    startPump(ctx => ctx.drawImage(v, 0, 0, FRAME_W, FRAME_H))
    e.target.value = ''
  }, [stopReal, startPump])

  const handleImages = useCallback(async (e) => {
    const files = Array.from(e.target.files ?? [])
    if (!files.length) return
    stopReal()
    setLabel(`${files.length} ${files.length === 1 ? 'image' : 'images'}`)
    const imgs = await Promise.all(files.map(loadImage))
    imgsRef.current   = imgs
    imgIdxRef.current = 0
    startPump(ctx => {
      ctx.drawImage(imgs[imgIdxRef.current % imgs.length], 0, 0, FRAME_W, FRAME_H)
      imgIdxRef.current++
    })
    e.target.value = ''
  }, [stopReal, startPump])

  const handleWebcam = useCallback(async () => {
    stopReal()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_W, height: FRAME_H, frameRate: TARGET_FPS },
      })
      mediaRef.current = stream
      const v = videoRef.current
      v.srcObject = stream
      v.muted = true
      await v.play()
      setLabel('Webcam')
      startPump(ctx => ctx.drawImage(v, 0, 0, FRAME_W, FRAME_H))
    } catch (err) {
      alert(`Webcam error: ${err.message}`)
    }
  }, [stopReal, startPump])

  // ── UI helpers ───────────────────────────────────────────────────────────
  const btn = 'flex items-center gap-1.5 text-[10px] px-2.5 py-1 rounded border tracking-widest uppercase font-semibold transition-colors'

  return (
    <>
      {/* Hidden capture elements */}
      <canvas ref={canvasRef} width={FRAME_W} height={FRAME_H} className="hidden" />
      <video  ref={videoRef}  width={FRAME_W} height={FRAME_H} className="hidden" playsInline muted>
        <track kind="captions" />
      </video>
      <input ref={videoInputRef}  type="file" accept="video/*"          className="hidden" onChange={handleVideoFile} />
      <input ref={imagesInputRef} type="file" accept="image/*" multiple className="hidden" onChange={handleImages} />

      {/* Floating panel */}
      <div className="fixed bottom-4 right-4 bg-rs-panel border border-rs-border rounded-lg p-3 z-50 flex flex-col gap-2.5 min-w-56">

        {/* ── Demo row ── */}
        <div className="flex items-center justify-between gap-3">
          <span className="text-[10px] tracking-widest uppercase text-rs-muted">Demo Mode</span>
          <button
            type="button"
            onClick={() => setDemoRunning(r => !r)}
            className={`text-[10px] px-3 py-1 rounded font-semibold tracking-wide
              ${demoRunning ? 'bg-rs-red text-white' : 'bg-rs-green text-white'}`}
          >
            {demoRunning ? 'Stop' : 'Start'} Simulation
          </button>
        </div>

        <div className="h-px bg-rs-border" />

        {/* ── Live / real mode row ── */}
        {streaming ? (
          <div className="flex items-center gap-2 justify-between">
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="w-1.5 h-1.5 rounded-full bg-rs-red animate-pulse flex-shrink-0" />
              <span className="text-[10px] font-mono text-rs-text truncate">{label}</span>
              <span className="text-[10px] text-rs-muted flex-shrink-0">@ {TARGET_FPS} fps</span>
            </div>
            <button onClick={stopReal}
              className={`${btn} border-rs-red text-rs-red flex-shrink-0`}
              style={{ backgroundColor: 'rgba(192,23,43,0.08)' }}>
              <Square className="w-2.5 h-2.5" /> Stop
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] tracking-widest uppercase text-rs-muted mr-1">Live</span>
            <button className={`${btn} border-rs-border text-rs-muted hover:text-rs-text hover:border-rs-muted`}
              onClick={() => videoInputRef.current?.click()}>
              <Video className="w-3 h-3" /> Video
            </button>
            <button className={`${btn} border-rs-border text-rs-muted hover:text-rs-text hover:border-rs-muted`}
              onClick={() => imagesInputRef.current?.click()}>
              <Images className="w-3 h-3" /> Images
            </button>
            <button className={`${btn} border-rs-border text-rs-muted hover:text-rs-text hover:border-rs-muted`}
              onClick={handleWebcam}>
              <Camera className="w-3 h-3" /> Webcam
            </button>
          </div>
        )}

      </div>
    </>
  )
}

TestDemo.propTypes = {
  sendFrame: PropTypes.func.isRequired,
}

export default TestDemo
