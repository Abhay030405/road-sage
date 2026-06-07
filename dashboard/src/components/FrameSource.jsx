'use client'

import { useState, useRef, useCallback } from 'react'
import PropTypes from 'prop-types'
import { Video, Images, Camera, Square } from 'lucide-react'

const FRAME_W    = 640
const FRAME_H    = 480
const TARGET_FPS = 15

// Load a File into an HTMLImageElement
function loadImage(file) {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.src = URL.createObjectURL(file)
  })
}

export default function FrameSource({ sendFrame }) {
  const [streaming,  setStreaming]  = useState(false)
  const [mode,       setMode]       = useState(null)
  const [label,      setLabel]      = useState('')

  const canvasRef      = useRef(null)
  const videoRef       = useRef(null)
  const timerRef       = useRef(null)
  const mediaRef       = useRef(null)
  const imgIdxRef      = useRef(0)
  const imgsRef        = useRef([])
  const videoInputRef  = useRef(null)
  const imagesInputRef = useRef(null)

  // ─── Stop ──────────────────────────────────────────────────────────────────
  const stop = useCallback(() => {
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
    setMode(null)
    setLabel('')
  }, [])

  // ─── Blob sender (extracted to avoid deep nesting) ─────────────────────────
  const sendBlob = useCallback((blob) => {
    if (!blob) return
    sendFrame(blob)
  }, [sendFrame])

  // ─── Core pump ─────────────────────────────────────────────────────────────
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

  // ─── Video file ────────────────────────────────────────────────────────────
  const handleVideoFile = useCallback(async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    stop()
    setMode('video')
    setLabel(file.name)

    const v = videoRef.current
    v.src   = URL.createObjectURL(file)
    v.loop  = true
    v.muted = true
    await v.play()

    startPump(ctx => ctx.drawImage(v, 0, 0, FRAME_W, FRAME_H))
    e.target.value = ''
  }, [stop, startPump])

  // ─── Image sequence ────────────────────────────────────────────────────────
  const handleImages = useCallback(async (e) => {
    const files = Array.from(e.target.files ?? [])
    if (!files.length) return
    stop()
    setMode('images')
    setLabel(`${files.length} ${files.length === 1 ? 'image' : 'images'}`)

    const imgs = await Promise.all(files.map(loadImage))
    imgsRef.current   = imgs
    imgIdxRef.current = 0

    startPump(ctx => {
      ctx.drawImage(imgs[imgIdxRef.current % imgs.length], 0, 0, FRAME_W, FRAME_H)
      imgIdxRef.current++
    })
    e.target.value = ''
  }, [stop, startPump])

  // ─── Webcam ────────────────────────────────────────────────────────────────
  const handleWebcam = useCallback(async () => {
    stop()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_W, height: FRAME_H, frameRate: TARGET_FPS },
      })
      mediaRef.current    = stream
      const v             = videoRef.current
      v.srcObject         = stream
      v.muted             = true
      await v.play()

      setMode('webcam')
      setLabel('Webcam')
      startPump(ctx => ctx.drawImage(v, 0, 0, FRAME_W, FRAME_H))
    } catch (err) {
      alert(`Webcam error: ${err.message}`)
    }
  }, [stop, startPump])

  // ─── UI ────────────────────────────────────────────────────────────────────
  const btnBase   = 'flex items-center gap-1.5 px-3 py-1.5 rounded border text-[11px] font-semibold tracking-widest uppercase transition-colors'
  const btnActive = 'border-rs-red text-rs-text'
  const btnIdle   = 'border-rs-border text-rs-muted hover:border-rs-muted hover:text-rs-text'

  const activeStyle = { backgroundColor: 'rgba(192,23,43,0.1)' }

  return (
    <>
      {/* Hidden capture elements */}
      <canvas ref={canvasRef} width={FRAME_W} height={FRAME_H} className="hidden" />
      {/* track element satisfies accessibility requirement for media elements */}
      <video ref={videoRef} width={FRAME_W} height={FRAME_H} className="hidden" playsInline muted>
        <track kind="captions" />
      </video>
      <input ref={videoInputRef}  type="file" accept="video/*"          className="hidden" onChange={handleVideoFile} />
      <input ref={imagesInputRef} type="file" accept="image/*" multiple className="hidden" onChange={handleImages} />

      {/* Control bar */}
      <div className="mx-4 mt-3 mb-1 px-4 py-2 bg-rs-panel border border-rs-border rounded-lg flex items-center justify-center gap-3">

        <div className="flex items-center gap-8">
          <button
            className={`${btnBase} ${mode === 'video' && streaming ? btnActive : btnIdle}`}
            style={mode === 'video' && streaming ? activeStyle : {}}
            onClick={() => videoInputRef.current?.click()}
            title="Stream a video file"
          >
            <Video className="w-3.5 h-3.5" /> Video File
          </button>

          <button
            className={`${btnBase} ${mode === 'images' && streaming ? btnActive : btnIdle}`}
            style={mode === 'images' && streaming ? activeStyle : {}}
            onClick={() => imagesInputRef.current?.click()}
            title="Stream a folder of images"
          >
            <Images className="w-3.5 h-3.5" /> Images
          </button>

          <button
            className={`${btnBase} ${mode === 'webcam' && streaming ? btnActive : btnIdle}`}
            style={mode === 'webcam' && streaming ? activeStyle : {}}
            onClick={handleWebcam}
            title="Stream from webcam"
          >
            <Camera className="w-3.5 h-3.5" /> Webcam
          </button>
        </div>

        {streaming && (
          <>
            <div className="w-px h-5 bg-rs-border" />
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-rs-red animate-pulse" />
              <span className="text-[10px] font-mono text-rs-text truncate max-w-40">{label}</span>
              <span className="text-[10px] font-mono text-rs-muted tabular-nums">@ {TARGET_FPS} fps</span>
            </div>
            <button
              onClick={stop}
              className={`${btnBase} border-rs-red text-rs-red`}
              style={{ backgroundColor: 'rgba(192,23,43,0.08)' }}
            >
              <Square className="w-3 h-3 fill-rs-red" /> Stop
            </button>
          </>
        )}
      </div>
    </>
  )
}

FrameSource.propTypes = {
  sendFrame:   PropTypes.func.isRequired,
  isConnected: PropTypes.bool,
}

FrameSource.defaultProps = {
  isConnected: false,
}
