import { useState, useEffect, useRef, useCallback } from 'react'
import { WS_URL } from '../constants'

const useWebSocket = (onMessage) => {
  const [connectionState, setConnectionState] = useState('connecting')
  const [lastError, setLastError] = useState(null)
  const [reconnectAttempt, setReconnectAttempt] = useState(0)

  const wsRef = useRef(null)
  const reconnectTimerRef = useRef()
  const shouldReconnectRef = useRef(true)
  const reconnectAttemptRef = useRef(0)

  const connect = useCallback(() => {
    clearTimeout(reconnectTimerRef.current)
    setConnectionState('connecting')

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnectionState('connected')
      setReconnectAttempt(0)
      reconnectAttemptRef.current = 0
      setLastError(null)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'prediction') {
          onMessage(data.result, data.frame_id, data.session_fps)
        }
      } catch {
        console.warn('Invalid WS message')
      }
    }

    ws.onerror = () => {
      setLastError('Connection error')
      setConnectionState('error')
    }

    ws.onclose = () => {
      setConnectionState('disconnected')
      if (shouldReconnectRef.current) {
        const attempt = reconnectAttemptRef.current
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000)
        reconnectTimerRef.current = setTimeout(() => {
          reconnectAttemptRef.current = attempt + 1
          setReconnectAttempt(attempt + 1)
          connect()
        }, delay)
      }
    }
  }, [onMessage])

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false
    clearTimeout(reconnectTimerRef.current)
    wsRef.current?.close()
  }, [])

  const sendFrame = useCallback((imageBlob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(imageBlob)
    } else {
      console.warn('WebSocket not open — frame dropped')
    }
  }, [])

  useEffect(() => {
    shouldReconnectRef.current = true
    connect()
    return () => disconnect()
  }, [])

  return { connectionState, lastError, reconnectAttempt, sendFrame, disconnect, connect }
}

export default useWebSocket
