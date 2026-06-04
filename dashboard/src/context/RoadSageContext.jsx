import { createContext, useContext, useState, useCallback } from 'react'
import { MAX_HISTORY_ITEMS } from '../constants'

const RoadSageContext = createContext(null)

export function RoadSageProvider({ children }) {
  const [latestResult, setLatestResult] = useState(null)
  const [history, setHistory] = useState([])
  const [sessionFps, setSessionFps] = useState(0)
  const [frameId, setFrameId] = useState(0)
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null)

  const addResult = useCallback((result, frameId, fps) => {
    setLatestResult(result)
    setFrameId(frameId)
    setSessionFps(fps || 0)
    setHistory(prev => [
      { ...result, frameId, receivedAt: new Date().toISOString() },
      ...prev.slice(0, MAX_HISTORY_ITEMS - 1)
    ])
  }, [])

  return (
    <RoadSageContext.Provider value={{
      latestResult, history, sessionFps, frameId,
      selectedHistoryItem, setSelectedHistoryItem,
      addResult
    }}>
      {children}
    </RoadSageContext.Provider>
  )
}

export const useRoadSage = () => {
  const ctx = useContext(RoadSageContext)
  if (!ctx) throw new Error('useRoadSage must be inside RoadSageProvider')
  return ctx
}

export default RoadSageContext
