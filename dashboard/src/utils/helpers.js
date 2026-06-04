export const formatConfidence = (conf) => {
  if (conf === null || conf === undefined) return '--'
  return `${(conf * 100).toFixed(1)}%`
}

export const formatOffset = (offset_m) => {
  if (offset_m === null || offset_m === undefined) return '--'
  const abs = Math.abs(offset_m).toFixed(2)
  if (Math.abs(offset_m) < 0.05) return `${abs}m (centered)`
  return offset_m > 0 ? `+${abs}m →` : `-${abs}m ←`
}

export const formatCurvature = (curv) => {
  if (curv === null || curv === undefined) return '--'
  const abs = Math.abs(curv)
  if (abs < 0.003) return 'Straight'
  if (abs < 0.008) return curv > 0 ? 'Mild right' : 'Mild left'
  return curv > 0 ? 'Sharp right' : 'Sharp left'
}

export const formatLatency = (ms) => {
  if (!ms && ms !== 0) return '--'
  return `${ms.toFixed(0)}ms`
}

export const getCommandColor = (command) => {
  const map = {
    FORWARD: '#22c55e',
    LEFT:    '#f59e0b',
    RIGHT:   '#3b82f6',
    STOP:    '#ef4444',
  }
  return map[command] || '#64748b'
}

export const timeAgo = (isoString) => {
  if (!isoString) return ''
  const seconds = Math.floor((Date.now() - new Date(isoString)) / 1000)
  if (seconds < 2) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.floor(seconds / 60)}m ago`
}
