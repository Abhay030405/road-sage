export const WS_URL = 'ws://localhost:8000/ws/live'
export const API_URL = 'http://localhost:8000'
export const HEALTH_POLL_INTERVAL_MS = 3000
export const MAX_HISTORY_ITEMS = 50

export const COMMAND_COLORS = {
  FORWARD: { bg: 'bg-rs-green', text: 'text-white', hex: '#22c55e' },
  LEFT:    { bg: 'bg-rs-amber', text: 'text-white', hex: '#f59e0b' },
  RIGHT:   { bg: 'bg-rs-blue',  text: 'text-white', hex: '#3b82f6' },
  STOP:    { bg: 'bg-rs-red',   text: 'text-white', hex: '#ef4444' },
}

export const COMMAND_ICONS = {
  FORWARD: '↑',
  LEFT:    '←',
  RIGHT:   '→',
  STOP:    '■',
}

export const DECISION_PATH_LABELS = {
  geometric:       'Geometric',
  single_lane:     'Single Lane',
  ml_fallback:     'ML Fallback',
  safety_gate:     'Safety Gate',
  confidence_gate: 'Confidence Gate',
}
