export const WS_URL = 'ws://localhost:8000/ws/live'
export const API_URL = 'http://localhost:8000'
export const HEALTH_POLL_INTERVAL_MS = 3000
export const MAX_HISTORY_ITEMS = 50

export const COMMAND_COLORS = {
  FORWARD: { bg: 'bg-rs-green', text: 'text-white', hex: '#3d8b5f' },
  LEFT:    { bg: 'bg-rs-amber', text: 'text-white', hex: '#c47d15' },
  RIGHT:   { bg: 'bg-rs-blue',  text: 'text-white', hex: '#2e6da4' },
  STOP:    { bg: 'bg-rs-red',   text: 'text-white', hex: '#c0172b' },
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
