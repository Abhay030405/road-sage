import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts'
import { useRoadSage } from '../context/RoadSageContext'

function getConfidenceMeta(pct) {
  if (pct >= 80) return { color: '#22c55e', statusColor: 'text-rs-green', label: 'HIGH CONFIDENCE' }
  if (pct >= 60) return { color: '#f59e0b', statusColor: 'text-rs-amber', label: 'MODERATE' }
  return { color: '#ef4444', statusColor: 'text-rs-red', label: 'LOW — STOPPING' }
}

function ConfidenceMeter() {
  const { latestResult } = useRoadSage()
  const confidence = latestResult?.confidence ?? 0
  const pct = Math.round(confidence * 100)
  const { color, statusColor, label } = getConfidenceMeta(pct)

  const data = [{ value: pct, fill: color }]

  return (
    <div className="bg-rs-panel rounded-lg border border-rs-border p-4">
      <div className="text-xs text-rs-muted mb-3 font-medium">Confidence</div>

      <div className="relative h-32">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="65%"
            outerRadius="100%"
            data={data}
            startAngle={200}
            endAngle={-20}
            barSize={12}
          >
            <RadialBar
              background={{ fill: '#1e1e2e' }}
              dataKey="value"
              cornerRadius={6}
              max={100}
            />
          </RadialBarChart>
        </ResponsiveContainer>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-black" style={{ color }}>
            {pct}%
          </span>
          <span className="text-xs text-rs-muted">confidence</span>
        </div>
      </div>

      <div className="mt-2 flex items-center justify-between text-xs text-rs-muted">
        <span>0%</span>
        <span className="text-rs-amber">⬤ 60% safety threshold</span>
        <span>100%</span>
      </div>

      <div className={`mt-2 text-center text-xs font-medium ${statusColor}`}>
        {label}
      </div>
    </div>
  )
}

export default ConfidenceMeter
