'use client'

import { RoadSageProvider } from '../context/RoadSageContext'

export function Providers({ children }) {
  return <RoadSageProvider>{children}</RoadSageProvider>
}
