/**
 * App.jsx
 * =======
 * Root layout for the Topo-Downscale WebGIS Dashboard.
 *
 * Layout (1440px+)
 * ─────────────────────────────────────────────────────────────────────
 * ┌──────────────────────── Header bar ───────────────────────────────┐
 * │  [Logo]  Topo-Downscale        [Region] [Date]  [API status]      │
 * ├────────────────────────────────────┬──────────────────────────────┤
 * │                                    │  MetricsPanel (fixed 320px)  │
 * │         MapViewer (Leaflet)        │──────────────────────────────│
 * │                                    │  AdvisoryCard (slides in)    │
 * └────────────────────────────────────┴──────────────────────────────┘
 *
 * On smaller screens the right panel stacks below the map.
 */

import { useState, useCallback } from 'react'
import MapViewer    from './components/MapViewer'
import AdvisoryCard from './components/AdvisoryCard'
import MetricsPanel from './components/MetricsPanel'

// ── Header ───────────────────────────────────────────────────────────────────

function Header({ apiOnline }) {
  return (
    <header className="h-14 shrink-0 glass border-b border-slate-700/60 flex items-center px-4 gap-4 z-50">
      {/* Logo + title */}
      <div className="flex items-center gap-2.5 mr-4">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-600 to-violet-600 flex items-center justify-center text-sm font-bold text-white shadow">
          TD
        </div>
        <div className="leading-tight">
          <h1 className="text-sm font-bold text-white tracking-tight">Topo-Downscale</h1>
          <p className="text-[9px] text-gray-500 uppercase tracking-widest">AI Microclimate Forecast · SIH 2024</p>
        </div>
      </div>

      {/* Divider */}
      <div className="h-6 w-px bg-slate-700 mx-1" />

      {/* Region badge */}
      <div className="flex items-center gap-1.5 bg-slate-800/60 rounded-lg px-2.5 py-1 border border-slate-700 text-[10px]">
        <span>📍</span>
        <span className="text-gray-400">North Rajasthan Pilot</span>
      </div>

      {/* Forecast date */}
      <div className="flex items-center gap-1.5 bg-slate-800/60 rounded-lg px-2.5 py-1 border border-slate-700 text-[10px]">
        <span>📅</span>
        <span className="text-gray-400">Forecast: 2024-06-15</span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Resolution badge */}
      <div className="flex items-center gap-2 text-[10px]">
        <span className="px-2 py-0.5 rounded-full bg-slate-700 text-gray-400 border border-slate-600">
          📡 25 km NWP
        </span>
        <span className="text-gray-600">→</span>
        <span className="px-2 py-0.5 rounded-full bg-violet-900/60 text-violet-300 border border-violet-700">
          🛰 1 km Topo-SRNet
        </span>
      </div>

      {/* API status indicator */}
      <div className={`flex items-center gap-1.5 text-[10px] px-2.5 py-1 rounded-lg border ${
        apiOnline
          ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800'
          : 'bg-red-950/60 text-red-400 border-red-800'
      }`}>
        <span className={`w-1.5 h-1.5 rounded-full ${apiOnline ? 'bg-emerald-400' : 'bg-red-400'}`} />
        {apiOnline ? 'API Online' : 'API Offline'}
      </div>
    </header>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [apiOnline,       setApiOnline]       = useState(true)

  const handlePanchayatSelect = useCallback((feature) => {
    setSelectedFeature(feature)
  }, [])

  const handleCloseAdvisory = useCallback(() => {
    setSelectedFeature(null)
  }, [])

  return (
    <div className="flex flex-col h-screen bg-gray-950 overflow-hidden">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <Header apiOnline={apiOnline} />

      {/* ── Body ────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Map (main, flexible) ────────────────────────────────── */}
        <div className="flex-1 relative overflow-hidden">
          <MapViewer onPanchayatSelect={handlePanchayatSelect} />
        </div>

        {/* ── Right panel (fixed 320px) ────────────────────────────── */}
        <div className="w-80 shrink-0 flex flex-col gap-3 p-3 overflow-hidden bg-gray-950/80 border-l border-slate-800">

          {/* Advisory card slides in when panchayat selected */}
          {selectedFeature ? (
            <div className="flex-1 overflow-hidden">
              <AdvisoryCard
                feature={selectedFeature}
                onClose={handleCloseAdvisory}
              />
            </div>
          ) : (
            /* Placeholder when no panchayat selected */
            <div className="glass rounded-2xl flex flex-col items-center justify-center p-6 text-center h-48 border border-dashed border-slate-700">
              <span className="text-3xl mb-2">🗺</span>
              <p className="text-xs text-gray-500 leading-relaxed">
                Click a <span className="text-brand-400">Block</span> on the map
                to load Panchayat forecasts,<br/>
                then click a <span className="text-violet-400">Panchayat polygon</span>
                to view its GKMS advisory.
              </p>
            </div>
          )}

          {/* Metrics panel (always visible below advisory) */}
          <div className={`overflow-hidden ${selectedFeature ? 'h-64' : 'flex-1'}`}>
            <MetricsPanel />
          </div>
        </div>
      </div>
    </div>
  )
}
