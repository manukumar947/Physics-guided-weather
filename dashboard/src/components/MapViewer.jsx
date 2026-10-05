/**
 * MapViewer.jsx
 * =============
 * Primary WebGIS map component for Topo-Downscale.
 *
 * Features
 * --------
 * • "Block Forecast View"     — choropleth of 25 km coarse NWP weather
 * • "Panchayat Downscaled View" — choropleth of 1 km Topo-SRNet output
 * • Variable selector: Tmax | Rainfall | RH | Wind
 * • Click a block → zooms in and loads its panchayats
 * • Click a panchayat → fires onPanchayatSelect (parent shows AdvisoryCard)
 * • Choropleth legend rendered inline below the map
 *
 * Props
 * -----
 * onPanchayatSelect(feature) — called when a GP polygon is clicked
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
  Popup,
} from 'react-leaflet'
import L from 'leaflet'
import { fetchBlocks, fetchPanchayats } from '../api/client'

// ── Constants ────────────────────────────────────────────────────────────────

const INITIAL_CENTER = [28.5, 74.5]   // North Rajasthan centroid
const INITIAL_ZOOM   = 8

/** Weather variables available for choropleth coloring */
const VARIABLES = [
  { key: 'tmax_c',      label: 'Tmax (°C)',     coarseKey: 'tmax_c',      unit: '°C',  min: 28, max: 46 },
  { key: 'rainfall_mm', label: 'Rainfall (mm)', coarseKey: 'rainfall_mm', unit: 'mm',  min: 0,  max: 60 },
  { key: 'rh_pct',      label: 'Humidity (%)',  coarseKey: 'rh_pct',      unit: '%',   min: 20, max: 100 },
  { key: 'wind_ms',     label: 'Wind (m/s)',    coarseKey: 'wind_ms',     unit: 'm/s', min: 0,  max: 15 },
]

// Colour ramp: cool (blue) → warm (red) via green-yellow
const COLOR_STOPS = [
  [0.00, '#0ea5e9'],  // sky-500
  [0.25, '#22d3ee'],  // cyan-400
  [0.50, '#84cc16'],  // lime-400
  [0.70, '#facc15'],  // yellow-400
  [0.85, '#f97316'],  // orange-400
  [1.00, '#dc2626'],  // red-600
]

/** Interpolate the color ramp for a normalised value [0,1]. */
function lerpColor(t) {
  t = Math.max(0, Math.min(1, t))
  for (let i = 1; i < COLOR_STOPS.length; i++) {
    const [t0, c0] = COLOR_STOPS[i - 1]
    const [t1, c1] = COLOR_STOPS[i]
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0)
      const hex = n => n.toString(16).padStart(2, '0')
      const parse = c => [
        parseInt(c.slice(1, 3), 16),
        parseInt(c.slice(3, 5), 16),
        parseInt(c.slice(5, 7), 16),
      ]
      const [r0, g0, b0] = parse(c0)
      const [r1, g1, b1] = parse(c1)
      return `#${hex(Math.round(r0 + f * (r1 - r0)))}${hex(Math.round(g0 + f * (g1 - g0)))}${hex(Math.round(b0 + f * (b1 - b0)))}`
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1]
}

function choroplethColor(value, varMeta) {
  const t = (value - varMeta.min) / (varMeta.max - varMeta.min)
  return lerpColor(t)
}

// ── Sub-component: zoom to bounds helper ─────────────────────────────────────
function FlyToBounds({ bounds }) {
  const map = useMap()
  useEffect(() => {
    if (bounds) map.flyToBounds(bounds, { padding: [40, 40], duration: 0.8 })
  }, [bounds, map])
  return null
}

// ── Choropleth Legend ─────────────────────────────────────────────────────────
function ChoroplethLegend({ varMeta }) {
  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-[1000] glass rounded-xl px-4 py-2 flex items-center gap-3 text-xs text-gray-300 pointer-events-none">
      <span className="font-medium text-gray-400">{varMeta.min}{varMeta.unit}</span>
      <div className="legend-gradient w-40 h-3 rounded-full" />
      <span className="font-medium text-gray-400">{varMeta.max}{varMeta.unit}</span>
      <span className="ml-2 text-brand-500 font-semibold">{varMeta.label}</span>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function MapViewer({ onPanchayatSelect }) {
  const [viewMode,   setViewMode]   = useState('block')      // 'block' | 'panchayat'
  const [activeVar,  setActiveVar]  = useState(VARIABLES[0]) // selected weather var
  const [blocks,     setBlocks]     = useState(null)
  const [panchayats, setPanchayats] = useState(null)
  const [flyBounds,  setFlyBounds]  = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [hoveredId,  setHoveredId]  = useState(null)

  // Load blocks on mount
  useEffect(() => {
    setLoading(true)
    fetchBlocks()
      .then(data => { setBlocks(data); setError(null) })
      .catch(err  => setError('Failed to load blocks: ' + err.message))
      .finally(()  => setLoading(false))
  }, [])

  // ── Block click handler ────────────────────────────────────────────────
  const handleBlockClick = useCallback((feature, layer) => {
    const blockId = feature.properties.block_id
    const bounds  = layer.getBounds()
    setFlyBounds(bounds)
    setLoading(true)
    fetchPanchayats(blockId)
      .then(data => {
        setPanchayats(data)
        setViewMode('panchayat')
        setError(null)
      })
      .catch(err => setError('Failed to load panchayats: ' + err.message))
      .finally(() => setLoading(false))
  }, [])

  // ── Style functions ────────────────────────────────────────────────────
  const blockStyle = useCallback((feature) => {
    const w   = feature.properties.weather
    const val = w[activeVar.coarseKey] ?? 0
    const fill = choroplethColor(val, activeVar)
    const isHovered = hoveredId === feature.properties.block_id
    return {
      fillColor:   fill,
      fillOpacity: isHovered ? 0.80 : 0.55,
      color:       isHovered ? '#6366f1' : '#334155',
      weight:      isHovered ? 2.5 : 1.5,
    }
  }, [activeVar, hoveredId])

  const panchayatStyle = useCallback((feature) => {
    const w    = feature.properties.weather
    const ds   = viewMode === 'panchayat' ? w.downscaled : w.coarse
    const val  = ds[activeVar.key] ?? 0
    const fill = choroplethColor(val, activeVar)
    const isHovered = hoveredId === feature.properties.panchayat_id
    return {
      fillColor:   fill,
      fillOpacity: isHovered ? 0.85 : 0.65,
      color:       isHovered ? '#a78bfa' : '#475569',
      weight:      isHovered ? 2.5 : 1,
    }
  }, [activeVar, viewMode, hoveredId])

  // ── GeoJSON event binders ──────────────────────────────────────────────
  const onEachBlock = useCallback((feature, layer) => {
    const p = feature.properties
    const w = p.weather
    layer.bindTooltip(
      `<div class="font-semibold text-brand-500">${p.block_name}</div>
       <div class="text-xs text-gray-400">${p.district} District</div>
       <div class="mt-1 text-sm">Tmax <b>${w.tmax_c}°C</b> · Rain <b>${w.rainfall_mm} mm</b></div>
       <div class="text-xs text-gray-400 mt-0.5">${p.n_panchayats} Panchayats — click to explore</div>`,
      { className: 'leaflet-tooltip-dark', sticky: true }
    )
    layer.on({
      click:     () => handleBlockClick(feature, layer),
      mouseover: (e) => { setHoveredId(p.block_id); e.target.bringToFront() },
      mouseout:  () => setHoveredId(null),
    })
  }, [handleBlockClick])

  const onEachPanchayat = useCallback((feature, layer) => {
    const p  = feature.properties
    const ds = p.weather.downscaled
    const co = p.weather.coarse
    layer.bindTooltip(
      `<div class="font-semibold text-violet-400">${p.panchayat_name}</div>
       <div class="text-xs text-gray-400">${p.block_name} Block · ${p.elevation_m} m</div>
       <div class="mt-1 grid grid-cols-2 gap-x-3 text-xs">
         <span class="text-gray-400">Coarse Tmax</span><b>${co.tmax_c}°C</b>
         <span class="text-gray-400">↓ Downscaled</span><b class="text-emerald-400">${ds.tmax_c}°C</b>
         <span class="text-gray-400">Rain (DS)</span><b>${ds.rainfall_mm} mm</b>
         <span class="text-gray-400">RH (DS)</span><b>${ds.rh_pct}%</b>
       </div>
       <div class="text-xs text-violet-400 mt-1">Click for GKMS Advisory</div>`,
      { className: 'leaflet-tooltip-dark', sticky: true }
    )
    layer.on({
      click:     () => onPanchayatSelect && onPanchayatSelect(feature),
      mouseover: (e) => { setHoveredId(p.panchayat_id); e.target.bringToFront() },
      mouseout:  () => setHoveredId(null),
    })
  }, [onPanchayatSelect])

  const activeVarMeta = activeVar

  return (
    <div className="relative w-full h-full flex flex-col">

      {/* ── Toolbar ────────────────────────────────────────────────────── */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1000] flex items-center gap-2 glass rounded-2xl px-3 py-2 shadow-card">

        {/* View mode toggle */}
        <div className="flex rounded-lg overflow-hidden border border-slate-600 mr-2">
          <button
            className={`px-3 py-1.5 text-xs font-semibold transition-colors ${
              viewMode === 'block'
                ? 'bg-brand-600 text-white'
                : 'bg-slate-800 text-gray-400 hover:text-white'
            }`}
            onClick={() => setViewMode('block')}
            title="Show 25km coarse NWP block forecasts"
          >
            📡 Block Forecast
          </button>
          <button
            className={`px-3 py-1.5 text-xs font-semibold transition-colors ${
              viewMode === 'panchayat'
                ? 'bg-violet-600 text-white'
                : 'bg-slate-800 text-gray-400 hover:text-white'
            }`}
            onClick={() => panchayats && setViewMode('panchayat')}
            title="Show 1km Topo-SRNet downscaled panchayat forecasts"
            disabled={!panchayats}
          >
            🛰 Downscaled GP
          </button>
        </div>

        {/* Variable selector */}
        <select
          className="bg-slate-800 border border-slate-600 text-gray-200 text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-brand-500"
          value={activeVar.key}
          onChange={e => setActiveVar(VARIABLES.find(v => v.key === e.target.value))}
        >
          {VARIABLES.map(v => (
            <option key={v.key} value={v.key}>{v.label}</option>
          ))}
        </select>

        {/* Back button */}
        {panchayats && (
          <button
            className="ml-2 px-3 py-1.5 text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-gray-300 rounded-lg border border-slate-600 transition-colors"
            onClick={() => { setPanchayats(null); setViewMode('block'); setFlyBounds(null) }}
          >
            ← All Blocks
          </button>
        )}
      </div>

      {/* ── Loading spinner ──────────────────────────────────────────── */}
      {loading && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1000] glass rounded-full px-4 py-2 flex items-center gap-2 text-xs text-gray-300">
          <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full spinner" />
          Loading…
        </div>
      )}

      {/* ── Error banner ─────────────────────────────────────────────── */}
      {error && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-[1000] bg-red-900/80 border border-red-500 text-red-200 text-xs rounded-xl px-4 py-2">
          ⚠ {error}
        </div>
      )}

      {/* ── Map ──────────────────────────────────────────────────────── */}
      <div className="flex-1 relative">
        <MapContainer
          center={INITIAL_CENTER}
          zoom={INITIAL_ZOOM}
          className="h-full w-full"
          zoomControl={false}
        >
          {/* Dark basemap tile layer */}
          <TileLayer
            attribution='&copy; OpenStreetMap'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            className="map-tiles-dark"
            maxZoom={19}
          />

          {/* Fly-to on block click */}
          {flyBounds && <FlyToBounds bounds={flyBounds} />}

          {/* Block layer */}
          {blocks && viewMode === 'block' && (
            <GeoJSON
              key={`blocks-${activeVar.key}`}
              data={blocks}
              style={blockStyle}
              onEachFeature={onEachBlock}
            />
          )}

          {/* Panchayat layer */}
          {panchayats && viewMode === 'panchayat' && (
            <GeoJSON
              key={`panchayats-${activeVar.key}-${viewMode}`}
              data={panchayats}
              style={panchayatStyle}
              onEachFeature={onEachPanchayat}
            />
          )}
        </MapContainer>

        {/* ── Choropleth legend ────────────────────────────────────────── */}
        <ChoroplethLegend varMeta={activeVarMeta} />

        {/* ── View mode badge ──────────────────────────────────────────── */}
        <div className="absolute top-3 right-3 z-[1000] glass rounded-xl px-3 py-1.5 text-xs text-gray-400">
          {viewMode === 'block'
            ? '📡 25 km · Coarse NWP'
            : '🛰 1 km · Topo-SRNet'}
        </div>
      </div>
    </div>
  )
}
