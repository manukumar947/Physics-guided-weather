/**
 * AdvisoryCard.jsx
 * ================
 * Bilingual agro-meteorological advisory panel for a selected Gram Panchayat.
 *
 * Features
 * --------
 * • Fetches advisories from GET /api/v1/advisory/{panchayatId}
 * • Language toggle: English ↔ हिन्दी
 * • Severity badge: CRITICAL (red) | HIGH (orange) | MODERATE (yellow) | LOW (green)
 * • Category icons (emoji) for quick visual scanning
 * • Weather summary strip showing downscaled values
 * • Dismissible via onClose prop
 *
 * Props
 * -----
 * feature    {object}    GeoJSON feature of the selected panchayat
 * onClose    {function}  Called when the close button is pressed
 */

import { useEffect, useState } from 'react'
import { fetchAdvisory } from '../api/client'

// ── Constants ─────────────────────────────────────────────────────────────────

const SEVERITY_CONFIG = {
  CRITICAL: { cls: 'badge-critical', label: 'Critical',  icon: '🚨' },
  HIGH:     { cls: 'badge-high',     label: 'High',      icon: '⚠️' },
  MODERATE: { cls: 'badge-moderate', label: 'Moderate',  icon: '🟡' },
  LOW:      { cls: 'badge-low',      label: 'Low',       icon: '✅' },
}

const CATEGORY_ICON = {
  Rainfall:      '🌧',
  Temperature:   '🌡',
  Humidity:      '💧',
  Wind:          '💨',
  'Pest-Disease':'🦠',
  Irrigation:    '🚿',
  Composite:     '⚡',
  General:       '🌤',
}

// ── Weather summary strip ──────────────────────────────────────────────────────

function WeatherSummary({ weather }) {
  if (!weather) return null
  const items = [
    { icon: '🌡', label: 'Tmax',   val: `${weather.tmax_c}°C` },
    { icon: '🌡', label: 'Tmin',   val: `${weather.tmin_c}°C` },
    { icon: '🌧', label: 'Rain',   val: `${weather.rainfall_mm} mm` },
    { icon: '💧', label: 'RH',     val: `${weather.rh_pct}%` },
    { icon: '💨', label: 'Wind',   val: `${weather.wind_ms} m/s` },
  ]
  return (
    <div className="grid grid-cols-5 gap-1 mb-3">
      {items.map(({ icon, label, val }) => (
        <div key={label}
          className="bg-slate-800/70 rounded-lg px-1.5 py-1.5 text-center border border-slate-700">
          <div className="text-base">{icon}</div>
          <div className="text-[9px] text-gray-500 uppercase tracking-wide">{label}</div>
          <div className="text-xs font-semibold text-gray-200">{val}</div>
        </div>
      ))}
    </div>
  )
}

// ── Single alert row ───────────────────────────────────────────────────────────

function AlertRow({ alert, lang }) {
  const [expanded, setExpanded] = useState(false)
  const sev    = SEVERITY_CONFIG[alert.severity] ?? SEVERITY_CONFIG.LOW
  const catIcon = CATEGORY_ICON[alert.category] ?? '📌'
  const message = lang === 'hi' ? alert.message_hi : alert.message_en
  const action  = lang === 'hi' ? alert.action_hi  : alert.action_en
  const isHindi = lang === 'hi'

  return (
    <div
      className="rounded-xl border border-slate-700 bg-slate-800/50 mb-2 overflow-hidden transition-all"
    >
      {/* Header row */}
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-700/40 transition-colors"
        onClick={() => setExpanded(e => !e)}
      >
        <span className="text-base">{catIcon}</span>
        <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-md ${sev.cls}`}>
          {sev.icon} {sev.label}
        </span>
        <span className="text-xs text-gray-400 flex-1">{alert.category}</span>
        <span className="text-gray-500 text-xs">{expanded ? '▲' : '▼'}</span>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className={`px-3 pb-3 border-t border-slate-700/50 pt-2 ${isHindi ? 'font-hindi' : ''}`}>
          <p className="text-xs text-gray-300 leading-relaxed mb-2">{message}</p>
          <div className="bg-slate-900/60 rounded-lg px-3 py-2 border-l-2 border-brand-500">
            <span className="text-[10px] uppercase tracking-widest text-brand-500 font-semibold">
              Recommended Action
            </span>
            <p className={`text-xs text-gray-200 leading-relaxed mt-1 ${isHindi ? 'font-hindi' : ''}`}>
              {action}
            </p>
          </div>
          {/* Triggered-by chips */}
          {alert.triggered_by && Object.keys(alert.triggered_by).length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {Object.entries(alert.triggered_by).map(([k, v]) => (
                <span key={k}
                  className="text-[9px] bg-slate-700 text-gray-400 px-1.5 py-0.5 rounded-full border border-slate-600">
                  {k}: {typeof v === 'number' ? v.toFixed(1) : v}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function AdvisoryCard({ feature, onClose }) {
  const [advisory, setAdvisory] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [lang,     setLang]     = useState('en')   // 'en' | 'hi'

  const panchayatId   = feature?.id
  const panchayatName = feature?.properties?.panchayat_name ?? panchayatId
  const blockName     = feature?.properties?.block_name ?? ''
  const elevation     = feature?.properties?.elevation_m
  const crop          = feature?.properties?.dominant_crop

  useEffect(() => {
    if (!panchayatId) return
    setLoading(true)
    setAdvisory(null)
    setError(null)
    fetchAdvisory(panchayatId, { maxAlerts: 5 })
      .then(data => setAdvisory(data))
      .catch(err  => setError('Advisory unavailable: ' + err.message))
      .finally(()  => setLoading(false))
  }, [panchayatId])

  return (
    <div className="glass rounded-2xl shadow-card flex flex-col h-full overflow-hidden">

      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between px-4 pt-4 pb-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg">📋</span>
            <h2 className="text-sm font-bold text-gray-100 leading-tight">
              {panchayatName}
            </h2>
          </div>
          <p className="text-[11px] text-gray-500 mt-0.5">
            {blockName} · {elevation != null ? `${elevation} m` : ''} · {crop ?? ''}
          </p>
          <p className="text-[10px] text-brand-500 mt-0.5 font-semibold tracking-wide uppercase">
            GKMS Agromet Advisory
          </p>
        </div>
        <button
          className="text-gray-500 hover:text-gray-200 transition-colors text-lg leading-none ml-2"
          onClick={onClose}
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* ── Language toggle ───────────────────────────────────────────── */}
      <div className="flex px-4 mb-2">
        <div className="flex rounded-lg overflow-hidden border border-slate-600 text-xs">
          <button
            className={`px-3 py-1 font-semibold transition-colors ${
              lang === 'en' ? 'bg-brand-600 text-white' : 'bg-slate-800 text-gray-400 hover:text-white'
            }`}
            onClick={() => setLang('en')}
          >
            English
          </button>
          <button
            className={`px-3 py-1 font-semibold font-hindi transition-colors ${
              lang === 'hi' ? 'bg-violet-600 text-white' : 'bg-slate-800 text-gray-400 hover:text-white'
            }`}
            onClick={() => setLang('hi')}
          >
            हिन्दी
          </button>
        </div>

        {/* Alert count badge */}
        {advisory && (
          <span className="ml-auto text-[10px] bg-slate-700 text-gray-400 rounded-full px-2 py-0.5 self-center border border-slate-600">
            {advisory.n_alerts} alert{advisory.n_alerts !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* ── Weather summary strip ─────────────────────────────────────── */}
      {advisory?.weather_summary && (
        <div className="px-4">
          <p className="text-[9px] uppercase tracking-widest text-gray-500 mb-1">
            Downscaled Forecast · {advisory.forecast_date}
          </p>
          <WeatherSummary weather={advisory.weather_summary} />
        </div>
      )}

      {/* ── Alert list ───────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {loading && (
          <div className="flex items-center justify-center h-24 gap-2 text-gray-500 text-sm">
            <div className="w-5 h-5 border-2 border-brand-500 border-t-transparent rounded-full spinner" />
            Fetching advisory…
          </div>
        )}

        {error && (
          <div className="bg-red-900/40 border border-red-700 text-red-300 text-xs rounded-xl p-3 mt-2">
            {error}
          </div>
        )}

        {advisory?.alerts?.map((alert, i) => (
          <AlertRow key={`${alert.alert_id}-${i}`} alert={alert} lang={lang} />
        ))}

        {advisory && advisory.alerts.length === 0 && (
          <div className="text-center text-gray-500 text-sm mt-8">
            ✅ No active advisories for this panchayat.
          </div>
        )}

        {/* Footer attribution */}
        {advisory && (
          <p className="text-[9px] text-gray-600 text-center mt-3">
            ICAR Agromet Advisory Service · {advisory.generated_at?.slice(0, 19)}Z
          </p>
        )}
      </div>
    </div>
  )
}
