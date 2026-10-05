/**
 * client.js
 * Axios API client for the Topo-Downscale FastAPI backend.
 * All requests go through Vite's proxy → http://localhost:8000
 */
import axios from 'axios'

const BASE = '/api/v1'

const api = axios.create({
  baseURL: BASE,
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
})

/** GET /api/v1/blocks — returns GeoJSON FeatureCollection */
export const fetchBlocks = (date) =>
  api.get('/blocks', { params: date ? { date } : {} }).then(r => r.data)

/** GET /api/v1/panchayats/{blockId} — returns GeoJSON FeatureCollection */
export const fetchPanchayats = (blockId, date) =>
  api.get(`/panchayats/${blockId}`, { params: date ? { date } : {} }).then(r => r.data)

/** GET /api/v1/advisory/{panchayatId} — returns AdvisoryResponse */
export const fetchAdvisory = (panchayatId, { maxAlerts = 5 } = {}) =>
  api.get(`/advisory/${panchayatId}`, { params: { max_alerts: maxAlerts } }).then(r => r.data)

export default api
