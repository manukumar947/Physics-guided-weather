"""
main.py
=======
Topo-Downscale FastAPI Application — Phase 3 MVP

Endpoints
---------
GET  /health
    Service health check.

GET  /api/v1/blocks
    Returns a GeoJSON FeatureCollection of all blocks in the pilot region,
    each carrying representative coarse NWP weather.

GET  /api/v1/panchayats/{block_id}
    Returns a GeoJSON FeatureCollection of Gram Panchayats within the block,
    with both coarse (bilinear upsampled) and downscaled (Topo-SRNet)
    weather payloads for the latest forecast date.

GET  /api/v1/advisory/{panchayat_id}
    Returns the bilingual GKMS advisory for the given panchayat, derived
    from its downscaled weather prediction.

Architecture Notes
------------------
* The MVP uses a **mock data layer** (`_mock_store`) that simulates the
  Topo-SRNet inference output. In production, replace ``_infer_downscaled()``
  with actual model forward-pass logic loading ``checkpoints/best_model.pth``.
* CORS is enabled for all origins to allow the React dashboard to call the
  API during local development.
* All responses are validated against Pydantic v2 schemas before serialisation,
  ensuring the frontend always receives well-typed GeoJSON.

Run locally
-----------
    uvicorn src.api.main:app --reload --port 8000

    or from repo root:
    python -m uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import datetime
import math
import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Use sys.path insertion for flat-module imports when run directly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import (
    AdvisoryAlertSchema,
    AdvisoryResponse,
    BlockCollection,
    BlockFeature,
    BlockProperties,
    CoarseWeather,
    DownscaledWeather,
    HealthResponse,
    PanchayatCollection,
    PanchayatFeature,
    PanchayatProperties,
    PanchayatWeather,
    Polygon,
)
from advisory.advisory_rules import evaluate_advisory


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Topo-Downscale API",
    description=(
        "AI-driven geospatial downscaling pipeline — "
        "translates 25 km Block NWP forecasts to 1 km Gram Panchayat "
        "microclimate predictions using Topo-SRNet."
    ),
    version="1.0.0",
    contact={
        "name": "SIH 2024 Team — Topo-Downscale",
        "url": "https://github.com/sih2024/topo-downscale",
    },
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow React dashboard on localhost:5173 (Vite default) + all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mock data store
# Simulates the pilot region: 3 blocks in North Rajasthan, each with 5 GPs.
# Replace with database / model inference calls in production.
# ---------------------------------------------------------------------------

_FORECAST_DATE = "2024-06-15"
_MODEL_VERSION  = "topo-srnet-v1"

# Pilot blocks: [block_id, name, district, centroid_lat, centroid_lon]
_BLOCKS_META = [
    ("RJ_BL_001", "Churu Sadar",   "Churu",   28.30, 74.95),
    ("RJ_BL_002", "Sujangarh",     "Churu",   27.70, 74.46),
    ("RJ_BL_003", "Taranagar",     "Churu",   28.68, 75.03),
]

# Panchayat seeds per block: [gp_id, name, pop, area_km2, crop, elevation_m]
_GP_META: Dict[str, List[tuple]] = {
    "RJ_BL_001": [
        ("RJ_CH_001", "Bhagwanpura GP",  4320, 42.5, "Pearl Millet",  285.0),
        ("RJ_CH_002", "Salasar GP",      6100, 38.2, "Clusterbean",   270.0),
        ("RJ_CH_003", "Ratangarh GP",    8750, 55.0, "Mustard",       295.0),
        ("RJ_CH_004", "Bidasar GP",      3900, 40.8, "Pearl Millet",  278.0),
        ("RJ_CH_005", "Malsisar GP",     5600, 47.3, "Moth Bean",     268.0),
    ],
    "RJ_BL_002": [
        ("RJ_CH_006", "Dholiya GP",      3100, 31.0, "Pearl Millet",  310.0),
        ("RJ_CH_007", "Khajuwala GP",    2800, 28.5, "Cotton",        305.0),
        ("RJ_CH_008", "Nokha GP",        4500, 44.1, "Clusterbean",   290.0),
        ("RJ_CH_009", "Bajju GP",        3700, 36.0, "Pearl Millet",  315.0),
        ("RJ_CH_010", "Kolayat GP",      6200, 58.0, "Wheat",         280.0),
    ],
    "RJ_BL_003": [
        ("RJ_CH_011", "Sardarshahar GP", 9100, 62.0, "Pearl Millet",  330.0),
        ("RJ_CH_012", "Dulchasar GP",    3400, 33.0, "Moth Bean",     345.0),
        ("RJ_CH_013", "Rajgarh GP",      5200, 48.5, "Mustard",       322.0),
        ("RJ_CH_014", "Sujangarh GP",    7300, 52.0, "Clusterbean",   318.0),
        ("RJ_CH_015", "Fatehpur GP",     4800, 41.5, "Pearl Millet",  340.0),
    ],
}


def _make_block_polygon(lat: float, lon: float, half_deg: float = 0.12) -> Polygon:
    """Generate an approximate square block boundary polygon."""
    coords = [
        [lon - half_deg, lat - half_deg],
        [lon + half_deg, lat - half_deg],
        [lon + half_deg, lat + half_deg],
        [lon - half_deg, lat + half_deg],
        [lon - half_deg, lat - half_deg],  # close ring
    ]
    return Polygon(type="Polygon", coordinates=[coords])


def _make_gp_polygon(lat: float, lon: float, half_deg: float = 0.03) -> Polygon:
    """Generate an approximate square GP boundary polygon (smaller than block)."""
    coords = [
        [lon - half_deg, lat - half_deg],
        [lon + half_deg, lat - half_deg],
        [lon + half_deg, lat + half_deg],
        [lon - half_deg, lat + half_deg],
        [lon - half_deg, lat - half_deg],
    ]
    return Polygon(type="Polygon", coordinates=[coords])


def _seed_for(pid: str) -> int:
    """Deterministic seed from string ID for reproducible mock weather."""
    return sum(ord(c) for c in pid)


def _mock_coarse_weather(block_id: str) -> CoarseWeather:
    """Simulate a block-level NWP coarse weather prediction."""
    rng = random.Random(_seed_for(block_id))
    return CoarseWeather(
        tmax_c=round(rng.uniform(33.0, 40.0), 1),
        tmin_c=round(rng.uniform(22.0, 29.0), 1),
        rainfall_mm=round(max(0.0, rng.normalvariate(18.0, 12.0)), 1),
        rh_pct=round(rng.uniform(55.0, 88.0), 1),
        wind_ms=round(rng.uniform(2.0, 9.0), 1),
    )


def _infer_downscaled(
    coarse: CoarseWeather,
    elevation_m: float,
    gp_id: str,
) -> DownscaledWeather:
    """
    Simulate Topo-SRNet inference: apply topographic corrections to coarse weather.

    In production, replace this function with:
        model = TopoSRNet(...)
        model.load_state_dict(torch.load('checkpoints/best_model.pth')['model'])
        out = model(weather_tensor, static_tensor)

    MVP mock logic:
    * Temperature: apply environmental lapse rate (−6.5°C per 1000 m above base 270 m)
    * Rainfall: orographic enhancement on windward slopes (elevation proxy)
    * RH: adjust inversely with temperature correction
    """
    rng = random.Random(_seed_for(gp_id) + 7)
    elev_base = 270.0  # approx. minimum elevation in region (m)
    elev_delta = max(0.0, elevation_m - elev_base) / 1000.0  # km above base

    # Lapse rate correction: −6.5°C per 1 km
    lapse_correction = -6.5 * elev_delta
    tmax_ds = round(coarse.tmax_c + lapse_correction + rng.uniform(-0.5, 0.5), 1)
    tmin_ds = round(coarse.tmin_c + lapse_correction * 0.8 + rng.uniform(-0.4, 0.4), 1)

    # Orographic rainfall enhancement (higher elevations → slightly more rain)
    orographic_factor = 1.0 + 0.15 * elev_delta
    rain_ds = round(max(0.0, coarse.rainfall_mm * orographic_factor + rng.uniform(-1.5, 1.5)), 1)

    # RH adjustment (higher = cooler = relatively more humid)
    rh_ds = round(min(99.0, coarse.rh_pct + 3.0 * elev_delta + rng.uniform(-2, 2)), 1)

    # Wind: terrain channeling effect (simplified)
    wind_ds = round(max(0.2, coarse.wind_ms + rng.uniform(-1.0, 1.5)), 1)

    return DownscaledWeather(
        tmax_c=tmax_ds,
        tmin_c=min(tmin_ds, tmax_ds - 1.0),
        rainfall_mm=rain_ds,
        rh_pct=rh_ds,
        wind_ms=wind_ds,
        tmax_delta_c=round(tmax_ds - coarse.tmax_c, 2),
        rainfall_delta_mm=round(rain_ds - coarse.rainfall_mm, 2),
    )


def _get_gp_latlon(gp_id: str, block_lat: float, block_lon: float, idx: int) -> tuple:
    """Scatter GP centroids around the block centroid in a grid pattern."""
    offsets = [
        (-0.06, -0.07), (+0.05, -0.06), (+0.00, +0.08),
        (-0.07, +0.06), (+0.08, +0.04),
    ]
    dy, dx = offsets[idx % len(offsets)]
    return block_lat + dy, block_lon + dx


# ---------------------------------------------------------------------------
# Lookup helpers (O(1) from in-memory store)
# ---------------------------------------------------------------------------

def _all_blocks() -> Dict[str, dict]:
    """Build full block records (lazy, cached in module scope)."""
    result = {}
    for bid, bname, district, blat, blon in _BLOCKS_META:
        coarse = _mock_coarse_weather(bid)
        result[bid] = {
            "bid": bid, "bname": bname, "district": district,
            "blat": blat, "blon": blon, "coarse": coarse,
        }
    return result


_BLOCK_STORE: Dict[str, dict] = _all_blocks()


def _panchayats_for_block(block_id: str) -> List[dict]:
    """Return GP records for a given block."""
    if block_id not in _GP_META:
        return []
    blk = _BLOCK_STORE[block_id]
    records = []
    for idx, (gid, gname, pop, area, crop, elev) in enumerate(_GP_META[block_id]):
        glat, glon = _get_gp_latlon(gid, blk["blat"], blk["blon"], idx)
        coarse = blk["coarse"]
        downscaled = _infer_downscaled(coarse, elev, gid)
        records.append({
            "gid": gid, "gname": gname, "glat": glat, "glon": glon,
            "pop": pop, "area": area, "crop": crop, "elev": elev,
            "coarse": coarse, "downscaled": downscaled,
            "block_id": block_id, "bname": blk["bname"], "district": blk["district"],
        })
    return records


def _find_panchayat(panchayat_id: str) -> Optional[dict]:
    """Search all blocks for a given GP id."""
    for block_id in _GP_META:
        for gp in _panchayats_for_block(block_id):
            if gp["gid"] == panchayat_id:
                return gp
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Service liveness probe."""
    return HealthResponse(status="ok")


@app.get(
    "/api/v1/blocks",
    response_model=BlockCollection,
    tags=["Forecast"],
    summary="List all blocks with coarse NWP weather",
    response_description="GeoJSON FeatureCollection of administrative blocks.",
)
def get_blocks(
    date: str = Query(
        default=_FORECAST_DATE,
        description="Forecast date (YYYY-MM-DD)",
        examples=["2024-06-15"],
    )
) -> BlockCollection:
    """
    Returns a GeoJSON FeatureCollection of all blocks in the pilot region.
    Each feature carries the coarse NWP weather for the specified forecast date.
    """
    features: List[BlockFeature] = []

    for bid, blk in _BLOCK_STORE.items():
        props = BlockProperties(
            block_id=bid,
            block_name=blk["bname"],
            district=blk["district"],
            state="Rajasthan",
            n_panchayats=len(_GP_META.get(bid, [])),
            centroid_lat=blk["blat"],
            centroid_lon=blk["blon"],
            weather=blk["coarse"],
        )
        feature = BlockFeature(
            type="Feature",
            id=bid,
            geometry=_make_block_polygon(blk["blat"], blk["blon"]),
            properties=props,
        )
        features.append(feature)

    return BlockCollection(
        type="FeatureCollection",
        features=features,
        meta={
            "region": "north_rajasthan_pilot",
            "forecast_date": date,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_blocks": len(features),
            "model_version": _MODEL_VERSION,
        },
    )


@app.get(
    "/api/v1/panchayats/{block_id}",
    response_model=PanchayatCollection,
    tags=["Forecast"],
    summary="Get coarse vs. downscaled weather for all panchayats in a block",
    response_description="GeoJSON FeatureCollection of Gram Panchayats.",
)
def get_panchayats(
    block_id: str,
    date: str = Query(
        default=_FORECAST_DATE,
        description="Forecast date (YYYY-MM-DD)",
    ),
) -> PanchayatCollection:
    """
    Returns a GeoJSON FeatureCollection of all Gram Panchayats within the
    given block. Each feature carries both coarse (bilinear baseline) and
    Topo-SRNet downscaled weather for comparison.
    """
    if block_id not in _BLOCK_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Block '{block_id}' not found. "
                   f"Available blocks: {list(_BLOCK_STORE.keys())}",
        )

    gps = _panchayats_for_block(block_id)
    if not gps:
        raise HTTPException(
            status_code=404,
            detail=f"No panchayats registered for block '{block_id}'.",
        )

    features: List[PanchayatFeature] = []
    for gp in gps:
        weather = PanchayatWeather(
            coarse=gp["coarse"],
            downscaled=gp["downscaled"],
            forecast_date=date,
            model_version=_MODEL_VERSION,
        )
        props = PanchayatProperties(
            panchayat_id=gp["gid"],
            panchayat_name=gp["gname"],
            block_id=gp["block_id"],
            block_name=gp["bname"],
            district=gp["district"],
            state="Rajasthan",
            population=gp["pop"],
            area_km2=gp["area"],
            dominant_crop=gp["crop"],
            elevation_m=gp["elev"],
            weather=weather,
        )
        feature = PanchayatFeature(
            type="Feature",
            id=gp["gid"],
            geometry=_make_gp_polygon(gp["glat"], gp["glon"]),
            properties=props,
        )
        features.append(feature)

    return PanchayatCollection(
        type="FeatureCollection",
        features=features,
        block_id=block_id,
        meta={
            "block_name": _BLOCK_STORE[block_id]["bname"],
            "forecast_date": date,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_panchayats": len(features),
            "model_version": _MODEL_VERSION,
        },
    )


@app.get(
    "/api/v1/advisory/{panchayat_id}",
    response_model=AdvisoryResponse,
    tags=["Advisory"],
    summary="Get GKMS bilingual agro-meteorological advisory for a panchayat",
    response_description="Bilingual ICAR advisory alerts for the given GP.",
)
def get_advisory(
    panchayat_id: str,
    date: str = Query(default=_FORECAST_DATE),
    lang: str = Query(
        default="both",
        description="Language filter: 'en', 'hi', or 'both'",
        pattern="^(en|hi|both)$",
    ),
    max_alerts: int = Query(
        default=5,
        description="Maximum number of alerts to return",
        ge=1,
        le=12,
    ),
) -> AdvisoryResponse:
    """
    Evaluates the GKMS advisory rule engine against the Topo-SRNet downscaled
    weather for the given Gram Panchayat and returns actionable bilingual alerts.

    - Alerts are sorted by severity: CRITICAL → HIGH → MODERATE → LOW.
    - Each alert carries both English and Hindi text.
    - Use `?lang=hi` to receive Hindi-only messages (English fields still present
      in the schema for API compatibility).
    """
    gp = _find_panchayat(panchayat_id)
    if gp is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Panchayat '{panchayat_id}' not found. "
                "Check /api/v1/blocks for valid block IDs, then "
                "/api/v1/panchayats/{{block_id}} for GP IDs."
            ),
        )

    ds = gp["downscaled"]

    # Build weather dict for the rule engine (physical units from downscaled)
    weather_dict = {
        "tmax":     ds.tmax_c,
        "tmin":     ds.tmin_c,
        "rainfall": ds.rainfall_mm,
        "rh":       ds.rh_pct,
        "wind":     ds.wind_ms,
    }

    raw_alerts = evaluate_advisory(
        weather_dict,
        panchayat_id=panchayat_id,
        max_alerts=max_alerts,
    )

    # Convert to schema objects
    alert_schemas: List[AdvisoryAlertSchema] = []
    for a in raw_alerts:
        alert_schemas.append(
            AdvisoryAlertSchema(
                alert_id=a["alert_id"],
                category=a["category"],
                severity=a["severity"],
                message_en=a["message_en"],
                message_hi=a["message_hi"],
                action_en=a["action_en"],
                action_hi=a["action_hi"],
                triggered_by=a["triggered_by"],
            )
        )

    return AdvisoryResponse(
        panchayat_id=panchayat_id,
        panchayat_name=gp["gname"],
        block_id=gp["block_id"],
        district=gp["district"],
        forecast_date=date,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        weather_summary=ds,
        alerts=alert_schemas,
        n_alerts=len(alert_schemas),
    )


# ---------------------------------------------------------------------------
# Static file serving — built React dashboard
# ---------------------------------------------------------------------------
# Resolve dashboard/dist relative to this file's location:
#   src/api/main.py  →  ../../dashboard/dist
_THIS_DIR      = Path(__file__).resolve().parent          # src/api/
_REPO_ROOT     = _THIS_DIR.parent.parent                  # project root
DASHBOARD_DIST = _REPO_ROOT / "dashboard" / "dist"

# Mount the Vite-compiled asset bundle (JS, CSS, sourcemaps, fonts).
# This must be registered BEFORE the SPA catch-all route.
if DASHBOARD_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=DASHBOARD_DIST / "assets"),
        name="dashboard-assets",
    )


@app.get("/", include_in_schema=False, response_model=None)
def root():
    """Serve the dashboard index.html, or API info if not yet built."""
    index = DASHBOARD_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        content={
            "message": "Topo-Downscale API v1.0 — dashboard not built yet.",
            "hint": "Run:  cd dashboard && npm run build",
            "docs": "http://localhost:8000/docs",
            "endpoints": [
                "GET /health",
                "GET /api/v1/blocks",
                "GET /api/v1/panchayats/{block_id}",
                "GET /api/v1/advisory/{panchayat_id}",
            ],
        }
    )


# SPA catch-all — must be the LAST route registered.
# Handles React Router client-side paths (e.g. /map, /advisory) that are not
# API routes. Any path that didn't match an earlier route lands here and
# receives index.html so the React app can handle its own routing.
@app.get("/{full_path:path}", include_in_schema=False, response_model=None)
def serve_spa(full_path: str):
    index = DASHBOARD_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {"detail": f"Path '/{full_path}' not found and dashboard is not built."},
        status_code=404,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # disable reload in production single-file mode
    )


