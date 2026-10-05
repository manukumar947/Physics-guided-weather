"""
schemas.py
==========
Pydantic v2 data models for the Topo-Downscale API.

Schema hierarchy
----------------

    GeoJSON-compatible layer:
        BlockProperties          → properties of a Block Feature
        BlockFeature             → GeoJSON Feature (Block)
        BlockCollection          → GeoJSON FeatureCollection

        PanchayatWeather         → coarse vs. downscaled weather payload
        PanchayatProperties      → properties of a Panchayat Feature
        PanchayatFeature         → GeoJSON Feature (Panchayat)
        PanchayatCollection      → GeoJSON FeatureCollection

    Advisory layer:
        AdvisoryAlertSchema      → single bilingual alert
        AdvisoryResponse         → full advisory payload for a panchayat

All models use ``model_config = ConfigDict(extra="forbid")`` to prevent
accidental field drift between API versions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared geo primitives
# ---------------------------------------------------------------------------

class Point(BaseModel):
    """GeoJSON Point geometry."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["Point"] = "Point"
    coordinates: Tuple[float, float] = Field(
        ...,
        description="[longitude, latitude] in WGS-84",
        examples=[[74.5, 28.5]],
    )


class Polygon(BaseModel):
    """GeoJSON Polygon geometry (exterior ring only, no holes for MVP)."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["Polygon"] = "Polygon"
    coordinates: List[List[Tuple[float, float]]] = Field(
        ...,
        description="List of rings; first ring is the exterior boundary.",
    )


# ---------------------------------------------------------------------------
# Weather payload models
# ---------------------------------------------------------------------------

class CoarseWeather(BaseModel):
    """
    Block-level coarse NWP weather (12–25 km resolution).
    All values are in physical units (not normalised).
    """
    model_config = ConfigDict(extra="forbid")

    tmax_c: float = Field(..., description="Maximum 2-m temperature (°C)", ge=-10, le=60)
    tmin_c: float = Field(..., description="Minimum 2-m temperature (°C)", ge=-20, le=50)
    rainfall_mm: float = Field(..., description="Daily accumulated rainfall (mm)", ge=0)
    rh_pct: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    wind_ms: float = Field(..., description="Surface wind speed (m/s)", ge=0)

    @field_validator("tmin_c")
    @classmethod
    def tmin_less_than_tmax(cls, v: float, info: Any) -> float:
        """Sanity check: Tmin must be less than Tmax."""
        tmax = (info.data or {}).get("tmax_c")
        if tmax is not None and v >= tmax:
            raise ValueError(
                f"tmin_c ({v}) must be less than tmax_c ({tmax})"
            )
        return v


class DownscaledWeather(BaseModel):
    """
    Panchayat-level downscaled weather (1 km resolution, Topo-SRNet output).
    Inherits the same units as CoarseWeather.
    """
    model_config = ConfigDict(extra="forbid")

    tmax_c: float = Field(..., description="Downscaled maximum temperature (°C)")
    tmin_c: float = Field(..., description="Downscaled minimum temperature (°C)")
    rainfall_mm: float = Field(..., description="Downscaled rainfall (mm)", ge=0)
    rh_pct: float = Field(..., description="Downscaled relative humidity (%)", ge=0, le=100)
    wind_ms: float = Field(..., description="Downscaled wind speed (m/s)", ge=0)

    # Topographic correction deltas (pred - coarse bilinear)
    tmax_delta_c: Optional[float] = Field(
        None, description="Tmax correction from bilinear baseline (°C)"
    )
    rainfall_delta_mm: Optional[float] = Field(
        None, description="Rainfall correction from bilinear baseline (mm)"
    )


class PanchayatWeather(BaseModel):
    """
    Paired coarse vs. downscaled weather for a single Panchayat.
    Returned by GET /api/v1/panchayats/{block_id}.
    """
    model_config = ConfigDict(extra="forbid")

    coarse: CoarseWeather = Field(..., description="Block-level NWP weather (bilinear upsample)")
    downscaled: DownscaledWeather = Field(
        ..., description="Topo-SRNet downscaled panchayat-level weather"
    )
    forecast_date: str = Field(
        ..., description="Forecast valid date (YYYY-MM-DD)", examples=["2024-06-15"]
    )
    model_version: str = Field(
        "topo-srnet-v1", description="Model checkpoint identifier"
    )


# ---------------------------------------------------------------------------
# Block schemas (GeoJSON Feature / FeatureCollection)
# ---------------------------------------------------------------------------

class BlockProperties(BaseModel):
    """Properties carried on each Block GeoJSON Feature."""
    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(..., description="Unique block identifier", examples=["RJ_BL_001"])
    block_name: str = Field(..., description="Block name", examples=["Churu Block"])
    district: str = Field(..., description="District name", examples=["Churu"])
    state: str = Field(default="Rajasthan", description="State name")
    n_panchayats: int = Field(..., description="Number of Gram Panchayats in this block", ge=1)
    centroid_lat: float = Field(..., description="Block centroid latitude")
    centroid_lon: float = Field(..., description="Block centroid longitude")
    weather: CoarseWeather = Field(..., description="Representative coarse NWP weather")


class BlockFeature(BaseModel):
    """GeoJSON Feature representing a single administrative Block."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["Feature"] = "Feature"
    id: str = Field(..., description="Feature ID (== block_id)")
    geometry: Polygon = Field(..., description="Block boundary polygon")
    properties: BlockProperties


class BlockCollection(BaseModel):
    """GeoJSON FeatureCollection of all available Blocks."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[BlockFeature] = Field(..., description="List of Block features")
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata (region, generated_at, etc.)",
    )


# ---------------------------------------------------------------------------
# Panchayat schemas (GeoJSON Feature / FeatureCollection)
# ---------------------------------------------------------------------------

class PanchayatProperties(BaseModel):
    """Properties carried on each Panchayat GeoJSON Feature."""
    model_config = ConfigDict(extra="forbid")

    panchayat_id: str = Field(
        ..., description="Unique Gram Panchayat identifier", examples=["RJ_CH_001"]
    )
    panchayat_name: str = Field(
        ..., description="GP name", examples=["Taranagar GP"]
    )
    block_id: str = Field(..., description="Parent block identifier")
    block_name: str = Field(..., description="Parent block name")
    district: str
    state: str = "Rajasthan"
    population: Optional[int] = Field(None, description="Population (census)")
    area_km2: Optional[float] = Field(None, description="Area in km²")
    dominant_crop: Optional[str] = Field(
        None, description="Dominant Kharif/Rabi crop", examples=["Pearl Millet"]
    )
    elevation_m: Optional[float] = Field(None, description="Mean elevation (m)")
    weather: PanchayatWeather = Field(..., description="Coarse + downscaled weather pair")


class PanchayatFeature(BaseModel):
    """GeoJSON Feature representing a single Gram Panchayat."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["Feature"] = "Feature"
    id: str = Field(..., description="Feature ID (== panchayat_id)")
    geometry: Polygon = Field(..., description="GP boundary polygon")
    properties: PanchayatProperties


class PanchayatCollection(BaseModel):
    """GeoJSON FeatureCollection of Panchayats belonging to one Block."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[PanchayatFeature]
    block_id: str = Field(..., description="Parent block identifier")
    meta: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Advisory schemas
# ---------------------------------------------------------------------------

class AdvisoryAlertSchema(BaseModel):
    """Single bilingual agronomic advisory alert."""
    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(..., description="Stable rule identifier", examples=["RAIN_HEAVY"])
    category: str = Field(
        ...,
        description="Alert category",
        examples=["Rainfall", "Temperature", "Pest-Disease", "Irrigation", "Composite"],
    )
    severity: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] = Field(
        ..., description="Alert severity level"
    )
    message_en: str = Field(..., description="Advisory message in English")
    message_hi: str = Field(..., description="Advisory message in Hindi (Devanagari)")
    action_en: str = Field(..., description="Recommended action in English")
    action_hi: str = Field(..., description="Recommended action in Hindi")
    triggered_by: Dict[str, Any] = Field(
        default_factory=dict,
        description="Weather values that triggered this rule",
    )


class AdvisoryResponse(BaseModel):
    """Full advisory payload for a Gram Panchayat."""
    model_config = ConfigDict(extra="forbid")

    panchayat_id: str
    panchayat_name: str
    block_id: str
    district: str
    forecast_date: str = Field(..., description="Advisory valid date (YYYY-MM-DD)")
    generated_at: str = Field(..., description="UTC ISO-8601 generation timestamp")
    weather_summary: DownscaledWeather = Field(
        ..., description="Downscaled weather used to generate advisories"
    )
    alerts: List[AdvisoryAlertSchema] = Field(
        ..., description="List of alerts, sorted by severity descending"
    )
    n_alerts: int = Field(..., description="Total number of alerts fired")


# ---------------------------------------------------------------------------
# Health check schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for GET /health."""
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"] = "ok"
    version: str = "1.0.0"
    model: str = "topo-srnet-v1"
    region: str = "north_rajasthan_pilot"
