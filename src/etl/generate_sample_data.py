"""
generate_sample_data.py
=======================
Synthetic data generator for the Topo-Downscale pipeline.

Generates:
  1. Coarse-resolution (25 km / ~0.25Â°) weather NetCDF grids for 5 variables:
       Tmax, Tmin, Rainfall, RH, Wind
  2. A mock 30 m SRTM-like DEM array (~1 km equivalent grid) for the
     "north_rajasthan_pilot" pilot region.

Output directory (default): data/raw/
  - data/raw/weather_coarse.nc
  - data/raw/dem_30m.nc

Usage
-----
    python src/etl/generate_sample_data.py [--out_dir data/raw] [--seed 42]

Dependencies
------------
    numpy, xarray, scipy (optional, for smoothing), argparse
"""

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Region definition â€” North Rajasthan pilot bounding box
# ---------------------------------------------------------------------------
REGION = {
    "name": "north_rajasthan_pilot",
    "lat_min": 27.0,
    "lat_max": 30.0,
    "lon_min": 72.0,
    "lon_max": 76.0,
}

# Coarse grid resolution (~0.25 degrees â‰ˆ 25 km at this latitude)
COARSE_RES_DEG = 0.25

# Fine grid resolution (~0.01 degrees â‰ˆ 1 km at this latitude)
FINE_RES_DEG = 0.01

# Number of time steps (days)
N_DAYS = 30

# Random seed for reproducibility
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Helper: smooth random field using a simple Gaussian-like kernel
# ---------------------------------------------------------------------------

def _smooth_field(arr: np.ndarray, sigma: int = 2) -> np.ndarray:
    """
    Apply a basic box-blur to add spatial coherence to random fields.

    Parameters
    ----------
    arr : np.ndarray
        2-D array to smooth.
    sigma : int
        Half-width of the moving average kernel (in grid cells).

    Returns
    -------
    np.ndarray
        Smoothed array of the same shape.
    """
    from scipy.ndimage import uniform_filter  # lazy import

    return uniform_filter(arr, size=sigma * 2 + 1, mode="nearest")


def _try_smooth(arr: np.ndarray, sigma: int = 2) -> np.ndarray:
    """Smooth if scipy is available, otherwise return raw array."""
    try:
        return _smooth_field(arr, sigma=sigma)
    except ImportError:
        return arr


# ---------------------------------------------------------------------------
# Coarse weather grid generator
# ---------------------------------------------------------------------------

def generate_coarse_weather(
    rng: np.random.Generator,
    out_path: Path,
) -> xr.Dataset:
    """
    Generate a synthetic coarse-resolution NWP weather dataset.

    Variables
    ---------
    Tmax : float32, Â°C
        Maximum 2-m temperature (base ~35Â°C in summer, Rajasthan)
    Tmin : float32, Â°C
        Minimum 2-m temperature (Tmax - 8 to 12Â°C diurnal range)
    Rainfall : float32, mm/day
        Daily accumulated precipitation (right-skewed, mostly dry)
    RH : float32, %
        Relative humidity (0â€“100)
    Wind : float32, m/s
        Surface wind speed (2â€“15 m/s, desert regime)

    Grid
    ----
    Spatial  : COARSE_RES_DEG spacing over pilot bbox
    Temporal : N_DAYS daily time steps starting 2024-06-01

    Parameters
    ----------
    rng : np.random.Generator
        Seeded NumPy RNG for reproducibility.
    out_path : Path
        Destination file path (.nc).

    Returns
    -------
    xr.Dataset
        The generated dataset (also written to ``out_path``).
    """
    lats = np.arange(REGION["lat_min"], REGION["lat_max"], COARSE_RES_DEG, dtype=np.float32)
    lons = np.arange(REGION["lon_min"], REGION["lon_max"], COARSE_RES_DEG, dtype=np.float32)
    start = np.datetime64("2024-06-01", "D")
    times = np.array([start + i for i in range(N_DAYS)], dtype="datetime64[D]")

    nlat, nlon, ntime = len(lats), len(lons), len(times)

    # --- Tmax: base 35Â°C + spatial trend (hotter in the west desert) + time ---
    tmax_base = 35.0 + 3.0 * (1.0 - (lons[np.newaxis, :] - lons.min()) / (lons.max() - lons.min()))
    tmax_base = np.broadcast_to(tmax_base, (nlat, nlon))
    tmax = (
        tmax_base[np.newaxis, :, :]
        + rng.normal(0, 1.5, size=(ntime, nlat, nlon)).astype(np.float32)
        + np.linspace(0, -5, ntime)[:, np.newaxis, np.newaxis]  # slight cooling trend
    )
    tmax = np.stack([_try_smooth(tmax[t]) for t in range(ntime)], axis=0).astype(np.float32)

    # --- Tmin: Tmax minus a random diurnal range ---
    diurnal = rng.uniform(8, 12, size=(ntime, nlat, nlon)).astype(np.float32)
    tmin = (tmax - diurnal).astype(np.float32)

    # --- Rainfall: sparse, right-skewed (monsoon onset) ---
    rain_prob = rng.uniform(0, 1, size=(ntime, nlat, nlon))
    rainfall = np.where(
        rain_prob > 0.70,  # ~30% wet probability per cell-day
        rng.exponential(scale=8.0, size=(ntime, nlat, nlon)),
        0.0,
    ).astype(np.float32)
    rainfall = np.stack([_try_smooth(rainfall[t], sigma=1) for t in range(ntime)], axis=0)
    rainfall = np.clip(rainfall, 0, None).astype(np.float32)

    # --- RH: correlated with rainfall, 30â€“90% range ---
    rh_base = 40.0 + 30.0 * (rainfall / (rainfall.max() + 1e-6))
    rh = (rh_base + rng.normal(0, 5, size=(ntime, nlat, nlon))).astype(np.float32)
    rh = np.clip(rh, 20.0, 98.0).astype(np.float32)

    # --- Wind: 2â€“15 m/s, stronger in the west ---
    wind_base = 5.0 + 4.0 * (1.0 - (lons[np.newaxis, :] - lons.min()) / (lons.max() - lons.min()))
    wind_base = np.broadcast_to(wind_base[np.newaxis, :, :], (ntime, nlat, nlon))
    wind = (wind_base + rng.normal(0, 1.5, size=(ntime, nlat, nlon))).astype(np.float32)
    wind = np.clip(wind, 0.5, 20.0).astype(np.float32)

    # --- Assemble xarray Dataset ---
    coords = {
        "time": times,
        "lat": lats,
        "lon": lons,
    }
    dims = ("time", "lat", "lon")

    ds = xr.Dataset(
        {
            "Tmax": xr.Variable(dims, tmax, attrs={
                "long_name": "Maximum 2-m Air Temperature",
                "units": "degC",
                "standard_name": "air_temperature",
            }),
            "Tmin": xr.Variable(dims, tmin, attrs={
                "long_name": "Minimum 2-m Air Temperature",
                "units": "degC",
                "standard_name": "air_temperature",
            }),
            "Rainfall": xr.Variable(dims, rainfall, attrs={
                "long_name": "Daily Accumulated Precipitation",
                "units": "mm/day",
                "standard_name": "precipitation_flux",
            }),
            "RH": xr.Variable(dims, rh, attrs={
                "long_name": "Relative Humidity",
                "units": "%",
                "standard_name": "relative_humidity",
            }),
            "Wind": xr.Variable(dims, wind, attrs={
                "long_name": "Surface Wind Speed",
                "units": "m/s",
                "standard_name": "wind_speed",
            }),
        },
        coords=coords,
        attrs={
            "title": "Topo-Downscale Synthetic Coarse NWP Grid",
            "region": REGION["name"],
            "resolution": f"{COARSE_RES_DEG}Â° (~25 km)",
            "created_by": "generate_sample_data.py",
            "institution": "SIH 2024 â€” Topo-Downscale Team",
            "Conventions": "CF-1.8",
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, format="NETCDF4")
    print(f"[âœ“] Coarse weather grid saved â†’ {out_path}")
    print(f"    Shape: {nlat} lat Ã— {nlon} lon Ã— {ntime} time steps")
    return ds


# ---------------------------------------------------------------------------
# Mock 30 m SRTM DEM generator
# ---------------------------------------------------------------------------

def generate_dem(
    rng: np.random.Generator,
    out_path: Path,
) -> xr.Dataset:
    """
    Generate a synthetic 30 m SRTM-like Digital Elevation Model.

    The DEM is generated at FINE_RES_DEG (~1 km) spacing using multi-scale
    Perlin-like noise to mimic realistic semi-arid topography (Aravalli foothills
    in eastern extent, flat alluvial plains in the west).

    Variables
    ---------
    elevation : float32, metres
        Surface elevation above mean sea level.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded NumPy RNG for reproducibility.
    out_path : Path
        Destination file path (.nc).

    Returns
    -------
    xr.Dataset
        The generated DEM dataset (also written to ``out_path``).
    """
    lats = np.arange(REGION["lat_min"], REGION["lat_max"], FINE_RES_DEG, dtype=np.float32)
    lons = np.arange(REGION["lon_min"], REGION["lon_max"], FINE_RES_DEG, dtype=np.float32)

    nlat, nlon = len(lats), len(lons)

    # ---- Multi-scale terrain synthesis ----
    # Layer 1: Large-scale gradient (Aravalli range enters from SE â†’ lower elevation NW)
    lon_norm = (lons - lons.min()) / (lons.max() - lons.min())
    lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
    lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm)

    base_elevation = (
        200.0                                  # base plains (metres)
        + 400.0 * lat_grid                     # higher in the north (Siwalik foothills)
        + 300.0 * (1.0 - lon_grid)            # higher in the east (Aravallis)
    )

    # Layer 2: Meso-scale ridges and valleys (large noise)
    noise_large = rng.normal(0, 60.0, size=(nlat, nlon))
    noise_large = _try_smooth(noise_large, sigma=20)

    # Layer 3: Fine-scale micro-terrain
    noise_fine = rng.normal(0, 20.0, size=(nlat, nlon))
    noise_fine = _try_smooth(noise_fine, sigma=3)

    elevation = (base_elevation + noise_large + noise_fine).astype(np.float32)
    elevation = np.clip(elevation, 10.0, 1200.0)  # realistic range for NW Rajasthan

    ds_dem = xr.Dataset(
        {
            "elevation": xr.Variable(
                ("lat", "lon"),
                elevation,
                attrs={
                    "long_name": "Surface Elevation (mock SRTM 30m)",
                    "units": "m",
                    "standard_name": "surface_altitude",
                    "source": "Synthetic â€” multi-scale noise on regional gradient",
                },
            )
        },
        coords={"lat": lats, "lon": lons},
        attrs={
            "title": "Topo-Downscale Synthetic SRTM-like DEM",
            "region": REGION["name"],
            "resolution": f"{FINE_RES_DEG}Â° (~1 km)",
            "created_by": "generate_sample_data.py",
            "institution": "SIH 2024 â€” Topo-Downscale Team",
            "Conventions": "CF-1.8",
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_dem.to_netcdf(out_path, format="NETCDF4")
    print(f"[âœ“] Mock SRTM DEM saved       â†’ {out_path}")
    print(f"    Shape: {nlat} lat Ã— {nlon} lon | Elev range: {elevation.min():.1f}â€“{elevation.max():.1f} m")
    return ds_dem


# ---------------------------------------------------------------------------
# Mock LULC (Land Use / Land Cover) grid
# ---------------------------------------------------------------------------

def generate_lulc(
    rng: np.random.Generator,
    out_path: Path,
) -> xr.Dataset:
    """
    Generate a synthetic LULC classification grid at fine resolution.

    Classes (IGBP-inspired for Rajasthan)
    ------
    0 : Water body / Wetland
    1 : Dense cropland (Kharif / Rabi)
    2 : Sparse cropland / Fallow
    3 : Scrubland / Degraded
    4 : Sand dunes / Desert barren
    5 : Urban / Built-up
    6 : Grassland / Pasture

    Parameters
    ----------
    rng : np.random.Generator
    out_path : Path

    Returns
    -------
    xr.Dataset
    """
    lats = np.arange(REGION["lat_min"], REGION["lat_max"], FINE_RES_DEG, dtype=np.float32)
    lons = np.arange(REGION["lon_min"], REGION["lon_max"], FINE_RES_DEG, dtype=np.float32)

    nlat, nlon = len(lats), len(lons)

    # Generate spatially coherent LULC using random patches
    base = rng.integers(0, 7, size=(nlat, nlon)).astype(np.uint8)
    # Smooth then re-quantise to get patch-like regions
    from scipy.ndimage import uniform_filter as _uf
    try:
        smoothed = _uf(base.astype(np.float32), size=15)
        lulc = np.round(smoothed * 6.0 / smoothed.max()).astype(np.uint8)
    except Exception:
        lulc = base

    ds_lulc = xr.Dataset(
        {
            "lulc": xr.Variable(
                ("lat", "lon"),
                lulc,
                attrs={
                    "long_name": "Land Use / Land Cover Class",
                    "units": "1",
                    "flag_values": "0,1,2,3,4,5,6",
                    "flag_meanings": (
                        "water dense_cropland sparse_cropland scrubland "
                        "desert urban grassland"
                    ),
                    "source": "Synthetic â€” spatially smoothed random patches",
                },
            )
        },
        coords={"lat": lats, "lon": lons},
        attrs={
            "title": "Topo-Downscale Synthetic LULC Grid",
            "region": REGION["name"],
            "resolution": f"{FINE_RES_DEG}Â° (~1 km)",
            "created_by": "generate_sample_data.py",
            "Conventions": "CF-1.8",
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_lulc.to_netcdf(out_path, format="NETCDF4")
    print(f"[âœ“] Synthetic LULC saved      â†’ {out_path}")
    return ds_lulc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ETL data for the Topo-Downscale pipeline."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/raw",
        help="Output directory for generated NetCDF files (default: data/raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print("=" * 60)
    print(" Topo-Downscale â€” Synthetic Data Generator")
    print(f" Region  : {REGION['name']}")
    print(f" Seed    : {args.seed}")
    print(f" Out dir : {out_dir.resolve()}")
    print("=" * 60)

    rng = np.random.default_rng(seed=args.seed)

    # 1. Coarse weather grids
    generate_coarse_weather(rng=rng, out_path=out_dir / "weather_coarse.nc")

    # 2. Mock SRTM DEM
    generate_dem(rng=rng, out_path=out_dir / "dem_30m.nc")

    # 3. Synthetic LULC
    generate_lulc(rng=rng, out_path=out_dir / "lulc.nc")

    print("=" * 60)
    print("[âœ“] All synthetic data generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

