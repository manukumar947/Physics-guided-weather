"""
dem_processor.py
================
DEM-derived geophysical feature extractor for the Topo-Downscale pipeline.

Given a Digital Elevation Model (elevation array or xr.DataArray), this
module computes:

  * **Slope**  â€” steepness in degrees (0Â° = flat, 90Â° = vertical cliff)
  * **Aspect** â€” compass direction of the downhill face in degrees
                 (0Â° = North, 90Â° = East, 180Â° = South, 270Â° = West,
                  -1 = flat / undefined)

Both are computed via finite-difference gradient operators applied to the
elevation grid, following the standard ArcGIS / GDAL methodology.

Public API
----------
    compute_slope(elevation, cell_size_m) -> np.ndarray
    compute_aspect(elevation, cell_size_m) -> np.ndarray
    process_dem(ds_dem, out_path)          -> xr.Dataset

Usage
-----
    python src/etl/dem_processor.py [--dem_path data/raw/dem_30m.nc]
                                    [--out_path data/processed/dem_features.nc]

Dependencies
------------
    numpy, xarray, argparse
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Core gradient computation
# ---------------------------------------------------------------------------

def _gradient_xy(
    elevation: np.ndarray,
    cell_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eastâ€“west (dz/dx) and northâ€“south (dz/dy) elevation
    gradients using second-order central differences.

    The 3Ã—3 Horn (1981) algorithm is used for interior cells to maximise
    accuracy on noisy terrain.

    Parameters
    ----------
    elevation : np.ndarray, shape (H, W)
        Elevation values in metres.
    cell_size_m : float
        Grid cell size in metres.  For geographic grids use the equivalent
        resolution (e.g., 1000 m for 0.01Â° at ~27Â°N).

    Returns
    -------
    dzdx : np.ndarray
        Eastâ€“west gradient (positive = uphill to the east).
    dzdy : np.ndarray
        Northâ€“south gradient (positive = uphill to the north).
    """
    # Pad with edge replication so output shape = input shape
    elev_pad = np.pad(elevation, pad_width=1, mode="edge")

    # Horn (1981) weighted finite differences
    # dz/dx = [(c + 2f + i) - (a + 2d + g)] / (8 * cell_size)
    # dz/dy = [(a + 2b + c) - (g + 2h + i)] / (8 * cell_size)
    a = elev_pad[:-2, :-2]   # NW
    b = elev_pad[:-2, 1:-1]  # N
    c = elev_pad[:-2, 2:]    # NE
    d = elev_pad[1:-1, :-2]  # W
    # e = elev_pad[1:-1, 1:-1]  # centre (unused in Horn formula)
    f = elev_pad[1:-1, 2:]   # E
    g = elev_pad[2:, :-2]    # SW
    h = elev_pad[2:, 1:-1]   # S
    i = elev_pad[2:, 2:]     # SE

    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_size_m)
    dzdy = ((a + 2.0 * b + c) - (g + 2.0 * h + i)) / (8.0 * cell_size_m)

    return dzdx.astype(np.float32), dzdy.astype(np.float32)


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------

def compute_slope(
    elevation: np.ndarray,
    cell_size_m: float = 1000.0,
) -> np.ndarray:
    """
    Compute terrain slope in degrees from an elevation grid.

    Slope = arctan(sqrt((dz/dx)Â² + (dz/dy)Â²)) Ã— (180/Ï€)

    Parameters
    ----------
    elevation : np.ndarray, shape (H, W)
        Elevation in metres.
    cell_size_m : float
        Horizontal resolution in metres (default 1000 m for 1-km grid).

    Returns
    -------
    slope : np.ndarray, shape (H, W), dtype float32
        Slope in degrees [0Â°, 90Â°].
    """
    dzdx, dzdy = _gradient_xy(elevation, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return slope_deg


# ---------------------------------------------------------------------------
# Aspect
# ---------------------------------------------------------------------------

def compute_aspect(
    elevation: np.ndarray,
    cell_size_m: float = 1000.0,
) -> np.ndarray:
    """
    Compute terrain aspect in degrees (compass bearing of the downhill face).

    Convention
    ----------
    * 0Â°   = North-facing (slope descends toward North)
    * 90Â°  = East-facing
    * 180Â° = South-facing
    * 270Â° = West-facing
    * -1   = Flat / undefined (slope < 1e-6)

    The formula follows ESRI ArcGIS standard:
        aspect = 180Â° - arctan2(dz/dy, -dz/dx) + 90Â° Ã— (dz/dx / |dz/dx|)

    simplified as:
        cell_aspect = arctan2(dzdx, -dzdy) converted to compass bearing

    Parameters
    ----------
    elevation : np.ndarray, shape (H, W)
        Elevation in metres.
    cell_size_m : float
        Horizontal resolution in metres.

    Returns
    -------
    aspect : np.ndarray, shape (H, W), dtype float32
        Aspect in degrees [0Â°, 360Â°), or -1 for flat areas.
    """
    dzdx, dzdy = _gradient_xy(elevation, cell_size_m)

    # arctan2 gives the mathematical angle; convert to compass bearing
    aspect_rad = np.arctan2(dzdx, -dzdy)
    aspect_deg = np.degrees(aspect_rad)

    # Shift to [0Â°, 360Â°)
    aspect_deg = (aspect_deg + 360.0) % 360.0

    # Flat cells â†’ -1
    magnitude = np.sqrt(dzdx**2 + dzdy**2)
    flat_mask = magnitude < 1e-6
    aspect_deg[flat_mask] = -1.0

    return aspect_deg.astype(np.float32)


# ---------------------------------------------------------------------------
# Topographic Wetness Index (bonus feature for physics-informed loss)
# ---------------------------------------------------------------------------

def compute_twi(
    elevation: np.ndarray,
    cell_size_m: float = 1000.0,
    min_area: float = 1.0,
) -> np.ndarray:
    """
    Compute the Topographic Wetness Index (TWI) as a proxy for soil
    moisture redistribution.

    TWI = ln(a / tan(Î²))

    where:
        a    = upslope contributing area per unit contour length (m)
               approximated here as cell_size_m (no flow-accumulation model)
        Î²    = slope in radians

    Note: This is a simplified single-cell approximation. A full TWI
    implementation requires D8 flow accumulation, which is outside
    the scope of the MVP. This version is sufficient for conditioning
    the neural network.

    Parameters
    ----------
    elevation : np.ndarray, shape (H, W)
    cell_size_m : float
    min_area : float
        Minimum contributing area to avoid log(0) on flat cells.

    Returns
    -------
    twi : np.ndarray, shape (H, W), dtype float32
    """
    dzdx, dzdy = _gradient_xy(elevation, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    tan_slope = np.tan(slope_rad)
    # Avoid division by zero on flat areas
    tan_slope = np.where(tan_slope < 1e-4, 1e-4, tan_slope)
    # Contributing area proxy (uniform = cell area)
    a = np.full_like(elevation, fill_value=cell_size_m, dtype=np.float32)
    a = np.maximum(a, min_area)
    twi = np.log(a / tan_slope).astype(np.float32)
    return twi


# ---------------------------------------------------------------------------
# High-level processor
# ---------------------------------------------------------------------------

def process_dem(
    ds_dem: xr.Dataset,
    out_path: Path,
    cell_size_m: float = 1000.0,
) -> xr.Dataset:
    """
    Derive slope, aspect, and TWI from an elevation xr.Dataset and save
    the combined feature set as a new NetCDF.

    Parameters
    ----------
    ds_dem : xr.Dataset
        Dataset containing an ``elevation`` variable on (lat, lon) coords.
    out_path : Path
        Output .nc file path.
    cell_size_m : float
        Approximate horizontal resolution in metres.

    Returns
    -------
    xr.Dataset
        Dataset containing {elevation, slope, aspect, twi}.
    """
    elev = ds_dem["elevation"].values  # (H, W)

    print("[*] Computing slope â€¦")
    slope = compute_slope(elev, cell_size_m=cell_size_m)

    print("[*] Computing aspect â€¦")
    aspect = compute_aspect(elev, cell_size_m=cell_size_m)

    print("[*] Computing Topographic Wetness Index â€¦")
    twi = compute_twi(elev, cell_size_m=cell_size_m)

    coords = {"lat": ds_dem["lat"], "lon": ds_dem["lon"]}
    dims = ("lat", "lon")

    ds_features = xr.Dataset(
        {
            "elevation": xr.Variable(
                dims, elev,
                attrs={
                    "long_name": "Surface Elevation",
                    "units": "m",
                    "standard_name": "surface_altitude",
                },
            ),
            "slope": xr.Variable(
                dims, slope,
                attrs={
                    "long_name": "Terrain Slope (Horn 1981)",
                    "units": "degrees",
                    "valid_range": [0.0, 90.0],
                },
            ),
            "aspect": xr.Variable(
                dims, aspect,
                attrs={
                    "long_name": "Terrain Aspect (downhill compass bearing)",
                    "units": "degrees",
                    "valid_range": [-1.0, 360.0],
                    "note": "-1 indicates flat / undefined aspect",
                },
            ),
            "twi": xr.Variable(
                dims, twi,
                attrs={
                    "long_name": "Topographic Wetness Index (simplified)",
                    "units": "1",
                },
            ),
        },
        coords=coords,
        attrs={
            **ds_dem.attrs,
            "title": "Topo-Downscale DEM-Derived Geophysical Features",
            "created_by": "dem_processor.py",
            "cell_size_m": cell_size_m,
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_features.to_netcdf(out_path, format="NETCDF4")

    print(f"[âœ“] DEM features saved â†’ {out_path}")
    print(f"    Slope  : {slope.min():.2f}Â° â€“ {slope.max():.2f}Â°")
    print(f"    Aspect : {aspect[aspect >= 0].min():.1f}Â° â€“ {aspect.max():.1f}Â°")
    print(f"    TWI    : {twi.min():.2f} â€“ {twi.max():.2f}")

    return ds_features


# ---------------------------------------------------------------------------
# Utility: degrees-to-radians aspect normalisation (for model input)
# ---------------------------------------------------------------------------

def normalise_aspect_circular(aspect_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert aspect (degrees, [-1, 360]) into two circular features:
        sin(aspect_rad) and cos(aspect_rad)

    This avoids the 0Â°/360Â° discontinuity that would confuse a neural
    network trying to learn directional preferences.

    Flat cells (aspect == -1) are mapped to (sin=0, cos=0).

    Parameters
    ----------
    aspect_deg : np.ndarray

    Returns
    -------
    (sin_aspect, cos_aspect) : Tuple[np.ndarray, np.ndarray]
    """
    flat = aspect_deg < 0
    rad = np.where(flat, 0.0, np.radians(aspect_deg))
    sin_a = np.where(flat, 0.0, np.sin(rad)).astype(np.float32)
    cos_a = np.where(flat, 0.0, np.cos(rad)).astype(np.float32)
    return sin_a, cos_a


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute slope, aspect, and TWI from a DEM NetCDF."
    )
    parser.add_argument(
        "--dem_path",
        type=str,
        default="data/raw/dem_30m.nc",
        help="Path to the input DEM NetCDF (default: data/raw/dem_30m.nc)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/dem_features.nc",
        help="Output NetCDF path (default: data/processed/dem_features.nc)",
    )
    parser.add_argument(
        "--cell_size_m",
        type=float,
        default=1000.0,
        help="Approximate horizontal resolution in metres (default: 1000.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(" Topo-Downscale â€” DEM Processor")
    print(f" Input  : {args.dem_path}")
    print(f" Output : {args.out_path}")
    print(f" Cell   : {args.cell_size_m} m")
    print("=" * 60)

    dem_path = Path(args.dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM file not found: {dem_path}\n"
            "Run generate_sample_data.py first."
        )

    ds_dem = xr.open_dataset(dem_path)
    process_dem(ds_dem, out_path=Path(args.out_path), cell_size_m=args.cell_size_m)

    print("=" * 60)
    print("[âœ“] DEM processing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

