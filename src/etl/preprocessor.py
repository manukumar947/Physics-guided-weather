"""
preprocessor.py
===============
Spatial alignment and differentiable preparation module for Topo-Downscale.

Responsibilities
----------------
1. **Upsampling** â€” Bilinearly interpolate coarse (~25 km) weather grids to
   the fine (~1 km) DEM grid so that meteorological and geophysical tensors
   share identical spatial coordinates.
2. **Normalization** â€” Per-variable minâ€“max (or z-score) normalisation to
   [0, 1] / N(0, 1) for model ingestion.
3. **Patch extraction** â€” Slice (lat, lon) tiles suitable for training
   (default 64 Ã— 64 fine-grid cells).
4. **Tensor export** â€” Save aligned, normalised arrays as .npz archives
   ready to be loaded by the PyTorch Dataset.

Pipeline
--------
    Raw coarse NetCDF  â†’  bilinear upsample  â†’  align with DEM grid
    DEM features NetCDF  â†’  (already fine-res)  â†’  stack
    LULC NetCDF          â†’  one-hot encode    â†’  stack
            â†“
    Combined tensor (C, H, W) per time step
            â†“
    Normalise â†’ extract patches â†’ save to data/processed/patches/

Output layout
-------------
    data/processed/
        â”œâ”€â”€ aligned_weather.nc    â€” coarse vars upsampled to fine grid
        â”œâ”€â”€ static_features.nc    â€” elevation, slope, aspect, twi, lulc
        â””â”€â”€ patches/
            â”œâ”€â”€ patch_000.npz
            â”œâ”€â”€ patch_001.npz
            â””â”€â”€ â€¦

Each .npz contains:
    "coarse" : (N_vars, H_patch, W_patch)  float32  â€” input to Topo-SRNet
    "static" : (N_static, H_patch, W_patch) float32 â€” static conditioning
    "norm_stats" : dict â€” per-variable (mean, std) for inverse transform

Usage
-----
    python src/etl/preprocessor.py \
        --coarse  data/raw/weather_coarse.nc \
        --dem     data/processed/dem_features.nc \
        --lulc    data/raw/lulc.nc \
        --out_dir data/processed

Dependencies
------------
    numpy, xarray, scipy (for zoom), argparse, pathlib
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Weather variables to process (must match generate_sample_data.py)
WEATHER_VARS: List[str] = ["Tmax", "Tmin", "Rainfall", "RH", "Wind"]

#: Static DEM-derived variables
STATIC_DEM_VARS: List[str] = ["elevation", "slope", "aspect", "twi"]

#: Patch size in fine-grid cells (64 Ã— 64 â‰ˆ 64 kmÂ² at 1-km resolution)
PATCH_SIZE: int = 64

#: Overlap fraction between patches (0.0 = no overlap, 0.5 = 50% stride)
PATCH_OVERLAP: float = 0.25

#: Number of LULC classes
N_LULC_CLASSES: int = 7


# ---------------------------------------------------------------------------
# Step 1: Bilinear upsampling of coarse weather grid
# ---------------------------------------------------------------------------

def upsample_weather_to_fine_grid(
    ds_coarse: xr.Dataset,
    fine_lats: np.ndarray,
    fine_lons: np.ndarray,
    method: str = "linear",
) -> xr.Dataset:
    """
    Bilinearly interpolate a coarse weather Dataset onto fine DEM grid
    coordinates using xarray's built-in ``interp`` (wraps scipy).

    This operation is the differentiable preparation step: during training,
    we replicate it inside PyTorch via ``torch.nn.functional.interpolate``
    (bilinear, align_corners=True). This function pre-computes the aligned
    reference so the ETL pipeline can save normalised targets to disk.

    Parameters
    ----------
    ds_coarse : xr.Dataset
        Coarse weather dataset with dims (time, lat, lon).
    fine_lats : np.ndarray
        Target latitude coordinates (fine grid).
    fine_lons : np.ndarray
        Target longitude coordinates (fine grid).
    method : str
        Interpolation method passed to xr.Dataset.interp
        ('linear', 'nearest', 'cubic').

    Returns
    -------
    xr.Dataset
        Weather variables upsampled to (time, fine_lat, fine_lon).
    """
    print(f"[*] Upsampling weather grids ({method}) â€¦")
    print(f"    Coarse: {len(ds_coarse.lat)} lat Ã— {len(ds_coarse.lon)} lon")
    print(f"    Fine  : {len(fine_lats)} lat Ã— {len(fine_lons)} lon")

    ds_up = ds_coarse.interp(
        lat=fine_lats,
        lon=fine_lons,
        method=method,
        kwargs={"fill_value": "extrapolate"},
    )
    print(f"[âœ“] Upsampled shape: {ds_up['Tmax'].shape}")
    return ds_up


# ---------------------------------------------------------------------------
# Step 2: One-hot encode LULC
# ---------------------------------------------------------------------------

def encode_lulc_onehot(
    ds_lulc: xr.Dataset,
    n_classes: int = N_LULC_CLASSES,
) -> np.ndarray:
    """
    One-hot encode the LULC class map into a (n_classes, H, W) float32 array.

    Parameters
    ----------
    ds_lulc : xr.Dataset
        Dataset with a ``lulc`` variable of integer class labels.
    n_classes : int
        Total number of classes.

    Returns
    -------
    onehot : np.ndarray, shape (n_classes, H, W), dtype float32
    """
    lulc = ds_lulc["lulc"].values.astype(np.int32)  # (H, W)
    H, W = lulc.shape
    onehot = np.zeros((n_classes, H, W), dtype=np.float32)
    for c in range(n_classes):
        onehot[c] = (lulc == c).astype(np.float32)
    return onehot


# ---------------------------------------------------------------------------
# Step 3: Normalisation utilities
# ---------------------------------------------------------------------------

class VariableNormalizer:
    """
    Per-variable normalisation with invertible transform.

    Two modes
    ---------
    ``minmax`` : maps [v_min, v_max] â†’ [0, 1]
    ``zscore`` : maps to N(0, 1) using (mean, std) computed over all pixels

    Parameters
    ----------
    mode : str
        'minmax' or 'zscore'
    """

    def __init__(self, mode: str = "minmax") -> None:
        if mode not in ("minmax", "zscore"):
            raise ValueError(f"mode must be 'minmax' or 'zscore', got '{mode}'")
        self.mode = mode
        self.stats: Dict[str, Dict[str, float]] = {}

    def fit(self, data: np.ndarray, name: str) -> None:
        """Compute normalisation statistics from a data array."""
        if self.mode == "minmax":
            self.stats[name] = {
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
            }
        else:
            self.stats[name] = {
                "mean": float(np.nanmean(data)),
                "std":  float(np.nanstd(data) + 1e-8),
            }

    def transform(self, data: np.ndarray, name: str) -> np.ndarray:
        """Apply normalisation using stored statistics."""
        s = self.stats[name]
        if self.mode == "minmax":
            denom = s["max"] - s["min"] + 1e-8
            return ((data - s["min"]) / denom).astype(np.float32)
        else:
            return ((data - s["mean"]) / s["std"]).astype(np.float32)

    def inverse_transform(self, data: np.ndarray, name: str) -> np.ndarray:
        """Invert normalisation to recover physical units."""
        s = self.stats[name]
        if self.mode == "minmax":
            return (data * (s["max"] - s["min"]) + s["min"]).astype(np.float32)
        else:
            return (data * s["std"] + s["mean"]).astype(np.float32)

    def save(self, path: Path) -> None:
        """Persist statistics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mode": self.mode, "stats": self.stats}, f, indent=2)
        print(f"[âœ“] Norm stats saved â†’ {path}")

    @classmethod
    def load(cls, path: Path) -> "VariableNormalizer":
        """Load normaliser from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        obj = cls(mode=data["mode"])
        obj.stats = data["stats"]
        return obj


# ---------------------------------------------------------------------------
# Step 4: Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(
    coarse_arr: np.ndarray,
    static_arr: np.ndarray,
    patch_size: int = PATCH_SIZE,
    overlap: float = PATCH_OVERLAP,
) -> List[Dict[str, np.ndarray]]:
    """
    Extract overlapping spatial patches from aligned weather and static arrays.

    Parameters
    ----------
    coarse_arr : np.ndarray, shape (C_weather, H, W)
        Normalised upsampled weather variables (single time step).
    static_arr : np.ndarray, shape (C_static, H, W)
        Normalised static geophysical features.
    patch_size : int
        Side length of square patch in pixels.
    overlap : float
        Overlap fraction [0, 1). Stride = patch_size Ã— (1 - overlap).

    Returns
    -------
    List[Dict]
        Each element: {"coarse": (C_w, P, P), "static": (C_s, P, P)}
    """
    _, H, W = coarse_arr.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))
    patches = []

    row = 0
    while row + patch_size <= H:
        col = 0
        while col + patch_size <= W:
            patches.append({
                "coarse": coarse_arr[:, row:row + patch_size, col:col + patch_size].copy(),
                "static": static_arr[:, row:row + patch_size, col:col + patch_size].copy(),
            })
            col += stride
        row += stride

    return patches


# ---------------------------------------------------------------------------
# Step 5: Assemble and save static feature tensor
# ---------------------------------------------------------------------------

def build_static_tensor(
    ds_dem_features: xr.Dataset,
    ds_lulc: xr.Dataset,
    normalizer: VariableNormalizer,
    fine_lats: Optional[np.ndarray] = None,
    fine_lons: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Assemble and normalise the static geophysical conditioning tensor.

    Output channels
    ---------------
    0 : elevation (normalised)
    1 : slope     (normalised)
    2 : sin(aspect)  â€” circular encoding
    3 : cos(aspect)  â€” circular encoding
    4 : twi       (normalised)
    5â€¦(5+N_LULC-1) : LULC one-hot channels

    Parameters
    ----------
    ds_dem_features : xr.Dataset
        Output of dem_processor.process_dem.
    ds_lulc : xr.Dataset
        LULC class map from generate_sample_data.generate_lulc.
    normalizer : VariableNormalizer
        Will be mutated in-place to store statistics for DEM vars.
    fine_lats, fine_lons : optional np.ndarray
        If provided, LULC will be resampled to match DEM grid exactly.

    Returns
    -------
    static_tensor : np.ndarray, shape (C_static, H, W), dtype float32
    """
    channels: List[np.ndarray] = []

    # --- DEM-derived continuous variables ---
    for var in ["elevation", "slope", "twi"]:
        arr = ds_dem_features[var].values.astype(np.float32)
        normalizer.fit(arr, name=var)
        channels.append(normalizer.transform(arr, name=var)[np.newaxis])

    # --- Aspect: circular encoding (no wrap discontinuity) ---
    aspect = ds_dem_features["aspect"].values.astype(np.float32)
    flat_mask = aspect < 0
    aspect_rad = np.where(flat_mask, 0.0, np.radians(aspect))
    sin_a = np.where(flat_mask, 0.0, np.sin(aspect_rad)).astype(np.float32)
    cos_a = np.where(flat_mask, 0.0, np.cos(aspect_rad)).astype(np.float32)
    channels.append(sin_a[np.newaxis])
    channels.append(cos_a[np.newaxis])

    # --- LULC one-hot ---
    H_dem = len(ds_dem_features["lat"])
    W_dem = len(ds_dem_features["lon"])

    # Align LULC to DEM grid if needed
    if fine_lats is not None and fine_lons is not None:
        ds_lulc_aligned = ds_lulc.interp(
            lat=fine_lats, lon=fine_lons, method="nearest"
        )
    else:
        ds_lulc_aligned = ds_lulc

    lulc_onehot = encode_lulc_onehot(ds_lulc_aligned)  # (7, H, W)

    # Ensure spatial dims match
    if lulc_onehot.shape[1:] != (H_dem, W_dem):
        # Fallback: resize with nearest-neighbour
        from scipy.ndimage import zoom as _zoom
        zy = H_dem / lulc_onehot.shape[1]
        zx = W_dem / lulc_onehot.shape[2]
        lulc_onehot = np.stack(
            [_zoom(lulc_onehot[c], (zy, zx), order=0) for c in range(N_LULC_CLASSES)],
            axis=0,
        ).astype(np.float32)

    channels.append(lulc_onehot)

    static_tensor = np.concatenate(channels, axis=0)
    print(f"[âœ“] Static tensor shape: {static_tensor.shape}  "
          f"(elev, slope, twi, sin_asp, cos_asp + {N_LULC_CLASSES} LULC classes)")
    return static_tensor


# ---------------------------------------------------------------------------
# Step 6: Align and save the full aligned weather NetCDF
# ---------------------------------------------------------------------------

def save_aligned_weather(
    ds_upsampled: xr.Dataset,
    normalizer: VariableNormalizer,
    out_path: Path,
) -> xr.Dataset:
    """
    Fit normaliser statistics on the full upsampled weather dataset and
    save a normalised copy to disk.

    Parameters
    ----------
    ds_upsampled : xr.Dataset
        Weather dataset on fine grid (output of upsample_weather_to_fine_grid).
    normalizer : VariableNormalizer
        Will be updated with weather variable statistics.
    out_path : Path

    Returns
    -------
    xr.Dataset â€” normalised weather dataset.
    """
    data_vars: Dict[str, xr.Variable] = {}

    for var in WEATHER_VARS:
        arr = ds_upsampled[var].values.astype(np.float32)  # (T, H, W)
        normalizer.fit(arr, name=var)
        norm_arr = normalizer.transform(arr, name=var)
        data_vars[var] = xr.Variable(
            ds_upsampled[var].dims,
            norm_arr,
            attrs={**ds_upsampled[var].attrs, "normalization": normalizer.mode},
        )

    ds_norm = xr.Dataset(
        data_vars,
        coords=ds_upsampled.coords,
        attrs={
            **ds_upsampled.attrs,
            "title": "Topo-Downscale Aligned & Normalised Weather (fine grid)",
            "created_by": "preprocessor.py",
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_norm.to_netcdf(out_path, format="NETCDF4")
    print(f"[âœ“] Aligned weather saved â†’ {out_path}")
    return ds_norm


# ---------------------------------------------------------------------------
# Master pipeline function
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    coarse_path: Path,
    dem_path: Path,
    lulc_path: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE,
    norm_mode: str = "minmax",
    max_patches_per_day: Optional[int] = None,
) -> None:
    """
    Run the full ETL preprocessing pipeline end-to-end.

    1. Load raw datasets.
    2. Upsample coarse weather to fine DEM grid.
    3. Build static tensor (DEM features + LULC).
    4. Normalise all variables.
    5. Extract patches per time step.
    6. Save .npz patches and norm stats.

    Parameters
    ----------
    coarse_path : Path
        Path to data/raw/weather_coarse.nc
    dem_path : Path
        Path to data/processed/dem_features.nc
    lulc_path : Path
        Path to data/raw/lulc.nc
    out_dir : Path
        Root output directory (data/processed/).
    patch_size : int
        Square patch side length in fine-grid cells.
    norm_mode : str
        'minmax' or 'zscore'
    max_patches_per_day : int, optional
        Cap on patches per time step (useful for quick smoke tests).
    """
    print("=" * 60)
    print(" Topo-Downscale â€” Preprocessing Pipeline")
    print("=" * 60)

    # --- Load datasets ---
    print("[*] Loading datasets â€¦")
    ds_coarse = xr.open_dataset(coarse_path)
    ds_dem    = xr.open_dataset(dem_path)
    ds_lulc   = xr.open_dataset(lulc_path)

    fine_lats = ds_dem["lat"].values
    fine_lons = ds_dem["lon"].values

    # --- Step 1: Upsample weather to fine grid ---
    ds_upsampled = upsample_weather_to_fine_grid(
        ds_coarse=ds_coarse,
        fine_lats=fine_lats,
        fine_lons=fine_lons,
        method="linear",
    )

    # --- Step 2: Initialise normaliser ---
    normalizer = VariableNormalizer(mode=norm_mode)

    # --- Step 3: Save aligned weather ---
    aligned_weather_path = out_dir / "aligned_weather.nc"
    ds_norm_weather = save_aligned_weather(ds_upsampled, normalizer, aligned_weather_path)

    # --- Step 4: Build static tensor ---
    print("[*] Building static conditioning tensor â€¦")
    static_tensor = build_static_tensor(
        ds_dem_features=ds_dem,
        ds_lulc=ds_lulc,
        normalizer=normalizer,
        fine_lats=fine_lats,
        fine_lons=fine_lons,
    )

    # --- Step 5: Save normaliser stats ---
    normalizer.save(out_dir / "norm_stats.json")

    # --- Step 6: Extract patches per time step ---
    patch_dir = out_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    n_times = len(ds_norm_weather["time"])
    patch_count = 0

    print(f"[*] Extracting patches ({patch_size}Ã—{patch_size}) "
          f"from {n_times} time steps â€¦")

    for t_idx in range(n_times):
        # Build per-time weather tensor (C_weather, H, W)
        weather_channels = []
        for var in WEATHER_VARS:
            arr = ds_norm_weather[var].values[t_idx]  # (H, W)
            weather_channels.append(arr[np.newaxis])
        coarse_tensor = np.concatenate(weather_channels, axis=0)  # (5, H, W)

        patches = extract_patches(
            coarse_arr=coarse_tensor,
            static_arr=static_tensor,
            patch_size=patch_size,
        )

        if max_patches_per_day is not None:
            patches = patches[:max_patches_per_day]

        for patch in patches:
            fname = patch_dir / f"patch_{patch_count:05d}.npz"
            np.savez_compressed(
                fname,
                coarse=patch["coarse"],   # (5, P, P)
                static=patch["static"],   # (12, P, P) â†’ 5 DEM + 7 LULC
                time_idx=np.array([t_idx], dtype=np.int32),
            )
            patch_count += 1

    print(f"[âœ“] Saved {patch_count} patches â†’ {patch_dir}")
    print("=" * 60)
    print("[âœ“] Preprocessing pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align coarse weather grids with DEM and extract training patches."
    )
    parser.add_argument("--coarse",  default="data/raw/weather_coarse.nc")
    parser.add_argument("--dem",     default="data/processed/dem_features.nc")
    parser.add_argument("--lulc",    default="data/raw/lulc.nc")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    parser.add_argument("--norm_mode", choices=["minmax", "zscore"], default="minmax")
    parser.add_argument("--max_patches_per_day", type=int, default=None,
                        help="Cap patches per day (for quick tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing_pipeline(
        coarse_path=Path(args.coarse),
        dem_path=Path(args.dem),
        lulc_path=Path(args.lulc),
        out_dir=Path(args.out_dir),
        patch_size=args.patch_size,
        norm_mode=args.norm_mode,
        max_patches_per_day=args.max_patches_per_day,
    )


if __name__ == "__main__":
    main()

