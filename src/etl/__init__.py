"""
src/etl — ETL package for Topo-Downscale.

Modules
-------
generate_sample_data : Synthetic NetCDF data generation (weather + DEM + LULC)
dem_processor        : Slope, aspect, and TWI computation from DEM
preprocessor         : Spatial alignment, normalisation, and patch extraction
"""

from .dem_processor import compute_slope, compute_aspect, compute_twi, process_dem
from .preprocessor import (
    upsample_weather_to_fine_grid,
    VariableNormalizer,
    run_preprocessing_pipeline,
)

__all__ = [
    "compute_slope",
    "compute_aspect",
    "compute_twi",
    "process_dem",
    "upsample_weather_to_fine_grid",
    "VariableNormalizer",
    "run_preprocessing_pipeline",
]
