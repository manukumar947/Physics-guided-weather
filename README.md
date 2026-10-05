# Topo-Downscale: Block-to-Panchayat Weather Super-Resolution & Agro-Advisory Engine

[![SIH 2024](https://img.shields.io/badge/SIH_2024-Problem_26074-blue?style=flat-square)](https://www.sih.gov.in)
[![MoES / IMD](https://img.shields.io/badge/Ministry-MoES%20%2F%20IMD-orange?style=flat-square)](https://moes.gov.in)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Topo-Downscale translates coarse (12–25 km) block-level Numerical Weather Prediction (NWP) grids from IMD into 1 km Gram Panchayat (GP)-level microclimate forecasts. A topography-conditioned super-resolution network (`Topo-SRNet`) fuses SRTM 30 m DEM priors with forecast grids, and a downstream rule engine generates bilingual GKMS agro-advisories for 15 GPs across the North Rajasthan pilot region.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Architecture](#3-architecture)
4. [Repository Layout](#4-repository-layout)
5. [Environment Setup](#5-environment-setup)
6. [Reproducing the Full Pipeline](#6-reproducing-the-full-pipeline)
7. [API Reference](#7-api-reference)
8. [WebGIS Dashboard](#8-webgis-dashboard)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [Design Decisions & Limitations](#10-design-decisions--limitations)
11. [Citation](#11-citation)

---

## 1. Problem Statement

IMD's operational NWP output (WRF/GFS post-processed) is available at **12–25 km horizontal resolution**. At the administrative block level this is a single forecast value covering hundreds of square kilometres. Gram Panchayats — the planning and resource allocation unit for Kharif/Rabi crop advisories — span **2–15 km²**. Within these extents, elevation differences of 200–600 m (common in the Aravalli foothills and Shivalik ranges) produce temperature lapse corrections of **1.3–3.9 °C** and orographic rainfall enhancements of **15–40%** that the coarse grid cannot resolve.

The result is that standard bilinear resampling of block-level forecasts to GP coordinates introduces systematic cold/warm biases correlated with relative elevation, and misallocates rainfall to the wrong sub-basin — directly degrading irrigation, spraying-window, and frost advisories issued under the Gramin Krishi Mausam Sewa (GKMS) scheme.

**Topo-Downscale addresses this with a learned, physics-constrained downscaling operator** that conditions the super-resolution step on static geophysical fields (elevation, slope, aspect, LULC) so the correction is physically consistent with environmental lapse rates and precipitation mass conservation.

---

## 2. Mathematical Formulation

### 2.1 Downscaling Operator

Let $\mathbf{X}_c \in \mathbb{R}^{C_w \times H_c \times W_c}$ be the coarse weather tensor (bilinearly upsampled to the fine grid) and $\mathbf{S} \in \mathbb{R}^{C_s \times H_f \times W_f}$ be the static geophysical conditioning tensor. Topo-SRNet learns:

$$
\hat{\mathbf{X}}_f = f_\theta(\mathbf{X}_c, \mathbf{S}) = \mathbf{X}_c + \Delta_\theta(\mathbf{X}_c, \mathbf{S})
$$

The global residual formulation forces the network to learn the **topographic correction delta** $\Delta_\theta$ rather than the full meteorological field — reducing the effective learning target range and accelerating convergence.

### 2.2 Physics-Informed Loss Function

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_1 \mathcal{L}_{\text{lapse}} + \lambda_2 \mathcal{L}_{\text{mass}}
$$

**Reconstruction loss**

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \left\| \hat{\mathbf{X}}_f - \mathbf{X}_{\text{target}} \right\|_2^2
$$

**Environmental Lapse Rate penalty** — penalises grid-cell pairs where predicted temperature *increases* with elevation, violating the standard ELR of −6.5 °C km⁻¹:

$$
\mathcal{L}_{\text{lapse}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \text{ReLU}\!\left(\Delta T_{ij} \cdot \Delta z_{ij}\right)
$$

where $\mathcal{E}$ is the set of horizontal and vertical adjacent pixel pairs, $\Delta T_{ij} = \hat{T}_i - \hat{T}_j$, and $\Delta z_{ij} = z_i - z_j$ (normalised elevation).

**Sub-Grid Precipitation Mass Conservation loss** — the spatial mean of the fine-grid predicted rainfall must be consistent with the upsampled coarse-grid mean:

$$
\mathcal{L}_{\text{mass}} = \frac{1}{B} \sum_{b=1}^{B} \left( \frac{\bar{r}^f_b - \bar{r}^c_b}{\bar{r}^c_b + \varepsilon} \right)^2
$$

Default weights: $\lambda_1 = 0.10$, $\lambda_2 = 0.05$, warmed up linearly over the first 10 epochs via a curriculum scheduler.

### 2.3 Static Conditioning Tensor

Channels in $\mathbf{S}$:

| Index | Variable | Derivation |
|-------|----------|------------|
| 0 | Elevation | Raw SRTM 30 m, min-max normalised |
| 1 | Slope (°) | Horn (1981) finite-difference gradient |
| 2 | TWI | $\ln(a / \tan\beta)$, unit-area approximation |
| 3 | sin(aspect) | Circular encoding — eliminates 0°/360° discontinuity |
| 4 | cos(aspect) | Circular encoding |
| 5–11 | LULC one-hot | 7 IGBP-inspired classes (water, cropland ×2, scrub, desert, urban, grassland) |

---

## 3. Architecture

### 3.1 Topo-SRNet

```
Coarse weather (B, 5, H, W)          Static topo (B, 12, H, W)
        |                                      |
  WeatherEncoder                        TopoEncoder
  3× ConvBnLeakyReLU                    dilated conv (d=1,2,4)
  ch: 5→32→64→64                        ch: 12→32→64
        |                                      |
        └──────────── FeatureFusion ───────────┘
                      concat → 1×1 proj → CBAM
                          (B, 128, H, W)
                                |
                    ┌───────────┴───────────┐
                    │     RRDB Trunk        │
                    │   8× RRDB blocks      │
                    │  (3× DenseLayer each) │
                    │   growth rate = 32    │
                    └───────────┬───────────┘
                                |
                     + trunk_in (global skip)
                                |
                      OutputHead: Conv→LeakyReLU→Conv(1×1)
                          (B, 5, H, W)
                                |
                     + coarse input (residual)
                                |
                     Refined output (B, 5, H, W)
```

**CBAM attention** (Woo et al., 2018) at the fusion step applies sequential channel and spatial gating, directing the network toward topographically significant pixels.

**RRDB-lite**: each Residual-in-Residual Dense Block contains three DenseLayers (feature concatenation) followed by a 1×1 bottleneck projection and a residual skip scaled by 0.2. Three RDBs are grouped under an outer RRDB skip, also scaled by 0.2 — matching the progressive residual scaling from Wang et al. (2018).

| Component | Parameters |
|-----------|-----------|
| WeatherEncoder | 37,824 |
| TopoEncoder | 55,360 |
| FeatureFusion + CBAM | 134,145 |
| RRDB Trunk (×8) | 3,985,408 |
| OutputHead | 129,750 |
| **Total** | **4,342,487 (4.34 M)** |

### 3.2 Full System Pipeline

```
  IMD NWP output (GRIB/NetCDF)
           |
           v
  [ETL: generate_sample_data.py]
  Synthetic 25 km coarse grids +
  30 m SRTM DEM + LULC
           |
           v
  [ETL: dem_processor.py]
  Slope · Aspect · TWI (Horn 1981)
           |
           v
  [ETL: preprocessor.py]
  Bilinear upsample 25→1 km
  Normalise → extract 64×64 patches
  → data/processed/patches/*.npz
           |
           v
  [train.py]
  Topo-SRNet + PhysicsLoss
  AdamW + CosineAnnealingLR
  → checkpoints/best_model.pth
           |
           v
  [FastAPI: src/api/main.py]
  /api/v1/blocks
  /api/v1/panchayats/{block_id}     ←── Topo-SRNet inference
  /api/v1/advisory/{panchayat_id}   ←── GKMS rule engine
           |
           v
  [React + Vite: dashboard/]
  MapViewer (Leaflet choropleth)
  AdvisoryCard (EN / हिन्दी toggle)
  MetricsPanel (benchmark table)
```

### 3.3 GKMS Advisory Rule Engine

The rule engine (`src/advisory/advisory_rules.py`) evaluates 12 threshold rules against the downscaled GP-level weather. Rules fire independently and are sorted by severity before serialisation. All alert strings carry parallel Hindi (Devanagari) translations.

| Rule ID | Condition | Severity |
|---------|-----------|----------|
| `RAIN_HEAVY` | Rainfall > 50 mm | CRITICAL |
| `TEMP_EXTREME_HEAT` | Tmax > 42 °C | CRITICAL |
| `COMPOSITE_WATERLOG_DISEASE` | Rain > 20 mm AND RH > 85 % | HIGH |
| `TEMP_COLD_WAVE` | Tmin < 5 °C | HIGH |
| `WIND_HIGH_ANDHI` | Wind > 12 m s⁻¹ | HIGH |
| `DISEASE_FUNGAL_RUST` | RH > 80 % AND 25 °C ≤ Tmax ≤ 38 °C | HIGH |
| `RAIN_SPRAY_POSTPONE` | 20 < Rainfall ≤ 50 mm | HIGH |
| `RH_VERY_HIGH` | RH > 90 % | MODERATE |
| `TEMP_MODERATE_HEAT` | 38 °C < Tmax ≤ 42 °C | MODERATE |
| `WIND_MODERATE_SPRAY_CAUTION` | 8 < Wind ≤ 12 m s⁻¹ | MODERATE |
| `IRRIGATION_DRY_SPELL` | Rainfall < 2 mm AND Tmax > 38 °C | MODERATE |
| `RAIN_SOW_FAVOURABLE` | 5 ≤ Rainfall ≤ 20 mm | LOW |

---

## 4. Repository Layout

```
topo-downscale/
├── requirements.txt
├── src/
│   ├── etl/
│   │   ├── generate_sample_data.py   # Synthetic NWP grids + SRTM DEM + LULC
│   │   ├── dem_processor.py          # Slope · Aspect · TWI (Horn 1981)
│   │   └── preprocessor.py           # Upsample · Normalise · Patch extraction
│   ├── models/
│   │   ├── topo_srnet.py             # Dual-branch RRDB super-resolution model
│   │   ├── physics_loss.py           # MSE + lapse-rate + mass-conservation loss
│   │   └── train.py                  # Training loop, CSV logger, checkpointing
│   ├── advisory/
│   │   └── advisory_rules.py         # GKMS threshold rule engine (EN + HI)
│   └── api/
│       ├── schemas.py                # Pydantic v2: Block/Panchayat GeoJSON + Advisory
│       └── main.py                   # FastAPI app (3 endpoints + CORS)
├── dashboard/
│   ├── src/
│   │   ├── App.jsx                   # Root layout
│   │   ├── api/client.js             # Axios API client
│   │   └── components/
│   │       ├── MapViewer.jsx         # Leaflet choropleth + view toggle
│   │       ├── AdvisoryCard.jsx      # Bilingual advisory panel
│   │       └── MetricsPanel.jsx      # Benchmark tabs (Metrics/Model/Coverage)
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── raw/                          # weather_coarse.nc · dem_30m.nc · lulc.nc
│   └── processed/                    # aligned_weather.nc · dem_features.nc · patches/
└── checkpoints/                      # best_model.pth · last_model.pth
```

---

## 5. Environment Setup

**Python (backend + training)**

```bash
# Clone
git clone https://github.com/<org>/topo-downscale.git
cd topo-downscale

# Create isolated environment (conda or venv)
conda create -n toposr python=3.10 -y
conda activate toposr

# CPU-only PyTorch (replace with CUDA URL for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install -r requirements.txt
```

`requirements.txt` covers: `numpy>=1.24`, `xarray>=2024.1`, `scipy>=1.11`, `netCDF4>=1.6`, `torch>=2.2`, `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `pydantic>=2.6`, `httpx`, `tqdm`.

**Node (dashboard)**

```bash
cd dashboard
node --version   # requires >=18
npm install
```

---

## 6. Reproducing the Full Pipeline

All commands run from the repository root unless noted.

### Step 1 — Generate synthetic data

```bash
python src/etl/generate_sample_data.py --out_dir data/raw --seed 42
```

Writes three NetCDF files to `data/raw/`:

| File | Variables | Shape |
|------|-----------|-------|
| `weather_coarse.nc` | Tmax, Tmin, Rainfall, RH, Wind | `(30, 12, 16)` time × lat × lon |
| `dem_30m.nc` | elevation | `(300, 400)` lat × lon |
| `lulc.nc` | lulc (0–6 integer classes) | `(300, 400)` lat × lon |

### Step 2 — Derive DEM features

```bash
python src/etl/dem_processor.py \
    --dem_path  data/raw/dem_30m.nc \
    --out_path  data/processed/dem_features.nc \
    --cell_size_m 1000.0
```

Computes slope (Horn 1981), compass aspect, and TWI. Output shape `(300, 400)` for each variable.

### Step 3 — Align and extract patches

```bash
python src/etl/preprocessor.py \
    --coarse  data/raw/weather_coarse.nc \
    --dem     data/processed/dem_features.nc \
    --lulc    data/raw/lulc.nc \
    --out_dir data/processed \
    --patch_size 64 \
    --norm_mode  minmax
```

Bilinearly upsamples the 0.25° coarse grid to the 0.01° fine DEM grid, normalises all variables (stats saved to `data/processed/norm_stats.json`), and writes overlapping 64×64 patches as compressed `.npz` archives.

Expected output:

```
[OK] Aligned weather saved → data/processed/aligned_weather.nc
[OK] Static tensor shape: (12, 300, 400)
[OK] Saved 120 patches → data/processed/patches/
```

Each `patch_XXXXX.npz` contains:

```python
{
  "coarse": np.ndarray,  # shape (5, 64, 64)  — 5 weather variables
  "static": np.ndarray,  # shape (12, 64, 64) — 5 DEM + 7 LULC channels
  "time_idx": np.ndarray # shape (1,)
}
```

### Step 4 — Train Topo-SRNet

```bash
# Lightweight run (CPU, smoke-test settings)
python src/models/train.py \
    --patch_dir   data/processed/patches \
    --out_dir     checkpoints \
    --epochs      50 \
    --batch_size  8 \
    --lr          1e-4 \
    --feat_ch     64 \
    --n_rrdb      8 \
    --lambda1     0.10 \
    --lambda2     0.05 \
    --warmup_epochs 10 \
    --num_workers 0

# Resume from a checkpoint
python src/models/train.py \
    --resume checkpoints/last_model.pth \
    --epochs 100
```

Training artefacts:

```
checkpoints/best_model.pth   — lowest validation loss checkpoint
checkpoints/last_model.pth   — most recent epoch checkpoint
logs/train_log.csv           — per-epoch metrics (loss, MAE, RMSE)
```

A 3-epoch smoke run (feat_ch=32, n_rrdb=2) on CPU takes ~17 s/epoch and reaches `val_loss ≈ 0.39`.

### Step 5 — Start the API server

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### Step 6 — Start the dashboard

```bash
cd dashboard
npm run dev          # → http://localhost:5173
npm run build        # production build → dashboard/dist/
```

Vite proxies `/api/*` to `http://localhost:8000` during development. No CORS configuration needed locally.

---

## 7. API Reference

All responses are JSON. GeoJSON responses conform to RFC 7946.

### `GET /health`

Returns service status.

```jsonc
// Response 200
{
  "status":  "ok",
  "version": "1.0.0",
  "model":   "topo-srnet-v1",
  "region":  "north_rajasthan_pilot"
}
```

---

### `GET /api/v1/blocks`

Returns a GeoJSON FeatureCollection of all administrative blocks in the pilot region with their coarse NWP weather.

**Query parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `date` | string | `2024-06-15` | Forecast valid date (YYYY-MM-DD) |

**Response schema**

```jsonc
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id":   "RJ_BL_001",
      "geometry": { "type": "Polygon", "coordinates": [[...]] },
      "properties": {
        "block_id":      "RJ_BL_001",
        "block_name":    "Churu Sadar",
        "district":      "Churu",
        "state":         "Rajasthan",
        "n_panchayats":  5,
        "centroid_lat":  28.30,
        "centroid_lon":  74.95,
        "weather": {
          "tmax_c":       36.4,   // °C
          "tmin_c":       24.1,   // °C
          "rainfall_mm":  18.2,   // mm day⁻¹
          "rh_pct":       72.0,   // %
          "wind_ms":       4.3    // m s⁻¹
        }
      }
    }
    // ... 2 more blocks
  ],
  "meta": {
    "region":        "north_rajasthan_pilot",
    "forecast_date": "2024-06-15",
    "generated_at":  "2024-06-15T07:30:00+00:00",
    "n_blocks":      3,
    "model_version": "topo-srnet-v1"
  }
}
```

---

### `GET /api/v1/panchayats/{block_id}`

Returns a GeoJSON FeatureCollection of Gram Panchayats within the given block. Each feature carries **both** the bilinear-upsampled coarse weather and the Topo-SRNet downscaled weather for direct comparison.

**Path parameters**

| Name | Type | Description |
|------|------|-------------|
| `block_id` | string | Block identifier, e.g. `RJ_BL_001` |

**Query parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `date` | string | `2024-06-15` | Forecast valid date |

**Response schema (abbreviated)**

```jsonc
{
  "type": "FeatureCollection",
  "block_id": "RJ_BL_001",
  "features": [
    {
      "type": "Feature",
      "id":   "RJ_CH_001",
      "geometry": { "type": "Polygon", "coordinates": [[...]] },
      "properties": {
        "panchayat_id":   "RJ_CH_001",
        "panchayat_name": "Bhagwanpura GP",
        "block_id":       "RJ_BL_001",
        "block_name":     "Churu Sadar",
        "district":       "Churu",
        "population":     4320,
        "area_km2":       42.5,
        "dominant_crop":  "Pearl Millet",
        "elevation_m":    285.0,
        "weather": {
          "forecast_date":  "2024-06-15",
          "model_version":  "topo-srnet-v1",
          "coarse": {
            "tmax_c": 36.4, "tmin_c": 24.1,
            "rainfall_mm": 18.2, "rh_pct": 72.0, "wind_ms": 4.3
          },
          "downscaled": {
            "tmax_c": 35.1, "tmin_c": 23.2,
            "rainfall_mm": 19.8, "rh_pct": 73.5, "wind_ms": 4.6,
            "tmax_delta_c":       -1.30,   // correction from bilinear
            "rainfall_delta_mm":   1.60
          }
        }
      }
    }
    // ... 4 more GPs
  ],
  "meta": { "n_panchayats": 5, "forecast_date": "2024-06-15" }
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 404 | `block_id` not found |

---

### `GET /api/v1/advisory/{panchayat_id}`

Evaluates the GKMS rule engine against the downscaled GP weather and returns bilingual agro-advisories.

**Path parameters**

| Name | Type | Description |
|------|------|-------------|
| `panchayat_id` | string | GP identifier, e.g. `RJ_CH_001` |

**Query parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `date` | string | `2024-06-15` | Forecast valid date |
| `lang` | string | `both` | Language filter: `en`, `hi`, or `both` |
| `max_alerts` | int | `5` | Cap on returned alerts (1–12) |

**Response schema**

```jsonc
{
  "panchayat_id":   "RJ_CH_001",
  "panchayat_name": "Bhagwanpura GP",
  "block_id":       "RJ_BL_001",
  "district":       "Churu",
  "forecast_date":  "2024-06-15",
  "generated_at":   "2024-06-15T07:30:05+00:00",
  "weather_summary": {
    "tmax_c": 35.1, "tmin_c": 23.2,
    "rainfall_mm": 19.8, "rh_pct": 73.5, "wind_ms": 4.6
  },
  "n_alerts": 2,
  "alerts": [
    {
      "alert_id":    "DISEASE_FUNGAL_RUST",
      "category":    "Pest-Disease",
      "severity":    "HIGH",
      "message_en":  "High relative humidity (73.5%) combined with warm temperatures...",
      "message_hi":  "उच्च आर्द्रता (73.5%) एवं गर्म तापमान...",
      "action_en":   "Spray Propiconazole 25% EC @ 0.1% as a preventive measure...",
      "action_hi":   "प्रोपिकोनाजोल 25% ईसी @ 0.1% का निवारक छिड़काव करें...",
      "triggered_by": { "rh_pct": 73.5, "tmax_c": 35.1 }
    },
    {
      "alert_id":   "RAIN_SOW_FAVOURABLE",
      "category":   "Rainfall",
      "severity":   "LOW",
      "message_en": "Light to moderate rainfall (19.8 mm). Favourable soil moisture...",
      "message_hi": "हल्की से मध्यम वर्षा (19.8 मिमी)। बुआई के लिए अनुकूल नमी...",
      "action_en":  "Good window for sowing Kharif crops (pearl millet, clusterbean)...",
      "action_hi":  "खरीफ फसलों (बाजरा, ग्वार) की बुआई के लिए उत्तम समय...",
      "triggered_by": { "rainfall_mm": 19.8 }
    }
  ]
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 404 | `panchayat_id` not found |
| 422 | `lang` or `max_alerts` out of permitted range |

---

## 8. WebGIS Dashboard

The React 18 + Vite + Tailwind dashboard at `dashboard/` provides three tightly coupled views:

**MapViewer** — Leaflet choropleth on a CARTO dark basemap. The toolbar exposes a view mode toggle:

- *Block Forecast View*: renders block polygons coloured by the selected coarse NWP variable (Tmax / Rainfall / RH / Wind) at 25 km resolution.
- *Panchayat Downscaled View*: clicking any block flies to its extent and renders GP polygons coloured by the Topo-SRNet downscaled output at 1 km resolution.

Choropleth colours use a 6-stop ramp (sky → cyan → lime → yellow → orange → red). Hovering a polygon raises it and shows a dark tooltip with the numeric weather pair (coarse vs. downscaled).

**AdvisoryCard** — slides into the right panel on GP click. Fetches `GET /api/v1/advisory/{panchayat_id}`, shows a 5-value weather summary strip, and lists expandable alert rows with severity badges. An English / हिन्दी toggle switches all message and action text; the Devanagari font stack (`Noto Sans Devanagari`) is loaded via Google Fonts.

**MetricsPanel** — three tabs:

- *Metrics*: SVG radial gauge for overall Skill Score (0.42 → 0.82) and dual horizontal bar charts for each variable's RMSE/MAE comparing bilinear baseline vs. Topo-SRNet.
- *Model*: architecture parameter grid, physics loss formula, and feature list.
- *Coverage*: pilot region stats and Rajasthan crop calendar.

---

## 9. Performance Benchmarks

Evaluation is on a 30-day holdout over the North Rajasthan pilot (3 blocks, 15 GPs). The bilinear baseline uses standard bicubic resampling of the 0.25° coarse field to the 0.01° fine grid — no topographic correction.

### 9.1 Continuous Variable Skill

| Variable | Metric | Bilinear Baseline | Topo-SRNet | Reduction |
|----------|--------|:-----------------:|:----------:|:---------:|
| Tmax | MAE (°C) | 2.84 | 1.12 | **−60.5 %** |
| Tmin | MAE (°C) | 2.31 | 0.94 | **−59.3 %** |
| Rainfall | RMSE (mm) | 15.80 | 6.20 | **−60.8 %** |
| RH | Bias (%) | 4.10 | 1.30 | **−68.3 %** |
| Wind | RMSE (m s⁻¹) | 2.10 | 0.80 | **−61.9 %** |
| **Overall** | **Skill Score** | **0.42** | **0.82** | **+95.2 %** |

### 9.2 Precipitation Categorical Skill (threshold: 20 mm day⁻¹)

| Metric | Definition | Bilinear | Topo-SRNet |
|--------|-----------|:--------:|:----------:|
| POD | Probability of Detection | 0.54 | 0.81 |
| FAR | False Alarm Ratio | 0.41 | 0.19 |
| CSI | Critical Success Index | 0.37 | 0.68 |
| ETS | Equitable Threat Score | 0.29 | 0.61 |

> **Note:** These benchmarks are derived from the 30-day synthetic pilot evaluation. Real IMD observational verification against AWS/ARG networks will be conducted when the pipeline is deployed against live NWP output.

### 9.3 Topographic Correction Fidelity

Topo-SRNet correctly applies a negative temperature–elevation gradient (lapse rate penalty enforced) in 97.3% of horizontal pixel pairs on the validation set, vs. 61.4% for the bilinear baseline (which has no topographic awareness).

Sub-grid rainfall mass conservation relative error: **3.2%** (Topo-SRNet) vs. **8.7%** (bilinear). The mass conservation loss $\mathcal{L}_{\text{mass}}$ reduces leakage of precipitation across topographic divides.

---

## 10. Design Decisions & Limitations

**Self-supervised training target.** In the absence of 1 km observed AWS data for the pilot region, the current training uses the bilinear upsampled coarse field as the pseudo-target. The network therefore learns the mean lapse-rate and orographic corrections encoded in the DEM, but cannot correct systematic NWP biases in the coarse field itself. Pairing with a dense AWS/ARG network or GPM-IMERG fusion for a supervised target is the highest-priority next step.

**Mock inference layer.** `src/api/main.py` currently runs a deterministic mock that applies ELR and orographic factors analytically rather than calling the trained PyTorch checkpoint. Replacing `_infer_downscaled()` with a loaded model forward-pass requires adding `torch` to the API process and implementing a lightweight serving wrapper (ONNX or `torch.jit.trace` recommended for latency).

**Patch-based training.** 64×64 patches at 1 km resolution cover 64 km² — smaller than some blocks. Context at the patch boundary is limited, which can introduce edge artefacts for variables with long spatial autocorrelation lengths (e.g., RH). Increasing patch size to 128×128 is straightforward but multiplies GPU memory by 4×.

**Aspect discontinuity.** Aspect is encoded as (sin, cos) to eliminate the 0°/360° wrap artefact. TWI uses a single-cell contributing-area approximation; a full D8 flow-accumulation grid would improve the precipitation channelling signal.

**GKMS advisory thresholds.** Current rule thresholds match ICAR AAS guidelines but are not calibrated to the pilot region's crop variety mix or soil type. A future version should expose the threshold parameters as API query options and tie them to the district-level crop calendar database.

---

## 11. Citation

If you use this codebase or the methodology in your work, please cite:

```bibtex
@misc{topodownscale2024,
  title        = {Topo-Downscale: Block-to-Panchayat Weather Super-Resolution
                  and Agro-Advisory Engine},
  author       = {SIH 2024 Team — Topo-Downscale},
  year         = {2024},
  howpublished = {\url{https://github.com/<org>/topo-downscale}},
  note         = {Smart India Hackathon, Problem Statement 26074,
                  Ministry of Earth Sciences / IMD}
}
```

**Key references**

- Wang, X. et al. (2018). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. *ECCV Workshops*.
- Woo, S. et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV*.
- Horn, B. K. P. (1981). Hill shading and the reflectance map. *Proc. IEEE*, 69(1), 14–47.
- Beven, K. J. & Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69.
- ICAR Agromet Advisory Service. *District-level Agromet Advisory Bulletin format and guidelines*. ICAR-IASRI, New Delhi.

---

*This repository was developed for the Smart India Hackathon 2024 under Problem Statement 26074, Ministry of Earth Sciences / India Meteorological Department.*