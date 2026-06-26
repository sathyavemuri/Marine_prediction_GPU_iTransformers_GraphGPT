# Marine Local MTGNN

**Local-only MTGNN residual forecaster for marine buoy prediction**

## Overview

This project implements a **residual MTGNN** (Multi-Task Graph Neural Network) system for forecasting marine parameters using **only buoy observation history** and **deterministic local formulas**. It requires:

- **No external APIs** (no Open-Meteo, Copernicus, weather providers)
- **No network calls** for data (all calculations are local)
- **No API keys or credentials**

### Approach: Residual Learning

```
final_forecast = internal_baseline + MTGNN_predicted_residual
```

Each baseline is computed using only data available at forecast origin:
- **Persistence** / seasonal persistence
- **Harmonic tide** (UTide, astronomical only)
- **Clear-sky radiation** (pvlib)
- **Local trend** (Theil-Sen regression)

### Primary Profile: 15-Minute 7-Day Forecast

- **Input**: 7 days of 15-minute observations (672 steps)
- **Output**: Direct 7-day forecast (672 steps)
- **Not recursive**: All future steps predicted in one forward pass
- **Data split**: 90 days train / 20 days validation / 10 days test

## Installation

### 1. Python 3.11+

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
python -m pip install --upgrade pip setuptools wheel
```

### 2. PyTorch (hardware-specific)

Install the wheel appropriate to your system. See [pytorch.org](https://pytorch.org/).

**CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA CUDA:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
```

### 3. Project Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Edit `configs/local_15min_7day.yaml` and set:

- **Site metadata** (latitude, longitude, timezone)
- **Direction conventions** (wind/current from/to)
- **Raw CSV path**
- **Time zone** (must match sensor timestamps)

## Data Requirements

Expected raw CSV columns (19 total):
- timestamp
- 18 marine parameters (air/water temperature, wind, waves, tides, salinity, etc.)

See `constants.RAW_CSV_COLUMNS` for exact order.

## CLI Commands

```bash
# Validate raw data
python -m marine_local_mtgnn.cli validate-data --config configs/local_15min_7day.yaml

# Preprocess: transform, resample to 15min, split
python -m marine_local_mtgnn.cli preprocess --config configs/local_15min_7day.yaml

# Fit baseline candidates
python -m marine_local_mtgnn.cli fit-baselines --config configs/local_15min_7day.yaml

# Build graph prior from training correlations
python -m marine_local_mtgnn.cli build-graph-prior --config configs/local_15min_7day.yaml

# Train MTGNN
python -m marine_local_mtgnn.cli train --config configs/local_15min_7day.yaml

# Evaluate on validation/test
python -m marine_local_mtgnn.cli evaluate --config configs/local_15min_7day.yaml --split validation
python -m marine_local_mtgnn.cli evaluate --config configs/local_15min_7day.yaml --split test

# Generate forecast
python -m marine_local_mtgnn.cli forecast --config configs/local_15min_7day.yaml

# Launch API service
python -m marine_local_mtgnn.cli serve --config configs/local_15min_7day.yaml
```

## Architecture

### Data Pipeline
1. Load raw CSV (1-minute cadence)
2. Transform directions (deg → u/v components)
3. Resample to 15-minute grid
4. Create 90/20/10 chronological splits

### Baselines
Per-target local baselines (validation-based selection):
- **Persistence** — Repeat last observation
- **Seasonal Persistence** — 7-day cycle (same time-of-day from week prior)
- **Daily Seasonal** — Average by time-of-day (96 slots/day)
- **Weekly Seasonal** — Average by day-of-week + time-of-day (672 slots/week)
- **Local Trend** — Theil-Sen regression on recent window (default 3 hours)

Selection metric: validation MAE on rolling origins (rolling-origin evaluation on validation split)

### MTGNN Model
- **Input nodes**: 19 (all parameters + derived directions)
- **Target nodes**: 15 (direct forecast targets)
- **Architecture**:
  - Temporal dilated convolutions (9 blocks, receptive field > 672)
  - Graph attention layers (learned + sparse prior)
  - Horizon-conditioned decoder (direct 672-step output)

### Output
15 direct targets + 6 derived quantities:
- Direct: air temperature, pressure, wind/current u/v, water temp, tide, dew point, radiation, salinity, wave parameters
- Derived: relative humidity, wind/current speed/direction, conductivity (if validated)

## Key Limitations (READ CAREFULLY)

This system **cannot** predict:
- **Remote swell** — only local wind-driven waves
- **Cloud cover changes** — clear-sky formula is deterministic
- **Storm surge** — only astronomical tide
- **Large-scale ocean circulation** — only local history and tides
- **River discharge effects** — only sensor history available

All forecasts are local-only. External forcing (weather systems, distant swell, monsoon changes) may degrade skill at longer lead times.

## Testing

```bash
pytest tests/ -v
```

## Output

Evaluation produces:
- `metrics_by_target.csv` — MAE, RMSE, skill per parameter
- `metrics_by_horizon.csv` — skill decay by forecast lead time
- `predictions_test.parquet` — forecast vs. actual
- `learned_adjacency.csv` — learned graph structure
- `run_manifest.json` — config, seed, feature list, hyperparameters
- `figures/` — diagnostic plots

## References

- **MTGNN**: Connecting the Dots: Identifying Network Structure via Graph Signal Processing
- **UTide**: Harmonic tidal constituent estimation from irregular hourly sea level data
- **pvlib**: A Python Library for Solar Photovoltaic Modeling and Analysis
- **GSW/TEOS-10**: Gibbs SeaWater Oceanographic Toolbox
- **Specification**: See `MTGNN_local_only_7day_long_horizon_implementation_for_Claude_Code.txt` for full design document

## License

MIT

## Status

**Phase A: Complete** ✓ Project foundation, CLI skeleton, config system
**Phase B: Complete** ✓ Data validation, transforms, resampling, splitting (27 tests)
**Phase C: Complete** ✓ Baseline candidates, validation-based selection (11 tests)
**Phase D: Complete** ✓ Residual dataset, graph prior, scalers (7 tests, 45 total)
**Phase E: In Progress** — MTGNN model implementation and training
**Phase F: Planned** — Evaluation and results
**Phase G: Planned** — API service
