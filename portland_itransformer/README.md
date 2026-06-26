# Portland iTransformer: Marine Forecasting with Inverted Transformers

**A local-only, deterministic iTransformer system for 7-day marine buoy forecasting at Portland Harbor, Maine.**

## Project Status: Phase 1 - Scaffolding Complete ✓

This project implements the specification provided in `SPECIFICATION.md` for marine forecasting using an iTransformer architecture.

## Key Features

- ✅ **iTransformer Architecture**: Inverted transformer treating each variable/channel as a token
- ✅ **14-day Input / 7-day Output**: 1,344 steps in (15-min cadence) → 672 steps out
- ✅ **Deterministic Baselines**: 
  - UTide harmonic tide model (fit on first 60 days only)
  - pvlib clear-sky radiation
  - Daily seasonal patterns
- ✅ **Local-Only**: No external APIs; all calculations from site coordinates + local data
- ✅ **Proper Data Leakage Prevention**:
  - Scalers fit on training data only
  - Derived calibrators (conductivity, wave periods) fit on training only
  - UTide fitted on first 60 days before any model training
- ✅ **Full Reconstruction**: All 18 original columns reconstructed after inference
- ⏳ **Training Pipeline**: In progress

## Data

**File**: `data/raw/portland_harbor_2025_15min_synthetic_calibrated.csv`

**Dataset**: SYNTHETIC data for Portland Harbor, 2025, 35,040 timestamps at 15-minute cadence.

⚠️ **Important**: This synthetic dataset is for pipeline testing only. Strong scores on this data do not establish real-world forecast skill.

## Project Structure

```
portland_itransformer/
├── README.md
├── requirements.txt
├── pyproject.toml
├── SPECIFICATION.md           # Full implementation spec
├── configs/
│   └── portland_7day.yaml     # Configuration (14-day history, 7-day forecast)
├── data/
│   ├── raw/                   # Raw CSV (35,040 rows)
│   └── processed/             # Preprocessed Parquet (after preprocessing)
├── artifacts/                 # Fitted scalers, calibrators, tide model
├── outputs/                   # Training/evaluation results
├── src/portland_itransformer/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Pydantic config classes
│   ├── constants.py           # Feature definitions
│   ├── validate.py            # Data validation & QA
│   ├── features.py            # Transformations (direction→u/v, log, etc.)
│   ├── baselines.py           # UTide, pvlib, seasonal
│   ├── scaling.py             # StandardScaler pipeline
│   ├── calibrators.py         # Derived output calibrators
│   ├── dataset.py             # PyTorch Dataset
│   ├── metrics.py             # Evaluation metrics
│   ├── reconstruct.py         # Output reconstruction
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Test evaluation
│   ├── predict.py             # Inference
│   └── models/
│       ├── __init__.py
│       └── marine_itransformer.py  # MarineITransformer architecture
└── tests/
    ├── test_schema.py
    ├── test_transforms.py
    ├── test_no_leakage.py
    ├── test_windows.py
    ├── test_model_shapes.py
    └── test_reconstruction.py
```

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Quickstart

### 1. Validate Data

```bash
python -m portland_itransformer.cli validate-data \
  --config configs/portland_7day.yaml
```

Output:
- `outputs/<run_id>/data_quality_report.json`
- `outputs/<run_id>/data_quality_report.csv`

### 2. Preprocess

```bash
python -m portland_itransformer.cli preprocess \
  --config configs/portland_7day.yaml
```

Transforms:
- Directions → u/v components
- UTide fit on first 60 days
- pvlib clear-sky radiation
- Log transforms for positive variables
- Scaling (StandardScaler on training only)

Outputs:
- `data/processed/portland_preprocessed.parquet`
- `data/processed/feature_manifest.json`
- `artifacts/target_scaler.joblib`
- `artifacts/known_scaler.joblib`
- `artifacts/utide_coefficients.pkl`
- `artifacts/derived_calibrators.joblib`

### 3. Train

```bash
python -m portland_itransformer.cli train \
  --config configs/portland_7day.yaml \
  --device cuda  # or 'cpu'
```

Training:
- 186 days training (Mar 1 - Sep 2)
- 60 days validation (Sep 3 - Nov 1)
- Early stopping on validation loss
- Best model checkpoint saved

Outputs:
- `outputs/<run_id>/best_model.pt`
- `outputs/<run_id>/training_history.csv`
- `outputs/<run_id>/validation_metrics.json`

### 4. Evaluate

```bash
python -m portland_itransformer.cli evaluate \
  --config configs/portland_7day.yaml \
  --split test
```

Evaluation:
- Rolling 7-day forecasts over 60-day test period
- Metrics by feature and horizon bucket (0-6h, 6-24h, 24-72h, 72-168h)
- Comparison vs baselines (persistence, seasonal, tide, radiation)
- Circular MAE for directions

Outputs:
- `outputs/<run_id>/test_metrics_by_feature.csv`
- `outputs/<run_id>/test_metrics_by_horizon.csv`
- `outputs/<run_id>/baseline_comparison.csv`
- `outputs/<run_id>/test_predictions_long.csv`
- `outputs/<run_id>/plots/*.png`

### 5. Predict

```bash
python -m portland_itransformer.cli predict \
  --config configs/portland_7day.yaml \
  --origin '2025-12-24T00:00:00Z'
```

Outputs:
- Forecast for all 18 original columns
- 7-day horizon (672 time steps)
- Timestamps, predictions, and reconstruction details

## Implementation Status

### ✅ Complete (Phase 1)

- [x] Project scaffold and structure
- [x] Configuration system (Pydantic)
- [x] Constants and feature definitions
- [x] Data validation module
- [x] Feature transformations (u/v, log, cyclical)
- [x] Deterministic baselines (UTide, pvlib, seasonal)
- [x] MarineITransformer model architecture
- [x] Requirements and dependency management

### ⏳ In Progress / Needed (Phase 2)

- [ ] Scaling and preprocessing pipeline
- [ ] Derived output calibrators (conductivity, wave periods)
- [ ] PyTorch Dataset implementation
- [ ] Training loop with early stopping
- [ ] Evaluation metrics and baselines
- [ ] Output reconstruction and final reporting
- [ ] CLI command implementations
- [ ] Comprehensive test suite
- [ ] Documentation and examples

## Configuration

Edit `configs/portland_7day.yaml` to customize:

```yaml
# Data splits
train_target_start: '2025-03-01 00:00:00+00:00'
train_target_end: '2025-09-02 23:45:00+00:00'
val_target_start: '2025-09-03 00:00:00+00:00'
val_target_end: '2025-11-01 23:45:00+00:00'
test_target_start: '2025-11-02 00:00:00+00:00'
test_target_end: '2025-12-31 23:45:00+00:00'

# Model hyperparameters
model:
  d_model: 128        # Hidden dimension
  n_heads: 4          # Attention heads
  e_layers: 3         # Encoder layers
  d_ff: 256           # Feed-forward dim
  dropout: 0.20

# Training
training:
  batch_size: 16
  epochs: 40
  learning_rate: 0.0003
  early_stopping_patience: 8
```

## Architecture Overview

### MarineITransformer

**Inverted Transformer Principle**: Each variable is a token containing its full historical sequence.

```
Input: [batch, 1344 timesteps, 19 features]
                ↓
[Per-feature embeddings: 1344 timesteps → 128 dims]
                ↓
[Transformer encoder: attention over 19 features]
                ↓
[Select target tokens: 13 features]
                ↓
[Forecast head: 128 dims → 672 timesteps]
                ↓
[Future covariate MLP: add conditioning from tide/solar/calendar]
                ↓
Output: [batch, 672 timesteps, 13 targets]
```

### Data Pipeline

1. **Validation**: Check schema, cadence, ranges
2. **Transform**: Directions→u/v, log transforms, cyclical features
3. **Baselines**: UTide (60-day fit), pvlib clear-sky
4. **Split**: Train (186 days), Val (60 days), Test (60 days)
5. **Scale**: StandardScaler on training only
6. **Windows**: Sliding windows with stride (1h train, 1d eval)
7. **Train**: Adam, Huber loss, ReduceLROnPlateau scheduler
8. **Evaluate**: Metrics by feature, horizon, baselines
9. **Reconstruct**: All 18 original columns from 13 model targets

## Features and Targets

### 13 Model Targets (Directly Forecasted)

```
air_temp_c              (direct)
air_pressure_hpa        (direct)
water_temp_c            (direct)
dew_point_c             (direct)
salinity_psu            (direct)
wind_u_east_ms          (u/v component)
wind_v_north_ms         (u/v component)
current_u_east_ms       (u/v component)
current_v_north_ms      (u/v component)
tidal_residual_m        (observed tide - UTide baseline)
log_significant_wave_height_m
log_zero_crossing_period_s
log_clearness_index     (log(radiation / clear-sky))
```

### 12 Derived Outputs (Reconstructed)

```
relative_humidity_pct   (Magnus formula from temp + dew point)
wind_speed_ms           (hypot from u/v)
wind_direction_deg      (atan2 from u/v, meteorological)
current_speed_ms        (hypot from u/v)
current_direction_deg   (atan2 from u/v, oceanographic)
tidal_level_m           (UTide baseline + residual)
significant_wave_height_m     (exp from log)
zero_crossing_period_s        (exp from log)
significant_wave_period_s     (RidgeCV calibrator)
peak_wave_period_s            (RidgeCV calibrator)
conductivity_mscm             (RidgeCV calibrator)
global_radiation_wm2          (clear-sky * clearness)
compass_deg                   (NOT_MODELLED - excluded)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test
pytest tests/test_model_shapes.py -v

# With coverage
pytest tests/ --cov=src/portland_itransformer
```

## Provenance

**Dataset**: SYNTHETIC, Portland Harbor 2025
**Source**: Provided calibrated CSV, not NOAA/Copernicus/observed data
**Use**: Pipeline testing, hyperparameter tuning, feature validation only

⚠️ Strong scores on this synthetic data do NOT indicate real-world forecast performance.

## Documentation

See `SPECIFICATION.md` for complete technical specification including:

- Feature transformations (Section 7)
- Time splits and leakage prevention (Section 8)
- Loss function and training (Section 11)
- Reconstruction rules (Section 16)
- Acceptance criteria (Section 20)

## Next Steps

1. **Implement scaling pipeline** (`scaling.py`)
2. **Implement calibrators** (`calibrators.py`)
3. **Implement dataset** (`dataset.py`)
4. **Implement training loop** (`train.py`)
5. **Implement evaluation** (`evaluate.py`, `metrics.py`)
6. **Implement reconstruction** (`reconstruct.py`)
7. **Implement CLI** (`cli.py`)
8. **Add comprehensive tests**

## Contact

For questions about the specification or implementation, refer to `SPECIFICATION.md` or the issue tracker.

---

**Version**: 0.1.0 (Phase 1: Scaffolding)  
**Last Updated**: 2026-06-25
