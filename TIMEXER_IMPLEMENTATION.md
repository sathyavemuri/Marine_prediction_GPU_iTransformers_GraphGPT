# TimeXer for Marine Prediction: Implementation Guide

## Overview

**TimeXer** (Empowering Transformers for Time Series Forecasting with Exogenous Variables) is a novel transformer-based architecture specifically designed to leverage exogenous variables in time series forecasting.

This implementation adapts TimeXer for **marine parameter forecasting** on your 120-day, 18-parameter dataset.

## Problem Context

Your **Correlated Input MTGNN** achieved **+85.0% skill** on 10-day marine forecasting by:
- Creating 18 separate models (one per parameter)
- Coupling each model with intelligently-selected correlated inputs
- Using Graph Convolutional Networks (GCN) to learn nonlinear relationships

**Question:** Can a single unified transformer (TimeXer) match or exceed this ensemble performance?

## TimeXer Architecture

### Core Idea

Instead of multiple separate models, TimeXer uses a **single unified model** with:

1. **Endogenous path**: Patch-based tokenization of historical marine parameters
2. **Exogenous path**: Learnable global token representing external context
3. **Encoder**: Stacked transformer layers with self-attention
4. **Decoder**: Hierarchical projection to 1440-step (10-day) forecast

### Key Components

```
Input: (batch, 288 steps, 21 features)   # 2-day lookback
  |
  +-- Patch Embedding [1/3]
      Divide 288 steps into 12 patches (24 steps each)
      Project each patch: (288, 21) -> (12, 32)
      Add positional encoding
  |
  +-- Exogenous Embedding [2/3]
      Learnable global token (1, 32)
      Represents external context (not specific to any parameter)
  |
  +-- Transformer Encoder [3/3]
      Self-attention over all tokens (12 patches + 1 exo token)
      d_model=32, n_heads=4, n_layers=1
      Learns which patches matter for the full forecast
  |
Output: (batch, 1440 steps, 21 features)  # 10-day forecast
```

### Why This Architecture?

| Aspect | TimeXer | Correlated MTGNN |
|--------|---------|-----------------|
| **Models** | 1 unified | 18 separate |
| **Training** | ~5-10 min | ~11 min |
| **Inference** | 1 forward pass | 18 sequential passes |
| **Parameter coupling** | Implicit via self-attention | Explicit via input selection |
| **Exogenous handling** | Native (cross-attention) | Manual (input curation) |

## Implementation Details

### Configuration

```python
LOOKBACK = 288          # 2 days @ 10-min resolution
FORECAST = 1440         # 10 days @ 10-min resolution
D_MODEL = 32            # Embedding dimension
N_HEADS = 4             # Multi-head attention heads
N_LAYERS = 1            # Transformer encoder layers
PATCH_LEN = 24          # Steps per patch (24 steps = 240 min = 4 hours)
BATCH_SIZE = 16
EPOCHS = 30
```

### Data Preprocessing

1. **Circular parameters**: wind_direction_deg, current_direction_deg, compass_deg
   - Converted to sin/cos representations (6 features instead of 3)
   - Result: 18 params -> 21 features

2. **Standardization**: StandardScaler on training set

3. **Windowing**: Direct multi-step forecasting
   - Input: 288 consecutive steps (2 days of history)
   - Output: 1440 steps (10 days of forecast)
   - Total windows: ~169k from 110-day training set

### Model Size

- **Total parameters**: 7.9M (vs. MTGNN's 18M for 18 models)
- **Decoder**: Hierarchical compression -> intermediate layer -> expansion
  - Avoids the 30,240-dimensional bottleneck (1440 steps × 21 features)
  - Intermediate: 256-512 dim, tunable

## Training Process

```
[1/6] Load data               172,800 rows × 18 parameters
[2/6] Standardize            110 days training, 10 days test
[3/6] Window data            152,669 training windows
[4/6] Build model            7.9M parameters
[5/6] Train                  30 epochs, early stopping
[6/6] Evaluate               Skill metrics vs. persistence
```

### Optimization

- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Loss**: MSE
- **Scheduler**: ReduceLROnPlateau (patience=6)
- **Early stopping**: Patience=12 epochs
- **Device**: CPU or GPU (auto-detected)

## Expected Results

### Baseline Comparison

Your **Correlated Input MTGNN** skill by parameter (from memory):

| Parameter | MTGNN |
|-----------|-------|
| air_temp_c | +62.9% |
| water_temp_c | +62.9% |
| wind_speed_ms | +55.0% |
| tidal_level_m | +75.0% |
| significant_wave_height_m | **-133.5%** (hard) |
| significant_wave_period_s | **-92.3%** (hard) |
| **Median Skill** | **+85.0%** |

### Success Criteria

- **Excellent** (≥80%): TimeXer competitive with MTGNN → Deploy as single-model alternative
- **Good** (70-80%): TimeXer within 15% → Suitable for low-complexity requirement
- **Fair** (50-70%): TimeXer underperforms → Requires architecture tuning or hybrid
- **Poor** (<50%): TimeXer far below → Stick with Correlated Input MTGNN

## Files Generated

### Training Outputs

1. **timexer_predictions.csv**
   - 1440 rows (10-day forecast) × 18 parameters
   - Actual values for all marine parameters

2. **timexer_metrics.csv**
   - MAE, RMSE, Skill% for each parameter
   - Direct comparison to persistence baseline

### Comparison Outputs

3. **timexer_vs_mtgnn_comparison.csv**
   - Side-by-side: TimeXer vs. MTGNN skill
   - Delta (improvement/regression) per parameter
   - Highlights top gainers/losers

4. **comparison_summary.txt**
   - Verdict: Is TimeXer worth adopting?
   - Parameter-level breakdown
   - Recommendations for production

## Next Steps

### If TimeXer Wins (≥85% median skill)

1. **Deploy as single model**
   - Simpler monitoring (one model vs. 18)
   - Faster inference (~1 forward pass)
   - Easier to retrain (14×-rule still applies)

2. **Optional: Fine-tune**
   - Increase d_model to 64
   - Add N_LAYERS=2
   - Experiment with larger patch lengths (24 vs. 48)

### If TimeXer Underperforms (50-85%)

1. **Hybrid ensemble**: MTGNN for hard params (wave height, etc.) + TimeXer for easy ones
2. **Exogenous conditioning**: Pass iTransformer forecasts as explicit exogenous inputs to TimeXer
3. **Architecture tuning**: Test different patch lengths, embedding dimensions

### If TimeXer Fails (<50%)

1. **Use Correlated Input MTGNN** for production
2. **Why TimeXer lost**:
   - Single model may lack capacity for 18 diverse parameters
   - Parameter coupling learned implicitly (harder than explicit)
   - Architecture designed for structured exogenous data (we used learnable token)

## Code Structure

```
23_timexer_marine_120days.py
├── [1/6] Load data
├── [2/6] Standardization
├── [3/6] Windowed dataset
├── [4/6] TimeXer model definition
├── [5/6] Training loop
└── [6/6] Evaluation & metrics

24_timexer_vs_mtgnn_comparison.py
├── Load TimeXer metrics
├── Load MTGNN baseline
├── Parameter-by-parameter comparison
├── Summary statistics
└── Save comparison CSV
```

## References

- **TimeXer paper**: "Empowering Transformers for Time Series Forecasting with Exogenous Variables"
- **Time-Series-Library**: https://github.com/thuml/Time-Series-Library
- **MTGNN paper**: "Connecting the Dots: Identifying Network Structure via Graph Signal Processing"

## Troubleshooting

### Issue: Training too slow

- **Solution**: Reduce BATCH_SIZE to 8, increase learning rate to 2e-3
- **Or**: Run on GPU (detect automatically if available)

### Issue: Poor validation loss convergence

- **Solution**: Reduce d_model to 16, increase dropout to 0.2
- **Or**: Use warmup (e.g., LambdaLR with linear warmup)

### Issue: Predictions out of bounds

- **Solution**: Clip output after inverse transform: `pred.clip(min_val, max_val)`

## Performance Monitoring

Track during training:

```
Epoch  1/30 | train_loss=0.089765 | val_loss=0.356560 | epoch_time=274.1s
Epoch  2/30 | train_loss=0.087123 | val_loss=0.345234 | epoch_time=273.8s
  [BEST] val_loss: 0.345234
  ...
```

Early stopping triggers at patience=12 with no improvement.

---

**Created**: 2026-06-25  
**Marine Prediction Project**: 120-day forecast, 18 parameters, 10-day horizon
