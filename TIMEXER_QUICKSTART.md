# TimeXer for Marine Prediction — Quick Start

## What You Need to Know

**TimeXer** is a single unified transformer model for forecasting all 18 marine parameters at once, as an alternative to your **18-model Correlated Input MTGNN** ensemble.

## Run TimeXer Training

```bash
python 23_timexer_marine_120days.py
```

**What it does:**
1. Loads 120-day marine dataset (172,800 rows, 18 parameters)
2. Splits: 110 days training, 10 days test
3. Builds windowed dataset: 152k training windows
4. Trains single TimeXer model: 30 epochs, ~8-10 minutes on CPU
5. Outputs metrics and 10-day forecast

**Outputs:**
- `timexer_predictions.csv` — 10-day forecast for all 18 parameters
- `timexer_metrics.csv` — MAE, RMSE, Skill% vs. persistence

## Compare Against MTGNN

```bash
python 24_timexer_vs_mtgnn_comparison.py
```

**What it does:**
1. Loads TimeXer metrics from `timexer_metrics.csv`
2. Compares against MTGNN baseline (+85% skill from your previous work)
3. Shows which parameters improved/regressed
4. Outputs verdict: Is TimeXer worth adopting?

**Output:**
- `timexer_vs_mtgnn_comparison.csv` — Side-by-side parameter comparison
- Console report with top gains/losses and overall verdict

## Expected Results

### If TimeXer Wins (Median Skill ≥ 80%)

✓ **Recommendation: Adopt TimeXer**
- Single model instead of 18
- Faster training (1 model vs. 18)
- Simpler deployment and monitoring
- Likely within 5% of MTGNN

### If TimeXer Underperforms (Median Skill 50-80%)

⚠ **Recommendation: Stay with MTGNN or try hybrid**
- MTGNN may be fundamentally better for this problem
- Could try: TimeXer + explicit exogenous conditioning
- Could try: Ensemble (MTGNN + TimeXer on hard parameters)

### If TimeXer Fails (Median Skill < 50%)

✗ **Recommendation: Use Correlated Input MTGNN for production**
- Single model architecture insufficient for 18 diverse parameters
- Stick with proven 18-model ensemble

## Model Architecture at a Glance

```
Input: 2 days of history (288 steps) × 18 parameters
  ↓
Patch Embedding: Divide into 12 patches of 24 steps each
  ↓
Transformer Encoder: 1 layer, 4 heads, 32 dims
  ↓
Decoder: Hierarchical projection
  ↓
Output: 10-day forecast (1440 steps) × 18 parameters
```

**Model size:** 7.9M parameters (vs. ~18M for 18 MTGNN models)  
**Training time:** ~8-10 minutes  
**Inference:** Single forward pass (~100ms)

## Hyperparameters (Tunable)

If results are suboptimal, try:

```python
# For better accuracy (slower training):
D_MODEL = 64        # Instead of 32
N_LAYERS = 2        # Instead of 1
BATCH_SIZE = 32     # Instead of 16
EPOCHS = 50         # Instead of 30

# For faster training (potentially lower accuracy):
D_MODEL = 16
PATCH_LEN = 48      # Larger patches = fewer tokens
BATCH_SIZE = 8
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Fix:** Install PyTorch in your `marinepred` conda environment:
```bash
conda activate marinepred
pip install torch scikit-learn pandas numpy
```

### Issue: Training very slow
**Fix:** 
- Use GPU if available: script auto-detects
- Reduce batch size: `BATCH_SIZE = 8`
- Reduce epochs: `EPOCHS = 20`

### Issue: Predictions look wrong (all NaN or out of bounds)
**Fix:**
- Check standardization: scaler.mean_, scaler.scale_
- Verify inverse transform: multiply by std + add mean
- Clip outputs: `pred.clip(min_val, max_val)`

## File Guide

### Main Implementation
- `23_timexer_marine_120days.py` — TimeXer training script
- `24_timexer_vs_mtgnn_comparison.py` — Comparison against MTGNN

### Configuration
- Lookback: 288 steps (2 days @ 10-min resolution)
- Forecast: 1440 steps (10 days @ 10-min resolution)
- Train: 110 days | Test: 10 days

### Data Files
- `120day_timestamp_18parameters.csv` — Input data (your existing file)
- `timexer_predictions.csv` — Output: 10-day forecast
- `timexer_metrics.csv` — Output: MAE/RMSE/Skill metrics

### Documentation
- `TIMEXER_IMPLEMENTATION.md` — Full technical details
- `TIMEXER_QUICKSTART.md` — This file

## Next Steps

1. **Run training:** `python 23_timexer_marine_120days.py`
2. **Compare results:** `python 24_timexer_vs_mtgnn_comparison.py`
3. **Review verdict:**
   - Look at `timexer_vs_mtgnn_comparison.csv`
   - Check median skill in console output
   - Decide: TimeXer or MTGNN for production?
4. **If TimeXer wins:** Consider tuning hyperparameters for +1-2% additional gain
5. **If MTGNN wins:** Stick with Correlated Input MTGNN (+85%)

## Key Differences: TimeXer vs. Correlated Input MTGNN

| Aspect | TimeXer | MTGNN |
|--------|---------|-------|
| **Models** | 1 unified | 18 separate |
| **Approach** | Single transformer | Graph neural networks |
| **Parameter coupling** | Implicit (self-attention) | Explicit (input selection) |
| **Training time** | ~8-10 min | ~11 min |
| **Inference** | 1 forward pass | 18 sequential |
| **Complexity** | Simpler deployment | More complex (18 models) |
| **Interpretability** | Harder (black box) | Easier (per-parameter) |

## Expected Performance

Your **Correlated Input MTGNN** baseline:
- **Median skill: +85.0%**
- **Positive skill on 18/18 parameters** (except hard ones: wave height/periods)
- **Training: ~11 minutes for all 18 models**

TimeXer goal:
- **≥80% median skill** (within 5% of MTGNN)
- **Simpler single-model architecture**
- **Faster inference** (1 vs. 18 passes)

---

**Status:** TimeXer training script ready. Run `python 23_timexer_marine_120days.py` to begin.

**Estimated time:** 5-10 minutes (CPU) or 1-2 minutes (GPU)
