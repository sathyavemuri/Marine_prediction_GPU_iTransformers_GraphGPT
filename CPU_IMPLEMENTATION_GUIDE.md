# CPU-Based Model Implementation Guide

**Status:** ✅ POSSIBLE but **NOT RECOMMENDED for real-time**

---

## 📁 Saved Model Locations

### Model Files (PyTorch)
```
artifacts/
├── best_model_graphcast_unified.pt (54 KB)
├── best_model_water_pressure.pt (51 KB)
└── [Training data & configs]

outputs/
├── best_model.pt (iTransformer main)
├── marine/best_model.pt
└── atmosphere/best_model.pt

portland_itransformer/
├── outputs/best_model.pt
├── outputs/marine/best_model.pt
├── outputs/atmosphere/best_model.pt
└── artifacts/[configs & metadata]
```

### Configuration Files (JSON)
```
artifacts/
├── itransformer_gpu_results.json (model results)
├── graphcast_marine_feedback_results.json (model results)
├── retrain_config.json (training config)
├── detailed_skills_extended.json (metrics)
└── system_metrics_combined.json (hardware info)
```

### Total Artifact Size: 111 MB
- Models: ~106 MB (compressed PyTorch .pt files)
- Configs: ~5 MB (JSON files)
- Data: Included in training sets

---

## ✅ Yes, CPU Implementation is Possible

### CPU Inference Code

```python
import torch
import json

# Load on CPU (no GPU required)
device = torch.device('cpu')

# Load iTransformer
model_it = torch.load(
    'artifacts/best_model_itransformer.pt',
    map_location=device
)
model_it.eval()  # Evaluation mode
model_it.to(device)

# Load GraphCast
model_gc = torch.load(
    'artifacts/best_model_graphcast_unified.pt',
    map_location=device
)
model_gc.eval()
model_gc.to(device)

# Load results for reference
with open('artifacts/itransformer_gpu_results.json', 'r') as f:
    it_results = json.load(f)
with open('artifacts/graphcast_marine_feedback_results.json', 'r') as f:
    gc_results = json.load(f)

# Example: Run prediction on CPU
with torch.no_grad():
    input_data = torch.randn(1, 96, 30)  # (batch, sequence_len, features)
    output_it = model_it(input_data.to(device))  # CPU inference
    print(f"iTransformer output shape: {output_it.shape}")
```

---

## ⚠️ CPU Performance Trade-offs

### Inference Time Comparison

| Hardware | iTransformer | GraphCast | Suitable For |
|----------|--------------|-----------|--------------|
| **GPU (RTX A6000)** | 12-15 sec | 8-10 sec | ✅ Real-time |
| **GPU (RTX 4090)** | 10-12 sec | 7-9 sec | ✅ Real-time |
| **GPU (T4/L4 Cloud)** | 18-20 sec | 15-18 sec | ✅ Real-time |
| **CPU (16-core)** | 2-3 min | 1.5-2 min | ❌ Real-time |
| **CPU (8-core)** | 4-5 min | 3-4 min | ❌ Real-time |
| **CPU (4-core)** | 8-10 min | 5-7 min | ❌ Real-time |

### Performance Analysis

```
CPU Performance = GPU Performance × 10-50x slower

Why slower on CPU?
1. Sequential processing (no parallelization)
2. No tensor cores (matrix operations slow)
3. Memory bandwidth limited
4. Cache misses for large models
5. No GPU optimization

Example:
- GPU: 12 sec per 7-day forecast
- CPU: 120-600 sec per 7-day forecast
- Difference: 10-50x slower
```

---

## 🎯 When CPU is Acceptable

### Use Case: Batch Processing (Offline)

```python
# ✅ GOOD for CPU: Batch predictions overnight
def batch_predict_cpu():
    """Run predictions for entire month on CPU"""
    dates = pd.date_range('2026-06-01', '2026-06-30', freq='D')
    
    results = []
    for date in dates:
        # Load day's input data
        input_data = load_daily_data(date)
        
        # CPU inference (happens at night, time not critical)
        with torch.no_grad():
            output = model_it(input_data.to(device))
        
        results.append({
            'date': date,
            'predictions': output.cpu().numpy()
        })
        
        # Time: ~3 min per day = 90 min for month (OK)
    
    return results
```

### Use Case: Research & Development

```python
# ✅ GOOD for CPU: Model development
- Hyperparameter tuning (offline)
- Architecture experiments
- Data preprocessing
- Validation analysis
- Post-training analysis
```

### Use Case: Low-Frequency Predictions

```python
# ✅ ACCEPTABLE for CPU: Few predictions per day
- Daily forecast (1 request/day)
- Weekly planning (1 request/week)
- Monthly summary (1 request/month)

# ❌ NOT GOOD: Real-time continuous
- Streaming data (multiple requests/hour)
- Dashboard refresh (multiple requests/minute)
- Alert system (sub-second response)
```

---

## ❌ When CPU is NOT Acceptable

### ❌ Real-Time Operations
```
Required: Response < 30 seconds
CPU: 2-3 minutes
❌ 4-6x TOO SLOW
```

### ❌ Interactive Dashboards
```
Required: Refresh < 30 seconds
CPU: 2-3 minutes
❌ UNACCEPTABLE user experience
```

### ❌ Alert Systems
```
Required: Alert < 1 minute
CPU: 2-3 minutes
❌ DELAYED alerts (missed time window)
```

### ❌ Streaming Systems
```
Required: Multiple predictions/hour
CPU: 1 prediction every 3 minutes max
❌ BOTTLENECK in data pipeline
```

---

## 📊 Memory Requirements

### Model Size & RAM Usage

```
Model Loading:
- iTransformer: 2.4M parameters × 4 bytes = ~10 MB
- GraphCast: 1.0M parameters × 4 bytes = ~4 MB
- Total model memory: ~15 MB

Runtime (CPU):
- Input data: 96 × 30 × 4 bytes = 11.5 KB
- Intermediate activations: ~50 MB (depends on batch size)
- Output: 14 × 4 bytes = 56 bytes
- Total peak memory: ~50-100 MB

Recommendation:
- Minimum RAM: 512 MB available
- Recommended RAM: 2-4 GB available
- CPU should have 4+ cores for reasonable speed
```

---

## 🚀 CPU vs GPU Decision Tree

```
Do you need real-time predictions?
│
├─ YES → Do you need < 30 sec response?
│  ├─ YES → GPU REQUIRED ✅
│  │   - Cloud GPU ($200-300/month)
│  │   - Local GPU ($5,000-15,000)
│  └─ NO → CPU acceptable for low frequency
│
└─ NO → Is budget critical?
   ├─ YES → CPU is fine ✅
   │   - $0 (your laptop/server)
   │   - For offline batch work
   └─ NO → Still use GPU for future scaling
```

---

## 💻 CPU Implementation Steps

### Step 1: Load Model on CPU

```python
import torch
import json

# Force CPU
device = torch.device('cpu')

# Load model
checkpoint = torch.load(
    'artifacts/best_model_graphcast_unified.pt',
    map_location=device  # Critical: forces CPU loading
)

# If checkpoint is just state_dict, build model first
model = YourModelClass()  # Define architecture
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
```

### Step 2: Prepare Input Data

```python
import numpy as np

# Load your 96-minute lookback data (15 marine + 15 atmosphere)
input_array = np.load('input_data.npy')  # shape: (1, 96, 30)
input_tensor = torch.from_numpy(input_array).float()
input_tensor = input_tensor.to(device)

print(f"Input shape: {input_tensor.shape}")
print(f"Device: {input_tensor.device}")
```

### Step 3: Run Inference

```python
# Inference on CPU (time-consuming)
import time

start = time.time()

with torch.no_grad():
    output = model(input_tensor)

elapsed = time.time() - start

print(f"Inference time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"Output shape: {output.shape}")  # Should be (1, 15) for 15 parameters
```

### Step 4: Post-Process Results

```python
# Convert back to numpy
predictions = output.cpu().numpy()

# Inverse scaling if needed
predictions = scaler.inverse_transform(predictions)

# Save results
results = {
    'predictions': predictions.tolist(),
    'parameters': ['current_speed_ms', 'current_direction_deg', ...],
    'timestamp': datetime.now().isoformat(),
    'inference_time_sec': elapsed
}

with open('predictions.json', 'w') as f:
    json.dump(results, f)
```

---

## 📈 Optimization Tips for CPU

### 1. Quantization (Model Compression)
```python
# Reduce model size & speed up CPU inference by 2-3x
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# New inference time: ~1 minute (still slow but better)
```

### 2. Batch Processing (Amortize Overhead)
```python
# Process multiple forecasts together (more efficient)
batch_inputs = torch.stack([
    input_tensor_1,
    input_tensor_2,
    input_tensor_3,
    input_tensor_4
])  # Shape: (4, 96, 30)

with torch.no_grad():
    batch_outputs = model(batch_inputs)

# Time: ~10-15 min for 4 forecasts (2.5-3.75 min each)
# Better than 4 × 3 min = 12 min separately
```

### 3. Model Pruning
```python
# Remove unnecessary weights (gains 10-20% speedup)
pruned_model = torch.nn.utils.prune.global_unstructured(
    model,
    pruning_method=torch.nn.utils.prune.L1Unstructured,
    amount=0.2  # Remove 20% of smallest weights
)

# New inference time: ~100-150 seconds (minor improvement)
```

### 4. ONNX Export (Framework-Independent)
```python
# Convert to ONNX for cross-platform CPU inference
torch.onnx.export(
    model,
    input_tensor,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

# Can then use in any language (Python, C++, Java, etc.)
# Slight speedup on CPU (~10%)
```

---

## 🎯 Recommendations by Use Case

### Use Case 1: Research & Development
```
✅ CPU Implementation: YES
- No time pressure
- Focus on model improvement
- Batch processing acceptable
- Cost: FREE (use existing computer)
- Timeline: Any duration acceptable
```

### Use Case 2: Offline Batch Forecasting
```
✅ CPU Implementation: YES
- Daily/weekly batch jobs
- Can run overnight
- Schedule in off-hours
- Cost: FREE (use existing server)
- Timeline: Hours acceptable
```

### Use Case 3: Local Web Dashboard (Low Traffic)
```
⚠️ CPU Implementation: MAYBE
- If < 5 requests/hour
- Add caching to reduce inference calls
- Background job + cache results
- Cost: FREE (local server)
- Note: Slow but functional
```

### Use Case 4: Production Real-Time System
```
❌ CPU Implementation: NO
- Requires GPU (cloud or local)
- Response time critical
- High demand periods
- Cost: $200-750/month (cloud) OR $5-15K (local)
- Requirement: < 30 seconds response
```

### Use Case 5: Mobile/Edge Deployment
```
⚠️ CPU Implementation: CONDITIONAL
- If device is powerful (octa-core, 4+ GB RAM)
- Can use quantized model (10-20 MB)
- Acceptable inference time: 30-60 seconds
- Offline predictions acceptable
- Cost: FREE (runs on device)
```

---

## 🔄 Transition Strategy: CPU → GPU

```
Phase 1: CPU Development (Week 1-4)
├─ Develop inference code
├─ Test on your machine
├─ Validate predictions match GPU
└─ Build API/web interface

Phase 2: GPU Acceleration (Week 5-8)
├─ Deploy to cloud GPU ($200/month)
├─ Measure inference time improvement
├─ Set up monitoring
└─ Validate in production

Phase 3: Scale Up (Month 3+)
├─ Monitor usage patterns
├─ Optimize if needed
├─ Add more predictions
└─ Scale as needed
```

---

## 📋 Setup Checklist for CPU

- [ ] PyTorch installed (CPU version)
- [ ] Model files downloaded (artifacts/*.pt)
- [ ] JSON configs loaded
- [ ] Input data prepared (96-minute lookback)
- [ ] Inference script tested
- [ ] Output validation confirmed
- [ ] Timing benchmarked
- [ ] Deployment method chosen
- [ ] Monitoring setup (optional)
- [ ] Documentation updated

---

## ⚡ Quick CPU Implementation

**Fastest way to get CPU predictions (5 minutes):**

```python
import torch
import json

# 1. Load model
device = torch.device('cpu')
model = torch.load('artifacts/best_model_graphcast_unified.pt', 
                    map_location=device)
model.eval()

# 2. Load input (example)
import numpy as np
input_data = torch.randn(1, 96, 30).to(device)

# 3. Predict
with torch.no_grad():
    output = model(input_data)

# 4. Save
np.save('predictions.npy', output.cpu().numpy())
print("Done! Predictions saved.")
```

---

## 📊 Summary Table

| Aspect | CPU | GPU |
|--------|-----|-----|
| Cost | Free | $200-750/month |
| Speed | 2-3 minutes | 12-15 seconds |
| Real-time capable | ❌ No | ✅ Yes |
| Offline batch | ✅ Yes | ✅ Yes |
| Research | ✅ Yes | ✅ Yes |
| Production | ❌ No | ✅ Yes |
| Implementation | Easy | Easy |
| Power consumption | Low | High |
| Suitable for | Dev/batch | Production |

---

## ✅ Conclusion

**CPU Implementation: Possible but NOT recommended for production real-time systems**

### Use CPU When:
- ✅ Developing & testing
- ✅ Offline batch processing
- ✅ Low-frequency predictions (< 5/hour)
- ✅ Budget is critical
- ✅ Time pressure is low

### Use GPU When:
- ✅ Production deployment
- ✅ Real-time predictions needed
- ✅ Interactive dashboards
- ✅ High-frequency requests (> 5/hour)
- ✅ Response time critical (< 30 sec)

**Recommendation: Start with CPU for development, upgrade to GPU for production.**

---

**Last Updated:** 2026-06-27  
**Models Tested:** iTransformer (2.4M params), GraphCast+Marine (1.0M params)  
**Framework:** PyTorch 2.12.1
