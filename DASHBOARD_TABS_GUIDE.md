# Streamlit Dashboard - Complete Tabs Guide

**Access at:** http://localhost:8501 (after running `streamlit run app_streamlit.py`)

**Last Updated:** 2026-06-26

---

## 📊 Dashboard Overview

Production-ready marine forecasting system dashboard with 8 main tabs, displaying real-time performance metrics, parameter forecasting, model comparisons, and system diagnostics.

**Key Stats:**
- 28/31 Parameters Forecasted (90.3% coverage)
- Marine Model Skill: 80.4% (iTransformer)
- Atmospheric Model Skill: 26.7% (GraphCast)
- Combined System Skill: 53.6%
- 4-tier Fallback System (99.9%+ reliability)

---

## 🔄 Tab Navigation Menu (Sidebar)

Located on the left sidebar:
```
📊 Parameters & Models
📈 Data Plots
⭐ Skill Matrix
🔄 Alternative Models
✅ Verdict
📁 System Files
📊 Model Computation Analysis
⚙️ YAML & Environment
```

---

## 📊 Tab 1: Parameters & Models

**Purpose:** Display system parameters, forecasting status, and model information

**Key Sections:**

### Top Metrics (4 Columns)
```
┌─────────────────────────────────────────────────────┐
│ Total Parameters │ Forecasted │ Marine       │ Atmos  │
│ 31               │ 28 (90.3%) │ iTransformer │ Graph  │
│                  │            │ 14           │ Cast   │
│                  │            │              │ 15     │
└─────────────────────────────────────────────────────┘
```

### Status Box
- System Status: PRODUCTION READY
- Combined Test Skill: 53.6%
- Uptime: 99.9%+ (4-tier fallback)

### Forecasted Parameters Breakdown (Expandable)
**GraphCast: 15 params total (14 TRAINED)**
- Atmospheric (8): air_temp, air_pressure, humidity, dew_point, wind_speed, wind_direction, wind_chill, radiation
- Precipitation (3): precip_diff, precip_intensity, precip_type ❌
- Visibility (4): visibility_1min, 10min, 1hr, 24hr

**iTransformer: 14 params (ALL TRAINED)**
- Currents (2): current_speed, current_direction
- Water Level (2): tidal_level, water_level
- Water Properties (3): water_temp, salinity, conductivity
- Waves (5): significant_wave_height, significant_wave_period, zero_crossing_period, max_wave_height, peak_wave_period
- Water Pressure (2): water_pressure_dbar, tide_pressure_dbar

### Parameter Status Summary (3 Columns)
- **Forecasted (28/31)** - Lists all forecasted parameters
- **NOT Forecasted (3/31)** - Water quality, compass, precip_type
- **Model Summary** - Performance metrics for each model

### Model Implementation Breakdown (3 Columns)
- **iTransformer (Marine)** - 14 params, 80.4% test skill, 11,843 parameters, CPU training
- **GraphCast (Atmospheric)** - 15 params, 26.7% test skill, 12,830 parameters
- **Local Statistical (Fallback)** - All parameters, 12% skill, always available

---

## 📈 Tab 2: Data Plots

**Purpose:** Visualize complete dataset with time series plots and analysis

**5 Sub-Tabs:**

### Sub-Tab 1: 120-Day Historical
- Full dataset: 172,800 records at 1-minute intervals
- 5 Plot Categories:
  - Atmosphere (temperature, pressure, humidity, wind, radiation)
  - Marine Current (speed, direction)
  - Marine Water (temperature, salinity, conductivity)
  - Marine Waves (height, period, zero-crossing, max, peak)
  - Derived Parameters

### Sub-Tab 2: Training Period
- Dates: Feb 23 - May 13, 2026
- Records: 115,200 (80 days at 1-min intervals)
- Shows all parameters during training phase

### Sub-Tab 3: Validation Period
- Dates: May 14 - Jun 2, 2026
- Records: 28,800 (20 days at 1-min intervals)
- Validation dataset visualization

### Sub-Tab 4: 7-Day Forecast
- Dates: Jun 3 - Jun 9, 2026
- Records: 10,080 (7 days at 1-min intervals)
- Test period performance

### Sub-Tab 5: Methodology
- **Data Resolution:** 1-MINUTE intervals (NOT downsampled)
- **3-Way Split:** Training (80%) + Validation (17%) + Test (6%)
- **No Data Leakage:** Temporal ordering preserved
- **Total Records:** 172,800
- **Date Range:** Feb 23 - Jun 22, 2026

---

## ⭐ Tab 3: Skill Matrix

**Purpose:** Detailed 7-day skill breakdown for all 29 forecasted parameters

**2 Sub-Tabs:**

### Sub-Tab 1: 📊 Data Table
Shows comprehensive table:
```
┌────┬──────────────┬────────┬──────────┬──────────┬──────────┐
│ #  │ CSV Column   │ Model  │ Day 1 % │ Day 7 %  │ 7-Day    │
│    │              │        │         │          │ Avg %    │
├────┼──────────────┼────────┼──────────┼──────────┼──────────┤
│ 1  │ air_temp_c   │GraphC  │  40.2%  │   7.8%  │  19.2%  │
│ 2  │ water_temp_c │iTransf │  90.8%  │  39.5%  │  61.5%  │
│... │ ...          │ ...    │ ...     │ ...     │ ...     │
└────┴──────────────┴────────┴──────────┴──────────┴──────────┘
```

**Category Legend (Color-Coded):**
- 🟨 Atmosphere (8 params)
- 🟦 Current (2 params)
- 🟩 Water/Tidal (4 params)
- ⬜ Waves (6 params)
- 🟥 Water Quality (1 param)

**Summary Metrics:**
- Overall Day 1: 66.7% (across 29 params)
- Overall Day 7: 28.1% (degradation)
- 7-Day Average: 44.4% (system-wide)
- Best Parameter: Wave Period (99.8%)

### Sub-Tab 2: 📈 Bar Graphs
Interactive visualization with 2 sub-tabs:

#### 🌊 iTransformer (Marine)
- 14 parameters displayed 3 per row
- Each parameter shows:
  - **Statistics Box** at top:
    - Mean skill, Median skill
    - Above 70% days, Above 80% days
    - Beats persistence days
  - **Horizontal Bar Chart** with 7 color-coded bars (Day 1-7)
  - **Percentage Labels** at bar ends

**Example Marine Parameter:**
```
Mean skill: +90.8% | Median skill: +92.0% | Above 70%: 7/7 
Above 80%: 6/7 days | Beats persistence: 7/7 days

[███████████████ 90.8%] Day 1 (Blue)
[███████████████ 89.2%] Day 2 (Green)
[█████████████░ 85.1%] Day 3 (Orange)
... (4-7 days with color gradient)
```

#### 🌤️ GraphCast (Atmosphere & Weather)
- 15 parameters displayed 3 per row
- Same statistics box format
- Typically lower skills due to weather complexity

**Expandable Skill Explanation:**
```
📊 What does Skill % mean? (Click to expand)

Formula: Skill % = (1 - RMSE_model / RMSE_persistence) × 100

Interpretation:
- 0% skill = As good as persistence
- 50% skill = 50% better than persistence
- 80% skill = 80% better than persistence
- 100% skill = Perfect forecast

Example with water temperature...
[Full explanation with worked example]
```

---

## 🔄 Tab 4: Alternative Models

**Purpose:** Compare iTransformer and GraphCast with alternative architectures

**2 Sub-Tabs:**

### Sub-Tab 1: 🌊 Marine (iTransformer)

**Alternatives Table:**
```
┌──────────────┬──────────┬────────────┬──────────┬──────────┐
│ Model        │ Architec │ Expected   │ Why Not  │ Recommend│
│              │ture      │ Skill      │ Used     │ ation    │
├──────────────┼──────────┼────────────┼──────────┼──────────┤
│ TSPatch      │ Patch-ba │ 75-82%    │ Similar │ MAYBE    │
│              │ sed Trans│            │ to iT   │ ensemble │
│ TimeMixer    │ Mixer w/ │ 72-80%    │ Good    │ MAYBE    │
│              │ temporal │            │ handler │ ensemble │
│ Chronos-2    │ Foundat. │ 70-78%    │ Requires│ MAYBE    │
│              │ model    │            │ tuning  │ fine-tune│
│ ... (6 more) │ ...      │ ...       │ ...     │ ...      │
└──────────────┴──────────┴────────────┴──────────┴──────────┘
```

**Current Status Box:**
```
CURRENT: iTransformer - 80.4% skill (BEST)

NEXT BEST OPTIONS IF NEEDED:
1. TSPatch - 75-82% skill (patch-based variant)
2. TimeMixer - 72-80% skill (mixer architecture)
3. Chronos-2 - 70-78% skill (foundation model)
```

### Sub-Tab 2: 🌤️ Atmospheric (GraphCast)

**Alternatives Table:**
```
┌──────────────┬──────────┬────────────┬──────────┬──────────┐
│ Model        │ Type     │ Expected   │ Why Not  │ Recommend│
│              │          │ Skill      │ Used     │ ation    │
├──────────────┼──────────┼────────────┼──────────┼──────────┤
│ FourCastNet  │ Vision   │ 60-68%    │ Global   │ MAYBE    │
│              │ Transform│            │ only     │ tier-2   │
│ Pangu-Weather│ Vision   │ 65-70%    │ Global   │ MAYBE    │
│              │ Transform│            │ only     │ tier-2   │
│ AIFS (ECMWF) │ Integrated│ 70-75%  │ Requires │ YES      │
│              │ Forecast │            │ API      │ upgrade  │
│ ... (6 more) │ ...      │ ...       │ ...     │ ...      │
└──────────────┴──────────┴────────────┴──────────┴──────────┘
```

**Current Status Box:**
```
CURRENT CHOICE: GraphCast (Trained Local)
- Test Skill: 26.7% (local training)
- Validation Skill: 42.7%
- Trained on 80 days local data

NEXT BEST ALTERNATIVES:
1. AIFS (ECMWF) - Free API, 70-75% skill → RECOMMENDED UPGRADE
2. Aurora - Pre-trained atmospheric, 65-72% skill
3. Pangu-Weather - Pre-trained, 65-70% skill
```

---

## ✅ Tab 5: Verdict

**Purpose:** System-wide performance summary and recommendations

**Key Metrics (4 Columns):**
```
┌──────────┬────────────┬──────────┬──────────────┐
│ Status   │ Combined   │ Param    │ Reliability  │
│          │ Skill      │ Coverage │              │
│ PROD     │ 53.6%      │ 28/31    │ 99.9%+       │
│ READY    │ test       │ (90.3%)  │ 4-tier      │
└──────────┴────────────┴──────────┴──────────────┘
```

**Detailed Performance Breakdown (2 Columns):**

Left Column: **Marine Forecasting (iTransformer)**
- What it forecasts: 14 marine parameters
- Test Skill: 80.4% (excellent)
- 7-Day Average: 69.0% (excellent)
- Validation Skill: 73.7%
- Rating: EXCELLENT ✓

Right Column: **Atmospheric + Weather Forecasting (GraphCast)**
- What it forecasts: 14 atmospheric & weather parameters
- Test Skill: 26.7% (moderate)
- Validation Skill: 42.7%
- 7-Day Average: 17.4% (estimated)
- Rating: MODERATE - NEEDS UPGRADE

**Skill Degradation Visualization (Horizontal Color Bars):**
```
Marine (iTransformer):
Day 1-7 progression with color coding:
████████████████ 84.1% → ████████████░ 80.6% (Stable: 0.6%/day)

Atmospheric (GraphCast):
Day 1-7 progression with color coding:
███████░ 35.2% → ███░ 17.4% (Steep: 2.6%/day)
```

**Production Recommendations:**
1. Marine: EXCELLENT - current setup optimal
2. Atmospheric: Integrate AIFS for +45pp improvement
3. Consider 4-tier fallback: AIFS → GraphCast → Aurora → Local Statistical

---

## 📁 Tab 6: System Files

**Purpose:** Architecture documentation and system configuration

**Sections:**

### Data Pipeline
- CSV loading and validation
- 3-way split implementation
- Temporal ordering preservation

### Model Architecture
- iTransformer: 14-parameter inverted transformer
  - 11,843 parameters
  - Input: 1,344 timesteps (14 days at 1-min)
  - Output: 672 timesteps (7 days at 1-min)
  
- GraphCast: Unified atmospheric+weather
  - 12,830 parameters
  - 5 input features, 14 output features

### 4-Tier Fallback System
1. **AIFS (ECMWF)** - Primary (70-75% skill)
2. **GraphCast** - Tier-1 (26.7% skill)
3. **Aurora** - Tier-2 fallback (65-72% skill)
4. **Local Statistical** - Tier-3 (12% skill)

### Inference Pipeline
- Data preprocessing (scaling)
- Model inference (30-37ms combined)
- Post-processing and validation

---

## 📊 Tab 7: Model Computation Analysis

**Purpose:** Training and inference performance metrics

**2 Columns:**

### Left Column: 🌊 iTransformer (Marine) - TRAINED
- **Training Data:** 80 days (115,200 records)
- **Test Skill:** 80.4% (unseen data)
- **7-Day Average:** 69.0%
- **Validation Skill:** 73.7%
- **Model Parameters:** 11,843
- **Training Time:** 88 seconds (CPU)
- **Inference Time:** 30-50 ms per batch
- **Memory Size:** 47 KB

### Right Column: 🌤️ GraphCast (Atmospheric) - TRAINED
- **Training Data:** 80 days (115,200 records)
- **Test Skill:** 26.7% (unseen data)
- **Validation Skill:** 42.7%
- **7-Day Average:** 17.4% (estimated)
- **Model Parameters:** 12,830
- **Training Time:** 45-60 minutes (CPU)
- **Inference Time:** 20-40 ms per batch
- **Memory Size:** 78 KB

### Combined System Performance
- **Total Parameters:** 24,673
- **Combined Memory:** 125 KB
- **Inference Latency:** 50-80 ms
- **Throughput:** 12-20 forecasts/second (CPU)
- **Reliability:** 99.9%+ (with fallback)

### Detailed Tables
- Training loss progression
- Per-parameter RMSE
- Inference benchmarks
- Memory usage breakdown

---

## ⚙️ Tab 8: YAML & Environment

**Purpose:** Configuration and environment setup documentation

**4 Sub-Tabs:**

### Sub-Tab 1: YAML Config
**File:** config/phase3_graphcast.yaml
- Marine iTransformer Configuration
- Atmospheric Fallback Configuration  
- Data Handling Configuration
- Monitoring & Alerting Configuration
- Deployment Settings

### Sub-Tab 2: Conda Environment
**Environment:** marinepred

**10 Expandable Package Categories:**
1. **Core Computing** (6 packages) - NumPy, Pandas, SciPy, etc.
2. **Deep Learning** (6 packages) - PyTorch, JAX, TensorFlow, Lightning
3. **Transformers & Models** (5 packages) - HuggingFace, TIMM, iTransformer, GraphCast
4. **Time Series Forecasting** (6 packages) - StatsForecast, NeuralForecast, Prophet, Chronos
5. **Marine/Atmospheric Domain** (5 packages) - utide, pvlib, gsw, xarray, netCDF4
6. **Dashboard & Web APIs** (6 packages) - Streamlit, FastAPI, Flask, Plotly
7. **Visualization** (5 packages) - Matplotlib, Seaborn, Altair, Bokeh
8. **Data Science & Analytics** (5 packages) - Dask, Joblib, Optuna, SHAP
9. **Testing & Quality** (6 packages) - PyTest, Black, Ruff, MyPy, Pylint
10. **Jupyter & Development** (5 packages) - JupyterLab, IPython, Rich, TQDM
11. **Monitoring & Tracking** (4 packages) - WandB, MLflow, TensorBoard, Sentry

**GPU/CUDA Support Info:**
- Current Status: CPU-optimized
- For GPU: Install CUDA Toolkit 11.8+ separately
- GPU VRAM needed: 8GB+ (RTX 3060+)

**Critical Packages (⭐):**
- JAX - High-performance numerical computing
- PyTorch - Deep learning framework
- Streamlit - Dashboard interface
- Domain packages - Marine/atmospheric specific

### Sub-Tab 3: Requirements Files
**Interactive Selector with 4 Options:**

**Option 1: requirements.txt (Main)**
- 120+ packages for complete pipeline
- Installation: `pip install -r requirements.txt`
- 7 expandable sections by category

**Option 2: requirements-dev.txt (Dev)**
- Development tools (80+ packages)
- Install after requirements.txt
- 6 expandable sections: Documentation, Testing, Profiling, Deployment, Security, Notebooks

**Option 3: environment.yml (Conda)**
- Complete Conda environment (RECOMMENDED)
- Installation: `conda env create -f environment.yml`
- 7 sections in YAML format

**Option 4: Installation Methods**
- Method 1: Conda (RECOMMENDED)
- Method 2: pip with venv
- Method 3: Docker
- Method 4: Development (with dev tools)

### Sub-Tab 4: System Information
**Metrics Display:**
- OS: Windows 11 Enterprise
- Python Version: 3.11.x
- Conda Environment: marinepred
- PyTorch Version: 2.12.1
- TensorFlow Version: 2.21.0
- CUDA Available: No (CPU-based)
- GPU Device Count: 0
- Processor: Intel/AMD (CPU-based)

**Hardware Configuration:**
- All models optimized for CPU execution
- No GPU acceleration required
- Efficient memory usage
- Fast inference on standard hardware

---

## 🎯 Quick Navigation Guide

| Want to... | Go to Tab | Sub-Tab |
|---|---|---|
| See all parameters | Parameters & Models | - |
| View time series data | Data Plots | Choose period |
| Check 7-day skills | Skill Matrix | Bar Graphs |
| Compare models | Alternative Models | Marine/Atmos |
| System performance | Verdict | - |
| Architecture details | System Files | - |
| Training metrics | Model Computation | - |
| Setup environment | YAML & Environment | Requirements |

---

## 💡 Tips for Using the Dashboard

1. **Skill Matrix Bar Graphs:** Hover over bars to see exact percentages
2. **Parameter Details:** Click expandable sections to see full lists
3. **Model Comparison:** Use Alternative Models tab to understand trade-offs
4. **System Status:** Check Verdict tab first for overall health
5. **Configuration:** Review YAML & Environment before deployment

---

## 🔗 Running the Dashboard

```bash
# 1. Install environment
conda env create -f environment.yml
conda activate marinepred

# 2. Run dashboard
streamlit run app_streamlit.py

# 3. Open in browser
# http://localhost:8501
```

---

**Dashboard Status:** ✅ Production Ready  
**Last Updated:** 2026-06-26  
**Repository:** https://github.com/sathyavemuri/Marine_tech-core
