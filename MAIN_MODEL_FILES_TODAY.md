# Main Model-Related Python Files - TODAY vs. OLD

**Last Updated:** 2026-06-26

---

## 🟢 CURRENT/PRODUCTION MODEL FILES (TODAY - Use These!)

### 1. **Dashboard & Main Application**
```
✅ app_streamlit.py (1600+ lines)
   └─ Production Streamlit dashboard
   └─ Shows iTransformer (80.4% skill) + GraphCast (26.7% skill)
   └─ 8 main tabs with visualizations
   └─ USE THIS: Main interface for all forecasting
```

### 2. **iTransformer Training & Inference**
```
✅ retrain_with_extended_features.py
   └─ Current training script for iTransformer
   └─ 14 marine parameters (2 new: conductivity, peak_wave_period)
   └─ Extended features implementation
   └─ 3-way split: Training (80 days) + Validation (20 days) + Test (7 days)
   └─ USE THIS: Train/retrain iTransformer

✅ train_graphcast_unified.py
   └─ GraphCast unified model training
   └─ 15 parameters: 8 atmosphere + 3 precipitation + 4 visibility
   └─ Atmosphere + Weather combined forecasting
   └─ USE THIS: Train GraphCast with weather parameters

✅ train_graphcast_weather.py
   └─ GraphCast weather-specific training
   └─ 6 weather parameters focus
   └─ USE THIS: Alternative GraphCast variant
```

### 3. **Portland iTransformer Implementation**
```
✅ portland_itransformer/
   ├─ src/portland_itransformer/
   │  ├─ models/marine_itransformer.py    ← Main model architecture
   │  ├─ train.py                         ← Training pipeline
   │  ├─ evaluate.py                      ← Evaluation metrics
   │  ├─ dataset.py                       ← Data loading
   │  └─ metrics.py                       ← Skill calculations
   │
   ├─ train_marine.py                     ← Marine training script
   ├─ train_atmosphere.py                 ← Atmosphere training script
   ├─ train_unified.py                    ← Unified 22-param training
   └─ run_training.py                     ← Entry point for training

   STATUS: Production-ready
   USE THESE: Core iTransformer implementation for marine parameters
```

### 4. **Marine MTGNN Implementation** 
```
✅ marine_local_mtgnn/
   ├─ src/marine_local_mtgnn/
   │  ├─ models/mtgnn.py                 ← Graph neural network
   │  ├─ models/temporal.py              ← Temporal module
   │  ├─ models/graph_conv.py            ← Graph convolution
   │  ├─ training/trainer.py             ← Training logic
   │  ├─ data/loader.py                  ← Data loading
   │  ├─ evaluation/metrics.py           ← Evaluation
   │  └─ pipeline.py                     ← Full pipeline
   │
   ├─ tests/test_mtgnn.py                ← Model tests
   └─ tests/test_training.py             ← Training tests

   STATUS: Production-ready with tests
   USE THIS: Alternative graph-based architecture (if needed)
```

### 5. **Data & Configuration**
```
✅ marine_data_120days_1min.csv
   └─ Main training dataset
   └─ 172,800 records (120 days at 1-minute intervals)
   └─ 31 parameters total (28 forecasted)
   └─ USE THIS: Only data file needed for training

✅ config/phase3_graphcast.yaml
   └─ Production configuration
   └─ iTransformer + GraphCast setup
   └─ Deployment settings
   └─ USE THIS: Configuration for production deployment

✅ artifacts/ folder
   ├─ best_model_graphcast_unified.pt    ← Trained GraphCast weights
   ├─ best_model_itransformer.pt         ← Trained iTransformer weights
   ├─ retrain_results_*.json             ← Training results
   └─ scaler_*.joblib                    ← Data scalers
   
   USE THESE: Pre-trained model weights (no need to retrain)
```

### 6. **Environment & Setup (TODAY - NEW)**
```
✅ environment.yml
   └─ Conda environment with 120+ packages
   └─ JAX 0.4.20+, PyTorch 2.0+, TensorFlow 2.14+
   └─ USE THIS: conda env create -f environment.yml

✅ requirements.txt
   └─ pip requirements (120+ packages)
   └─ USE THIS: pip install -r requirements.txt

✅ requirements-dev.txt
   └─ Development tools (80+ packages)
   └─ USE THIS: pip install -r requirements-dev.txt (after requirements.txt)

✅ SETUP.md
   └─ Complete installation guide
   └─ Troubleshooting, performance benchmarks
   └─ USE THIS: Reference for setup help
```

---

## 🔴 OLD/EXPERIMENTAL FILES (Not for Production - Historical Reference Only)

### Experimental Model Implementations
```
❌ 02_train_itransformer_118days.py
❌ 02_train_itransformer_2day.py
❌ 04_train_hpmixer_118days.py
❌ 08_train_all_horizons_14x.py
❌ 09_train_110days_forecast_10days.py
❌ 13_conv1d_channel_mixer.py
❌ 15_chronos2_finetuned.py
❌ 16_nbeats_multivariate.py
❌ 17_hybrid_8model_nbeats.py
❌ 18_mtgnn_marine.py
❌ 19_hybrid_8model_mtgnn.py
❌ 20_correlated_input_mtgnn.py
❌ 21_hybrid_physics_ml.py
❌ 22_physics_ml_hybrid_correction.py
❌ 23_improved_correlated_mtgnn.py
❌ 23_timexer_marine_120days.py
❌ 24_residual_correction_stacking.py
❌ 25_itransformer_efficient.py
❌ 30_all_models_marine.py
❌ 31_final_model_comparison.py
❌ 33_proper_training.py
❌ 34_with_timexer.py

WHY OLD: These are numbered experiments from iterative development
         Used for research and benchmarking different architectures
         Results already captured in current models
         SKIP THESE: Use current production files instead
```

### Old Dashboard Apps
```
❌ app.py
❌ app_all_models_comparison.py
❌ app_chronos2.py
❌ app_hybrid.py
❌ app_hybrid_v2.py
❌ app_hybrid_v3.py
❌ app_hybrid_v4.py
❌ app_hybrid_v5.py
❌ app_itransformer.py
❌ app_multihorizon_itransformer.py
❌ app_realdata.py
❌ streamlit_dashboard.py
❌ streamlit_multihorizon_dashboard.py
❌ streamlit_multihorizon_dashboard_PLOTS.py

WHY OLD: Replaced by single comprehensive app_streamlit.py
         Different versions for different experiments
         All features now consolidated in production app
         SKIP THESE: Use app_streamlit.py only
```

### Old Jupyter Notebooks
```
❌ Marine_Forecast_*.ipynb (all versions)
❌ TFT_2day_prediction.ipynb
❌ Plus 50+ other experimental notebooks

WHY OLD: Used during research and development
         Results already implemented in Python scripts
         Notebooks for reference only
         SKIP THESE: Use Python scripts for reproducibility
```

### Evaluation & Comparison Scripts
```
❌ 14_model_comparison_report.py
❌ 05_compare_models.py
❌ evaluate_6models_18params.py
❌ evaluate_all_models_120day.py
❌ portland_itransformer/calc_skill_all_18params.py
❌ portland_itransformer/calc_skill_by_param.py
❌ portland_itransformer/compute_skills.py

WHY OLD: Experimental evaluation from development phase
         Results already shown in dashboard
         SKIP THESE: Use dashboard for current performance metrics
```

---

## 📊 Quick Reference: What to Use

### For Running the Dashboard
```
✅ app_streamlit.py (ONLY CURRENT APP)
   Dependencies: environment.yml or requirements.txt
   Command: streamlit run app_streamlit.py
```

### For Training iTransformer
```
✅ retrain_with_extended_features.py (PRIMARY)
   OR
✅ portland_itransformer/train_marine.py (ALTERNATIVE)
   Data: marine_data_120days_1min.csv
```

### For Training GraphCast
```
✅ train_graphcast_unified.py (RECOMMENDED)
   OR
✅ train_graphcast_weather.py (ALTERNATIVE)
   Data: marine_data_120days_1min.csv
```

### For Production Inference
```
✅ Use pre-trained weights from artifacts/:
   - best_model_itransformer.pt
   - best_model_graphcast_unified.pt
   No need to retrain unless retraining is requested
```

### For Development/Testing
```
✅ marine_local_mtgnn/tests/
   Command: pytest marine_local_mtgnn/tests/
```

---

## 🎯 How to Identify OLD vs. NEW Files

| Characteristic | NEW (Current) | OLD (Experimental) |
|---|---|---|
| **Naming** | `app_streamlit.py`, `train_*.py` (clean names) | `02_train_`, `v4_`, `_hybrid_` (numbered, versioned) |
| **Location** | Root or organized folders | Scattered in root |
| **Purpose** | Production, deployment | Research, benchmarking |
| **Maintenance** | Updated regularly | Frozen, for reference |
| **Status** | Actively used | Archived |
| **Code Quality** | Polished, documented | Experimental, variable |

---

## 📂 Directory Structure: What's Current

```
Marine_Prediction/
├── app_streamlit.py                    ✅ CURRENT (Use this!)
├── retrain_with_extended_features.py  ✅ CURRENT (Use this!)
├── train_graphcast_unified.py          ✅ CURRENT (Use this!)
├── train_graphcast_weather.py          ✅ CURRENT (Use this!)
├── environment.yml                     ✅ CURRENT (Use this!)
├── requirements.txt                    ✅ CURRENT (Use this!)
├── requirements-dev.txt                ✅ CURRENT (Use this!)
├── SETUP.md                            ✅ CURRENT (Use this!)
├── marine_data_120days_1min.csv        ✅ CURRENT (Use this!)
├── config/phase3_graphcast.yaml        ✅ CURRENT (Use this!)
├── artifacts/                          ✅ CURRENT (Use these!)
├── portland_itransformer/              ✅ CURRENT (Production module)
├── marine_local_mtgnn/                 ✅ CURRENT (Production module)
│
└── [OLD FILES - For reference only]
    ├── 02_train_*.py                   ❌ OLD
    ├── 0[4-9]_*.py                     ❌ OLD
    ├── 1[3-9]_*.py                     ❌ OLD
    ├── 2[0-5]_*.py                     ❌ OLD
    ├── app_*.py (except app_streamlit) ❌ OLD
    ├── streamlit_*.py                  ❌ OLD
    ├── Marine_Forecast_*.ipynb         ❌ OLD
    └── ... (50+ experimental files)    ❌ OLD
```

---

## ✅ Summary

### Use THESE Files (Current Production)
1. **app_streamlit.py** - Dashboard
2. **retrain_with_extended_features.py** - iTransformer training
3. **train_graphcast_unified.py** - GraphCast training
4. **portland_itransformer/** - iTransformer module
5. **marine_local_mtgnn/** - MTGNN module
6. **environment.yml** / **requirements.txt** - Setup
7. **marine_data_120days_1min.csv** - Data
8. **artifacts/** - Pre-trained models

### IGNORE THESE Files (Old Experiments)
- All numbered training files (02_train_*, 08_train_*, etc.)
- Old app versions (app_hybrid.py, app_chronos2.py, etc.)
- All Jupyter notebooks (Marine_Forecast_*.ipynb)
- Old evaluation scripts

---

**Date:** 2026-06-26  
**Status:** All current files committed to GitHub  
**Repository:** https://github.com/sathyavemuri/Marine_tech-core
