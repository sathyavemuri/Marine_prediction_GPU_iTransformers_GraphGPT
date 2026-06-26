# Marine Forecasting System - Complete Setup Guide

**Last Updated:** 2026-06-26  
**System Version:** Phase 3 (iTransformer + GraphCast)  
**Status:** Production Ready

---

## Quick Start

### Option 1: Conda Environment (RECOMMENDED)

```bash
# Clone or navigate to project
cd Marine_Prediction

# Create environment from YAML
conda env create -f environment.yml

# Activate environment
conda activate marinepred

# Verify installation
python -c "import torch, jax, streamlit; print('All packages installed!')"
```

### Option 2: pip with venv (For pip-only systems)

```bash
# Create virtual environment
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify
python -c "import torch, jax, streamlit; print('All packages installed!')"
```

---

## Environment Files Explained

### `environment.yml` (Conda)
**Best for:** Users with Anaconda/Miniconda installed

**Why Conda:**
- ✅ Handles compiled dependencies (CUDA, BLAS, etc.)
- ✅ Binary compatibility guaranteed
- ✅ Faster installation (pre-built wheels)
- ✅ GPU support ready (if CUDA installed)
- ✅ All scientific packages optimized

**Key Features:**
- **Python 3.11** - Latest stable with full type hints
- **PyTorch 2.0+** - Deep learning framework
- **JAX 0.4+** - High-performance numerical computing
- **TensorFlow 2.14+** - Alternative DL framework
- **Jupyter** - Notebook support
- **Streamlit 1.40+** - Dashboard
- **Domain packages** - utide, pvlib, gsw for marine/atmospheric

**Channels used:**
```yaml
channels:
  - conda-forge    # Community packages
  - pytorch        # Official PyTorch
  - defaults       # Anaconda defaults
```

---

### `requirements.txt` (pip)
**Best for:** Docker, CI/CD, constraints, pip-only systems

**Organized Sections:**
1. **Core Dependencies** - NumPy, Pandas, SciPy
2. **Deep Learning** - PyTorch, JAX, TensorFlow, Lightning
3. **Transformers** - HuggingFace, vision models
4. **Time Series** - StatsForecast, NeuralForecast, Chronos
5. **Web Stack** - Streamlit, FastAPI, Flask
6. **Visualization** - Plotly, Matplotlib, Seaborn
7. **Data Formats** - NetCDF, HDF5, Zarr, Xarray
8. **Domain-Specific** - Oceanography, atmospheric science
9. **Development** - Jupyter, testing, profiling
10. **Testing** - pytest, coverage, quality tools
11. **Optimization** - Optuna, Ray, hyperparameter tuning
12. **Monitoring** - W&B, MLflow, TensorBoard

**Total Packages:** 120+ packages covering entire pipeline

---

### `requirements-dev.txt` (Development Only)

**Install after `requirements.txt`:**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Includes:**
- Documentation (Sphinx, autodoc)
- Advanced testing (Hypothesis, faker)
- Code profiling (py-spy, scalene)
- Advanced visualization
- Cloud deployment tools (AWS, GCP, Azure)
- Security scanning (bandit, safety)
- Pre-commit hooks

---

## Package Categories

### Core Computing (Mandatory)
```
numpy, pandas, scipy, scikit-learn, pyarrow
```

### Deep Learning Frameworks (Choose at least one)
```
PyTorch + Lightning  (RECOMMENDED for this project)
JAX + Flax          (Alternative, high-performance)
TensorFlow + Keras  (Full neural network library)
```

### Key ML Models
```
iTransformer        (Marine: 80.4% skill)
GraphCast           (Atmospheric: 26.7% skill)
StatsForecast       (Fallback statistical)
ARIMA, ETS          (Classical time series)
```

### Marine/Atmospheric Domain
```
utide               (Tidal analysis)
pvlib               (Solar/renewable energy)
gsw                 (Seawater properties)
cftime              (Climate time handling)
```

### Dashboard & APIs
```
Streamlit           (Interactive dashboard)
Plotly              (Interactive visualizations)
FastAPI             (High-performance API)
Flask               (Web server)
```

### Jupyter & Development
```
jupyterlab          (Notebook IDE)
ipython             (Interactive shell)
pytest              (Testing framework)
black, ruff         (Code formatting)
mypy                (Type checking)
```

---

## Installation by Use Case

### Use Case 1: Run Streamlit Dashboard Only

```bash
# Minimal install
conda install -c conda-forge python=3.11 streamlit plotly pandas numpy

# Or pip
pip install streamlit plotly pandas numpy pytorch torchvision scipy

# Run dashboard
streamlit run app_streamlit.py
```

### Use Case 2: Develop/Train Models

```bash
# Full environment
conda env create -f environment.yml
conda activate marinepred

# Or pip
pip install -r requirements.txt

# Start Jupyter
jupyter lab
```

### Use Case 3: Production Inference API

```bash
# Install main packages only
pip install -r requirements.txt --no-dev

# Or with pip-tools
pip-compile --resolver=backtracking requirements.txt

# Run FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Use Case 4: Docker Containerization

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "app_streamlit.py"]
```

Build and run:
```bash
docker build -t marine-forecasting .
docker run -p 8501:8501 marine-forecasting
```

### Use Case 5: GPU Acceleration (NVIDIA)

```bash
# Install CUDA Toolkit first (outside conda)
# https://developer.nvidia.com/cuda-11-8-0-download

# Then create environment with GPU support
conda env create -f environment.yml

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Package Highlights

### JAX (High-Performance Numerical Computing)
```
✅ Numpy-like API
✅ Automatic differentiation
✅ GPU/TPU support
✅ JIT compilation
✅ Functional programming style
```

**Why included:**
- Alternative to PyTorch for production
- Better for certain scientific computing tasks
- Excellent for optimization

### PyTorch (Deep Learning)
```
✅ Dynamic computation graphs
✅ PyTorch Lightning for structured training
✅ TorchGeometric for graph neural networks
✅ Transformers integration
```

**Models using PyTorch:**
- iTransformer (marine forecasting)
- GraphCast (atmospheric forecasting)

### TensorFlow (Full ML Stack)
```
✅ Production-ready inference
✅ Keras for high-level APIs
✅ TensorFlow Lite for mobile
✅ TensorFlow Serving for APIs
```

### Time Series Forecasting
```
StatsForecast       - Classical + ML hybrid
NeuralForecast      - PyTorch + transformers
Prophet             - Facebook's methodology
Chronos             - Foundation model approach
GluonTS             - Probabilistic forecasting
PyTorch Forecasting - Deep learning suite
```

### Domain-Specific
```
utide               - Tidal constituent analysis
pvlib               - Solar irradiance models
gsw                 - TEOS-10 seawater equations
netCDF4             - Climate data formats
xarray              - Labeled multi-dimensional data
```

---

## Version Compatibility

### Python Version
- **Recommended:** 3.11.x
- **Minimum:** 3.10
- **Tested:** 3.9, 3.10, 3.11

### Framework Versions
| Package | Version | Status |
|---------|---------|--------|
| PyTorch | 2.0+ | ✅ Tested |
| JAX | 0.4.20+ | ✅ Tested |
| TensorFlow | 2.14+ | ✅ Compatible |
| Streamlit | 1.40+ | ✅ Current |
| FastAPI | 0.110+ | ✅ Production |
| CUDA | 11.8+ | ✅ Optional |

---

## Troubleshooting

### Issue: Conda slow/hanging
```bash
# Use libmamba solver (faster)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Issue: PyTorch not finding CUDA
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install specific CUDA version
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch
```

### Issue: JAX compatibility on Windows
```bash
# Windows-specific JAX installation
pip install jax jaxlib --upgrade --find-links https://storage.googleapis.com/jax-releases/windows
```

### Issue: Memory errors during training
```bash
# Reduce batch size in config
# Or install memory-efficient version
pip install bitsandbytes  # For 8-bit optimization
```

### Issue: Import errors after installation
```bash
# Clear pip cache and reinstall
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

---

## Development Workflow

### 1. Setup Development Environment

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate marinepred

# Install dev tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### 2. Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check --fix src/
mypy src/

# Security scan
bandit -r src/
safety check
```

### 3. Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_models.py::test_itransformer
```

### 4. Notebooks

```bash
# Run Jupyter Lab
jupyter lab

# Strip outputs (for git)
nbstripout notebooks/*.ipynb

# Convert notebook to script
jupytext --to script notebook.ipynb
```

### 5. Documentation

```bash
# Generate docs
cd docs/
make html

# View documentation
open build/html/index.html
```

---

## Performance Notes

### Training Times (Approximate)
- **iTransformer:** 45-60 minutes (80 days, CPU)
- **GraphCast:** 30-45 minutes (80 days, CPU)
- **With GPU:** 5-10 minutes each

### Memory Requirements
- **Minimum:** 8 GB RAM
- **Recommended:** 16 GB RAM
- **GPU:** 8GB VRAM (NVIDIA RTX 3060+)

### Inference Latency
- **iTransformer:** 30-50 ms per batch
- **GraphCast:** 20-40 ms per batch
- **Combined system:** 50-80 ms

---

## Support & Resources

### Official Documentation
- PyTorch: https://pytorch.org/docs/stable/
- JAX: https://jax.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- FastAPI: https://fastapi.tiangolo.com/

### Community Forums
- PyTorch Discussions: https://discuss.pytorch.org/
- JAX GitHub Issues: https://github.com/google/jax/issues
- Streamlit Community: https://discuss.streamlit.io/

### Troubleshooting Checklist
- [ ] Python version correct (3.10+)
- [ ] Virtual environment activated
- [ ] All requirements installed (`pip list | grep pytorch`)
- [ ] CUDA/GPU available if needed (`nvidia-smi`)
- [ ] Disk space sufficient (10+ GB)
- [ ] Internet connection working (for model downloads)

---

## Next Steps

1. **Create Environment:** `conda env create -f environment.yml`
2. **Activate:** `conda activate marinepred`
3. **Run Dashboard:** `streamlit run app_streamlit.py`
4. **View Notebooks:** `jupyter lab notebooks/`
5. **Train Models:** See `notebooks/01_training/`
6. **Run API:** `uvicorn api.main:app --reload`

---

**Questions?** Check the YAML & Environment tab in Streamlit for more details!
