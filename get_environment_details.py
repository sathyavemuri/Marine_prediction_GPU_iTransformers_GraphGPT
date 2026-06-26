"""Generate comprehensive environment and configuration details."""

import subprocess
import json
import sys
from pathlib import Path
import platform

print("=" * 100)
print("ENVIRONMENT & CONFIGURATION DETAILS")
print("=" * 100)

# System info
print("\n" + "-" * 100)
print("SYSTEM INFORMATION")
print("-" * 100)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")

# Conda environment
print("\n" + "-" * 100)
print("CONDA ENVIRONMENT")
print("-" * 100)
result = subprocess.run(['conda', 'info', '--json'], capture_output=True, text=True)
if result.returncode == 0:
    info = json.loads(result.stdout)
    print(f"Conda Version: {info.get('conda_version', 'N/A')}")
    print(f"Active Environment: marinepred")
    print(f"Environment Location: {info.get('envs', [None])[0]}")

# Check GPU availability
print("\n" + "-" * 100)
print("GPU/HARDWARE AVAILABILITY")
print("-" * 100)
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA: NOT AVAILABLE - Using CPU")
except ImportError:
    print("PyTorch not available")

# All installed packages
print("\n" + "-" * 100)
print("INSTALLED PACKAGES (conda list)")
print("-" * 100)
result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)

# Config file details
print("\n" + "=" * 100)
print("PROJECT CONFIGURATION FILES")
print("=" * 100)

config_file = Path('config/phase3_graphcast.yaml')
if config_file.exists():
    print(f"\n{'-' * 100}")
    print(f"FILE: {config_file}")
    print(f"{'-' * 100}")
    with open(config_file, 'r') as f:
        print(f.read())

# Requirements files
req_files = [
    'requirements_dashboard.txt',
    'portland_itransformer/requirements.txt',
    'marine_local_mtgnn/requirements.txt'
]

for req_file in req_files:
    req_path = Path(req_file)
    if req_path.exists():
        print(f"\n{'-' * 100}")
        print(f"FILE: {req_file}")
        print(f"{'-' * 100}")
        with open(req_path, 'r') as f:
            print(f.read())
