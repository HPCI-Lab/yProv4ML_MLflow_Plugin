# yProv4ML_MLflow_Plugin
```
yProv4ML_MLflow_Plugin/
├── yprov_mlflow_plugin/       # Plugin source code
│   ├── tracking.py             # TrackingStore implementation
│   ├── artifacts.py            # ArtifactRepository implementation
│   └── prov_export.py          # PROV document generator
├── examples/                   # Example training scripts
│   ├── demo.py                 # CIFAR-10 example
│   ├── mnist_mlflow_demo.py    # MNIST with schedulers
│   └── run_batch.py            # Grid/random search
├── test/                       # Test suite
└── pyproject.toml              # Plugin entry points

```

# yProv4ML MLflow Plugin

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3+-green.svg)](https://mlflow.org/)

A powerful MLflow plugin that seamlessly integrates [W3C PROV](https://www.w3.org/TR/prov-overview/)-compliant provenance tracking with sustainability metrics for machine learning experiments built on the [yProv4ML (prov4ml)](https://github.com/zdeniztas/yProvML) 

## 🎯 Key Features

- **🔄 Zero-Code Integration**: Drop-in replacement for standard MLflow tracking
- **📊 W3C PROV Compliance**: Automatic generation of standardized provenance graphs
- **🌱 Sustainability Tracking**: Built-in CodeCarbon integration for carbon footprint monitoring
- **📈 Full MLflow Compatibility**: Works with existing MLflow workflows, UI, and tools
- **🔍 Comprehensive Metadata**: Captures hyperparameters, metrics, artifacts, and system info
- **📁 Dual Storage**: Maintains both MLflow runs and PROV JSON documents



## 🏗️ Architecture Overview

The plugin implements two main MLflow extension points:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your ML Training Script                   │
│                  (uses standard MLflow API)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              yProv4ML MLflow Plugin Layer                    │
│  ┌──────────────────────┐    ┌─────────────────────────┐   │
│  │  YProvTrackingStore  │    │  YProvArtifactRepo      │   │
│  │  (tracking.py)       │    │  (artifacts.py)         │   │
│  └──────────┬───────────┘    └───────────┬─────────────┘   │
│             │                             │                  │
│             ▼                             ▼                  │
│  ┌──────────────────────┐    ┌─────────────────────────┐   │
│  │  Standard MLflow     │    │  Standard MLflow        │   │
│  │  FileStore/RestStore │    │  LocalArtifactRepo      │   │
│  └──────────┬───────────┘    └───────────┬─────────────┘   │
└─────────────┼─────────────────────────────┼─────────────────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐            ┌──────────────────┐
     │  mlruns/       │            │  mlartifacts/    │
     │  (MLflow data) │            │  (artifacts)     │
     └────────────────┘            └──────────────────┘
              │                             │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  prov4ml Integration │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  data/prov/          │
              │  (PROV JSON files)   │
              └──────────────────────┘
```

### Component Breakdown

1. **YProvTrackingStore** (`tracking.py`)
   - Intercepts MLflow tracking calls (params, metrics, runs)
   - Delegates to standard MLflow storage (FileStore or RestStore)
   - Simultaneously logs to prov4ml for W3C PROV generation
   - Generates PROV JSON on run completion

2. **YProvArtifactRepo** (`artifacts.py`)
   - Handles artifact logging (models, configs, plots)
   - Mirrors artifacts to both MLflow and PROV storage
   - Maintains artifact provenance metadata

3. **PROV Export** (`prov_export.py`)
   - Converts MLflow run data to W3C PROV format
   - Extracts metadata from activities and entities
   - Generates JSON-LD compatible provenance documents

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for installing from source)

### Installation (Recommended)

```bash
# Clone the repository
conda create -n mlflow python=3.9

git clone https://github.com/yourusername/yProv4ML_MLflow_Plugin.git
cd yProv4ML_MLflow_Plugin

# Install the plugin in development mode
pip install -e .
```

#### **Install Requirements** 
```bash
pip install -r requirements.txt
```


### Verify Installation

```bash
# Check that the plugin is registered
python -c "import mlflow; print(mlflow.__version__)"

# Verify prov4ml is installed
python -c "import prov4ml; print('prov4ml installed successfully')"

# Check plugin entry points
python -c "from importlib.metadata import entry_points; eps = [ep for ep in entry_points().get('mlflow.tracking_store', []) if 'yprov' in ep.name]; print(f'Found {len(eps)} yprov entry points')"

# Test basic functionality
python -c "import mlflow; mlflow.set_tracking_uri('yprov+file:///tmp/test'); print('✅ Plugin activated successfully')"
```

### Platform-Specific Notes

#### **Windows**
```bash
# Use forward slashes or Path objects
mlflow.set_tracking_uri("yprov+file:///C:/Users/user/mlruns")

# Or use pathlib
from pathlib import Path
mlflow.set_tracking_uri(f"yprov+file:///{Path('C:/Users/user/mlruns').as_posix()}")
```

#### **WSL / Linux**
```bash
# Standard Unix paths work directly
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
```

#### **macOS**
```bash
# Standard Unix paths
mlflow.set_tracking_uri("yprov+file:///Users/user/mlruns")
```

## ⚡ Quick Start

The plugin activates automatically when you use the `yprov+` URI scheme:

```python
import mlflow

# 🔑 CRITICAL: Use yprov+ prefix to activate the plugin
mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")

# Standard MLflow code - no changes needed!
mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="example_run"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    
    # Log metrics
    for epoch in range(5):
        mlflow.log_metric("loss", 0.5 - epoch * 0.05, step=epoch)
        mlflow.log_metric("accuracy", 0.7 + epoch * 0.05, step=epoch)
    
    # Log artifacts
    mlflow.log_artifact("model.pt", artifact_path="models")

# ✅ PROV JSON automatically generated in data/prov/my_experiment/
```


## 🔧 How It Works

### Plugin Activation

The plugin registers itself via setuptools entry points defined in `pyproject.toml`:

```toml
[project.entry-points."mlflow.tracking_store"]
"yprov+http" = "yprov_mlflow_plugin.tracking:YProvTrackingStore"
"yprov+file" = "yprov_mlflow_plugin.tracking:YProvTrackingStore"

[project.entry-points."mlflow.artifact_repository"]
"yprov+file" = "yprov_mlflow_plugin.artifacts:YProvArtifactRepo"
"yprov+http" = "yprov_mlflow_plugin.artifacts:YProvArtifactRepo"
```

When MLflow sees a URI starting with `yprov+`, it automatically routes calls through the plugin.

### Data Flow

#### 1. Run Creation (`mlflow.start_run()`)

```python
# User code
with mlflow.start_run(run_name="my_run"):
    # ...

# What happens internally:
# 1. YProvTrackingStore.create_run() is called
# 2. Creates standard MLflow run in mlruns/
# 3. Calls prov4ml.start_run() with:
#    - experiment_name
#    - provenance_save_dir (data/prov/experiment_name/)
#    - metrics_file_type (CSV or NetCDF)
# 4. Returns MLflow Run object
```

#### 2. Parameter Logging (`mlflow.log_param()`)

```python
# User code
mlflow.log_param("learning_rate", 0.001)

# What happens internally:
# 1. YProvTrackingStore.log_param() is called
# 2. Logs to MLflow: mlruns/exp_id/run_id/params/learning_rate
# 3. Calls prov4ml.log_param("learning_rate", 0.001)
# 4. Stores in PROV context for later JSON generation
```

#### 3. Metric Logging (`mlflow.log_metric()`)

```python
# User code
mlflow.log_metric("loss", 0.5, step=1)

# What happens internally:
# 1. YProvTrackingStore.log_metric() is called
# 2. Logs to MLflow: mlruns/exp_id/run_id/metrics/loss
# 3. Calls prov4ml.log_metric("loss", 0.5, step=1)
# 4. Writes to CSV/NetCDF: data/prov/experiment/metrics/loss.csv
```

#### 4. Artifact Logging (`mlflow.log_artifact()`)

```python
# User code
mlflow.log_artifact("model.pt", artifact_path="models")

# What happens internally:
# 1. YProvArtifactRepo.log_artifact() is called
# 2. Copies to MLflow: mlartifacts/run_id/models/model.pt
# 3. Calls prov4ml.log_artifact() to record in PROV
# 4. Stores artifact metadata for provenance graph
```

#### 5. Run Completion (`mlflow.end_run()` or context exit)

```python
# User code (explicit or implicit)
mlflow.end_run()
# or
# with mlflow.start_run():
#     ...  # auto-closes on exit

# What happens internally:
# 1. YProvTrackingStore.set_terminated() is called
# 2. Calls prov4ml.end_run(create_graph=True)
# 3. Generates PROV JSON in data/prov/experiment/prov_*.json
# 4. Updates MLflow run status


## 📚 Usage Examples

### Example 1: Simple Classification

```python
import mlflow
from pathlib import Path

# Setup
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
mlflow.set_experiment("iris_classification")

with mlflow.start_run(run_name="logistic_regression"):
    # Parameters
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 100)
    
    # Training (your code here)
    # ...
    
    # Metrics
    mlflow.log_metric("train_accuracy", 0.95)
    mlflow.log_metric("test_accuracy", 0.93)
    mlflow.log_metric("f1_score", 0.94)
    
    # Save model
    mlflow.log_artifact("model.pkl", artifact_path="models")

# PROV JSON available at: data/prov/iris_classification/prov_*.json
```

## 🔍 Configuration

### Environment Variables

Control plugin behavior via environment variables:

```bash
# PROV output directory
export YPROV_OUT_DIR="/path/to/prov/output"
# Default: data/prov

# Enable debug logging
export YPROV_DEBUG="1"
# Default: 0

# User namespace for PROV documents
export YPROV_USER_NAMESPACE="MyOrganization"
# Default: yProv4ML

# Collect all system processes (more overhead)
export YPROV_COLLECT_ALL="1"
# Default: 0

# Unify experiments in single PROV document
export YPROV_UNIFY="1"
# Default: 0

# Custom merged PROV file path (when YPROV_UNIFY=1)
export YPROV_MERGED_PATH="/path/to/merged_prov.json"
# Default: data/prov/merged_provenance.json
```

### Python Configuration

```python
import os
import mlflow

# Set configuration before mlflow.set_tracking_uri()
os.environ["YPROV_OUT_DIR"] = "/custom/prov/dir"
os.environ["YPROV_DEBUG"] = "1"
os.environ["YPROV_USER_NAMESPACE"] = "MyProject"

# Then activate plugin
mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")
```

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/yProv4ML_MLflow_Plugin.git
cd yProv4ML_MLflow_Plugin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest test/ -v
```


### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Areas for Contribution

- 🐛 **Bug fixes**: See [Issues](https://github.com/yourusername/yProv4ML_MLflow_Plugin/issues)
- 📚 **Documentation**: Improve guides, add examples
- ✨ **Features**: 
  - Additional dashboard visualizations
  - More export formats (CSV, Parquet, etc.)
  - Integration with other tracking tools
  - Performance optimizations
- 🧪 **Testing**: Increase test coverage, add edge cases


## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) - Open source ML lifecycle platform
- [yProv4ML (prov4ml)](https://github.com/zdeniztas/yProvML) - W3C PROV provenance tracking
- [CodeCarbon](https://codecarbon.io/) - Carbon emissions tracking
- [W3C PROV](https://www.w3.org/TR/prov-overview/) - Provenance standard


---
