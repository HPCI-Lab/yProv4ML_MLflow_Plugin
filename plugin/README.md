# yProv4ML MLflow Plugin

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3+-green.svg)](https://mlflow.org/)

A drop-in MLflow plugin that mirrors tracking calls to [yProv4ML (prov4ml)](https://github.com/zdeniztas/yProvML), producing **W3C PROV-compliant** provenance documents alongside standard MLflow runs.

---

## 🎯 Key Features

- **Zero-code integration** — use the `yprov+` URI prefix; all existing MLflow code stays unchanged.
- **W3C PROV compliance** — automatic generation of standardised provenance graphs.
- **Sustainability tracking** — optional CodeCarbon integration for carbon-footprint monitoring.
- **Full MLflow compatibility** — works with the MLflow UI, CLI, and tracking server.
- **Dual storage** — maintains both standard `mlruns/` data and PROV JSON documents.
- **Decision-support dashboard** — interactive Streamlit app for experiment analysis and recommendations.

---

## 🚀 Installation

See **[INSTALL.md](INSTALL.md)** for full instructions.

**Quick start:**

```bash
# 1. Install prov4ml from source (required – not yet on PyPI)
pip install git+https://github.com/zdeniztas/yProvML.git@experiment_unification

# 2. Install the plugin
pip install -e .

# 3. (Optional) training examples or dashboard extras
pip install -e ".[examples]"   # PyTorch, CodeCarbon, psutil
pip install -e ".[dashboard]"  # Streamlit, plotly, shap
pip install -e ".[dev]"        # everything for development
```

Or with `make`:

```bash
make install        # core only
make install-dev    # full dev environment
```

---

## ⚡ Quick Start

```python
import mlflow

# 🔑 Use yprov+ prefix to activate the plugin
mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")
mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="example_run"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    for epoch in range(5):
        mlflow.log_metric("loss",     0.5 - epoch * 0.05, step=epoch)
        mlflow.log_metric("accuracy", 0.7 + epoch * 0.05, step=epoch)

    mlflow.log_artifact("model.pt", artifact_path="models")

# ✅ PROV JSON auto-generated in data/prov/my_experiment/
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Your ML Training Script         │
│       (standard MLflow API)             │
└──────────────────┬──────────────────────┘
                   │  yprov+file:// URI
                   ▼
┌─────────────────────────────────────────┐
│       yProv4ML MLflow Plugin Layer       │
│  YProvTrackingStore  │  YProvArtifactRepo│
└──────┬───────────────┼───────────────────┘
       │               │
       ▼               ▼
  mlruns/          mlartifacts/        data/prov/
  (MLflow)         (artifacts)         (PROV JSON)
```

### Component breakdown

| File | Role |
|---|---|
| `yprov_mlflow_plugin/tracking.py` | `AbstractStore` subclass — intercepts params, metrics, run lifecycle |
| `yprov_mlflow_plugin/artifacts.py` | `ArtifactRepository` subclass — mirrors artifact logs to prov4ml |
| `yprov_mlflow_plugin/prov_export.py` | Standalone PROV JSON exporter for finished runs |

---

## 📚 Usage Examples

### Basic training run

```python
import mlflow
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
mlflow.set_experiment("iris_classification")

with mlflow.start_run(run_name="logistic_regression"):
    mlflow.log_params({"model_type": "logistic_regression", "max_iter": 100})
    mlflow.log_metrics({"train_accuracy": 0.95, "test_accuracy": 0.93})
    mlflow.log_artifact("model.pkl", artifact_path="models")
```

### Re-export PROV for a historical run

```python
from yprov_mlflow_plugin.prov_export import export_run_to_prov

path = export_run_to_prov(
    run_id="<your-run-id>",
    tracking_uri="yprov+file:///path/to/mlruns",
    out_dir="data/prov",
)
print(f"PROV JSON written to: {path}")
```

### Batch hyperparameter sweep

```bash
# Grid search
python examples/run_batch.py \
    --script examples/demo.py \
    --mode grid \
    --grid batch_size=8,16,32,64 \
    --grid lr=1e-3,1e-4,1e-5 \
    --jobs 4

# Random search
python examples/run_batch.py \
    --script examples/demo.py \
    --mode random \
    --n 50 \
    --rand batch_size=choice(8,16,32,64) \
    --rand lr=log10(-5,-2) \
    --jobs 4
```

---

## 🔧 Configuration

Control plugin behaviour via environment variables:

| Variable | Default | Description |
|---|---|---|
| `YPROV_OUT_DIR` | `data/prov` | Root directory for PROV JSON output |
| `YPROV_DEBUG` | `0` | Set to `1` for verbose plugin logging |
| `YPROV_USER_NAMESPACE` | `yProv4ML` | PROV namespace prefix |
| `YPROV_COLLECT_ALL` | `0` | Collect all system processes (more overhead) |
| `YPROV_UNIFY` | `0` | Merge all runs into a single PROV document |
| `YPROV_MERGED_PATH` | `data/prov/merged_provenance.json` | Path for the merged file (when `YPROV_UNIFY=1`) |

```python
import os, mlflow

os.environ["YPROV_OUT_DIR"]        = "/custom/prov"
os.environ["YPROV_DEBUG"]          = "1"
os.environ["YPROV_USER_NAMESPACE"] = "MyProject"

mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")
```

---

## 🎨 Streamlit Dashboard

```bash
# 1. Export PROV data to CSV
python src/prov_to_csv.py --root data/prov --out-dir data/unified

# 2. Launch dashboard
streamlit run src/decision_app.py
```

Upload the generated CSV via the sidebar to access:

- **Solution-space heatmaps** — parameter × metric exploration
- **Pareto frontier** — accuracy vs. sustainability trade-offs
- **Intelligent recommendations** — EXPLOIT / EXPLORE / BALANCE / INTERPOLATE
- **SHAP-based clustering** — understand which parameters drive experiment similarity

---

## 🛠️ Developer Workflow

```bash
make install-dev    # install everything
make test           # run test suite
make test-cov       # run with coverage
make lint           # ruff lint
make fmt            # ruff format
make demo-dry       # 1-epoch smoke test
make verify         # check plugin is registered correctly
make help           # list all targets
```

---

## 📁 Project Structure

```
yProv4ML_MLflow_Plugin/
├── yprov_mlflow_plugin/       # Plugin package
│   ├── __init__.py
│   ├── tracking.py            # YProvTrackingStore
│   ├── artifacts.py           # YProvArtifactRepo
│   └── prov_export.py         # PROV JSON exporter
├── examples/                  # Example training scripts
│   ├── demo.py                # CIFAR-10 / toy-metrics demo
│   ├── mnist_mlflow_demo.py   # MNIST with schedulers
│   └── run_batch.py           # Grid / random hyperparameter search
├── src/                       # Utilities
│   ├── prov_to_csv.py         # Convert PROV JSON → CSV
│   ├── import_prov_to_mlflow.py
│   └── decision_app.py        # Streamlit dashboard
├── test/                      # Test suite
│   ├── conftest.py            # Shared fixtures
│   ├── test_store.py
│   ├── test_artifacts.py
│   ├── test_integration.py
│   └── test_yprov_logging.py
├── pyproject.toml             # Package metadata + optional extras
├── requirements.txt           # Core deps (pip -r alternative)
├── requirements-examples.txt  # Example training deps
├── requirements-dashboard.txt # Dashboard deps
├── requirements-dev.txt       # Full dev environment
├── INSTALL.md                 # Detailed installation guide
└── Makefile                   # Common developer tasks
```

---

## 🙏 Acknowledgements

- [MLflow](https://mlflow.org/) — Open-source ML lifecycle platform
- [yProv4ML / prov4ml](https://github.com/zdeniztas/yProvML) — W3C PROV provenance tracking
- [CodeCarbon](https://codecarbon.io/) — Carbon emissions tracking
- [W3C PROV](https://www.w3.org/TR/prov-overview/) — Provenance standard
