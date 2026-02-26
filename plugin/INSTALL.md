# Installation Guide

## Prerequisites

- Python 3.9 or higher
- `pip` and `git`

---

## Step 1 – Install `prov4ml` from source

`prov4ml` is not yet on PyPI; it must be installed directly from the upstream git fork:

```bash
pip install git+https://github.com/zdeniztas/yProvML.git@experiment_unification
```

> **Why a git install?**  
> The `experiment_unification` branch contains APIs the plugin depends on that
> have not yet been released to PyPI.  Once an official release is published,
> this step will be replaced by `pip install prov4ml`.

---

## Step 2 – Install the plugin

### Option A – Core only (just the MLflow plugin)

```bash
pip install -e .
```

### Option B – With training example dependencies (PyTorch, CodeCarbon, …)

```bash
pip install -e ".[examples]"
```

### Option C – With Streamlit dashboard dependencies

```bash
pip install -e ".[dashboard]"
```

### Option D – Full development environment

```bash
pip install -e ".[dev]"
```

Or using `make`:

```bash
make install-dev
```

---

## Step 3 – Verify the installation

```bash
make verify
```

Or manually:

```bash
# Check mlflow is importable
python -c "import mlflow; print(mlflow.__version__)"

# Check prov4ml is installed
python -c "import prov4ml; print('prov4ml OK')"

# Check plugin entry points are registered
python -c "
from importlib.metadata import entry_points
eps = [ep for ep in entry_points().get('mlflow.tracking_store', []) if 'yprov' in ep.name]
print(f'Found {len(eps)} yprov entry points: {[ep.name for ep in eps]}')
"

# Quick smoke test
python -c "import mlflow; mlflow.set_tracking_uri('yprov+file:///tmp/test'); print('Plugin activated!')"
```

---

## Platform notes

### macOS / Linux

No special steps needed. Use absolute paths in the tracking URI:

```python
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
```

### Windows

Use forward slashes or the `pathlib` helper:

```python
from pathlib import Path
mlflow.set_tracking_uri(f"yprov+file:///{Path('C:/Users/user/mlruns').as_posix()}")
```

### WSL (Windows Subsystem for Linux)

Standard Unix paths work:

```python
mlflow.set_tracking_uri("yprov+file:///home/user/mlruns")
```

---

## Virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install-dev
```
