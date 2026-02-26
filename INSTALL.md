# Installation Guide

## 1. Install yProv4ML (HPCI-Lab main)

```bash
# From GitHub (recommended — always gets latest main)
pip install git+https://github.com/HPCI-Lab/yProv4ML.git

# Or if published on PyPI
pip install yprov4ml
```

> **Important**: The package name is `yprov4ml`, NOT `prov4ml`.  
> The old `prov4ml` fork (zdeniztas/yProvML) has a different API and will NOT work.

## 2. Install this plugin

```bash
pip install -e .
```

## 3. Verify the plugin is registered

```bash
python -c "
import mlflow
from mlflow.tracking._tracking_service.utils import _get_store
mlflow.set_tracking_uri('yprov+file:///tmp/test_mlruns')
store = _get_store()
print('Store:', type(store).__name__)  # Should print: YProvTrackingStore
"
```

## 4. Use it in your code

```python
import mlflow

# Activate the plugin by using the yprov+ URI prefix
mlflow.set_tracking_uri("yprov+file:///absolute/path/to/mlruns")

# Everything else is standard MLflow — provenance is automatic
with mlflow.start_run(run_name="my-run") as run:
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    
    for epoch in range(10):
        mlflow.log_metric("loss", 0.9 ** epoch, step=epoch)
        mlflow.log_metric("accuracy", 1 - 0.9 ** epoch, step=epoch)

# Provenance files are written to: data/prov/<experiment_name>/
```

## 5. Enable debug logging (optional)

```bash
YPROV_DEBUG=1 python your_script.py
```

You'll see output like:
```
[yProv Plugin] 🟢 yprov4ml v2.x.x loaded — provenance ACTIVE
[yProv Plugin] create_run: exp_id='1', run_name='my-run'
[yProv Plugin]   ✓ yprov4ml.start_run(experiment_name='Default')
[yProv Plugin] log_param: learning_rate=0.001
[yProv Plugin] log_metric: loss=1.0 step=0
[yProv Plugin] set_terminated: run_id=..., status=FINISHED
[yProv Plugin]   ✓ yprov4ml.end_run()
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `YPROV_OUT_DIR` | `data/prov` | Directory for provenance JSON output |
| `YPROV_USER_NAMESPACE` | `yProv4ML` | PROV namespace identifier |
| `YPROV_DEBUG` | `0` | Set to `1` for verbose logging |
| `YPROV_TEST_SHIM` | `0` | Force in-memory shim (tests only) |
