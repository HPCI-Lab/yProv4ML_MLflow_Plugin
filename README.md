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
├── src/                        # Utilities
│   ├── prov_to_csv.py          # Convert PROV to CSV
│   └── decision_app.py         # Streamlit dashboard
├── test/                       # Test suite
└── pyproject.toml              # Plugin entry points

```