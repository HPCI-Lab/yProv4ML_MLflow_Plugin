"""yProv4ML MLflow Plugin.

A drop-in MLflow plugin that mirrors tracking calls to yProv4ML / prov4ml,
producing W3C PROV-compliant provenance documents alongside standard MLflow runs.

Usage
-----
Set your tracking URI with the ``yprov+`` prefix to activate the plugin::

    import mlflow
    mlflow.set_tracking_uri("yprov+file:///path/to/mlruns")

Everything else is standard MLflow.
"""

__version__ = "0.0.1"

from yprov_mlflow_plugin import tracking, artifacts, prov_export
from yprov_mlflow_plugin.prov_export import export_run_to_prov

__all__ = [
    "tracking",
    "artifacts",
    "prov_export",
    "export_run_to_prov",
    "__version__",
]
