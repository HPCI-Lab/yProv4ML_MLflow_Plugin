"""
conftest.py – Shared pytest fixtures and configuration.

Located at the test/ level so all test modules can use these fixtures.
"""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make the project root importable even without `pip install -e .`
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers used by multiple test modules
# ---------------------------------------------------------------------------

def make_fake_mlflow(exp_name: str = "DefaultExp"):
    """
    Build a minimal fake mlflow module tree and inject it into sys.modules.

    Returns the top-level fake ``mlflow`` module.
    """
    class RunInfo:
        def __init__(self, run_id, exp_id="1"):
            self.run_id = run_id
            self.experiment_id = exp_id
            self.artifact_uri = "file:///tmp/artifacts"
            self.status = "RUNNING"
            self.start_time = None
            self.end_time = None
            self.run_name = None
            self.user_id = "tester"

    class RunData:
        def __init__(self):
            self.tags = {}
            self.params = {}
            self.metrics = {}

    class Run:
        def __init__(self, run_id):
            self.info = RunInfo(run_id)
            self.data = RunData()

    class Experiment:
        def __init__(self, name):
            self.name = name
            self.experiment_id = "1"

    class FileStore:
        def __init__(self, uri):
            self.uri = uri

        def create_run(self, exp_id, user_id=None, start_time=None,
                       tags=None, run_name=None, **kwargs):
            return Run("test_run_123")

        def create_experiment(self, name, artifact_location=None, tags=None):
            return "1"

        def get_experiment(self, exp_id):
            return Experiment(exp_name)

        def get_experiment_by_name(self, name):
            return Experiment(name)

        def get_run(self, run_id):
            return Run(run_id)

        def update_run_info(self, run_id, run_status, end_time, **kwargs):
            return Run(run_id)

        def set_terminated(self, run_id, status, end_time):
            return Run(run_id)

        def log_param(self, run_id, param):
            return None

        def log_metric(self, run_id, metric):
            return None

        def log_batch(self, run_id, metrics, params, tags):
            return None

        def search_runs(self, experiment_ids, filter_string,
                        run_view_type, max_results, order_by, page_token):
            return []

        def list_run_infos(self, experiment_id, run_view_type,
                           max_results, order_by, page_token):
            return []

    class RestStore:
        def __init__(self, uri):
            self.uri = uri

    # Build module tree
    mlflow_mod    = types.ModuleType("mlflow")
    store_mod     = types.ModuleType("mlflow.store")
    tracking_mod  = types.ModuleType("mlflow.store.tracking")
    rest_mod      = types.ModuleType("mlflow.store.tracking.rest_store")
    file_mod      = types.ModuleType("mlflow.store.tracking.file_store")

    rest_mod.RestStore   = RestStore
    file_mod.FileStore   = FileStore
    tracking_mod.rest_store = rest_mod
    tracking_mod.file_store = file_mod
    store_mod.tracking   = tracking_mod
    mlflow_mod.store     = store_mod

    sys.modules.update({
        "mlflow":                              mlflow_mod,
        "mlflow.store":                        store_mod,
        "mlflow.store.tracking":               tracking_mod,
        "mlflow.store.tracking.rest_store":    rest_mod,
        "mlflow.store.tracking.file_store":    file_mod,
    })
    return mlflow_mod


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_prov_dir(tmp_path, monkeypatch):
    """Set YPROV_OUT_DIR to a fresh tmp directory and return it."""
    prov_dir = tmp_path / "prov"
    prov_dir.mkdir()
    monkeypatch.setenv("YPROV_OUT_DIR", str(prov_dir))
    return prov_dir


@pytest.fixture()
def fake_yprov():
    """
    Inject a fake prov4ml module that records all calls.
    Returns a dict of call logs: start_run, log_param, log_metric, end_run.
    """
    calls = {"start_run": [], "log_param": [], "log_metric": [], "end_run": []}
    mod = types.ModuleType("prov4ml")
    mod.start_run  = lambda **kw: calls["start_run"].append(kw)
    mod.log_param  = lambda k, v: calls["log_param"].append({"key": k, "value": v})
    mod.log_metric = lambda k, v, step=None, **kw: calls["log_metric"].append(
        {"key": k, "value": v, "step": step}
    )
    mod.end_run    = lambda **kw: calls["end_run"].append(kw)
    mod.log_artifact = lambda path, artifact_path=None: None
    sys.modules["prov4ml"] = mod
    yield calls
    sys.modules.pop("prov4ml", None)


@pytest.fixture()
def tracking_store(tmp_path, tmp_prov_dir, fake_yprov, monkeypatch):
    """
    A fully configured YProvTrackingStore backed by fake mlflow + fake yprov.
    Yields (store, yprov_calls).
    """
    make_fake_mlflow()
    monkeypatch.setenv("YPROV_DEBUG", "0")

    mod_name = "yprov_mlflow_plugin.tracking"
    sys.modules.pop(mod_name, None)
    mod = importlib.import_module(mod_name)
    store = mod.YProvTrackingStore(f"file://{tmp_path}/mlruns")
    yield store, fake_yprov
