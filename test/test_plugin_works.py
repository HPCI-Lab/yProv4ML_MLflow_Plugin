import sys, types, importlib
from pathlib import Path
import pytest

PKG = "yprov_mlflow_plugin.tracking"

@pytest.fixture
def clean_sys_modules():
    """Clean up sys.modules after test to avoid mock pollution."""
    original_modules = {k: v for k, v in sys.modules.items() if 'mlflow' in k}
    yield
    # Remove any mlflow modules added during test
    to_remove = [k for k in sys.modules.keys() if 'mlflow' in k and k not in original_modules]
    for k in to_remove:
        del sys.modules[k]
    # Restore original modules  
    for k, v in original_modules.items():
        sys.modules[k] = v


def _install_fake_mlflow_for_update(exp_name="DefaultExp"):
    # Minimal MlflowClient + structures for update_run_info path
    mlflow = types.ModuleType("mlflow")

    class RunInfo: 
        def __init__(self, run_id, exp_id="1"): 
            self.run_id = run_id
            self.experiment_id = exp_id
            self.artifact_uri = "file:///tmp/artifacts"
            
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

    class MlflowClient:
        def __init__(self): pass
        def get_run(self, run_id): return Run(run_id)
        def get_experiment(self, exp_id): return Experiment(exp_name)
        def list_artifacts(self, run_id, path=""): return []

    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = MlflowClient

    store = types.ModuleType("mlflow.store")
    tracking_store = types.ModuleType("mlflow.store.tracking")
    rest_store = types.ModuleType("mlflow.store.tracking.rest_store")
    file_store = types.ModuleType("mlflow.store.tracking.file_store")

    class Rest: 
        def __init__(self, uri): self.uri = uri
        
    class File: 
        def __init__(self, uri): 
            self.uri = uri
            
        def create_experiment(self, name, artifact_location=None, tags=None):
            return "123"
            
        def get_experiment(self, exp_id):
            return Experiment(exp_name)
            
        def get_experiment_by_name(self, name):
            return Experiment(name)
            
        def get_run(self, run_id):
            return Run(run_id)
            
        def update_run_info(self, run_id, run_status, end_time, **kwargs):
            return ("delegate.update_run_info", run_id, run_status, end_time)
            
        def set_terminated(self, run_id, status, end_time):
            return ("delegate.set_terminated", run_id, status, end_time)
            
        def log_param(self, run_id, param):
            return ("delegate.log_param", run_id, param)
            
        def log_metric(self, run_id, metric):
            return ("delegate.log_metric", run_id, metric)
            
        def log_batch(self, run_id, metrics, params, tags):
            return ("delegate.log_batch", run_id, metrics, params, tags)

    rest_store.RestStore = Rest
    file_store.FileStore = File
    tracking_store.rest_store = rest_store
    tracking_store.file_store = file_store
    store.tracking = tracking_store

    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.tracking"] = tracking
    sys.modules["mlflow.store"] = store
    sys.modules["mlflow.store.tracking"] = tracking_store
    sys.modules["mlflow.store.tracking.rest_store"] = rest_store
    sys.modules["mlflow.store.tracking.file_store"] = file_store

def test_update_run_info_triggers_export_when_terminal_status(tmp_path, monkeypatch, clean_sys_modules):
    _install_fake_mlflow_for_update(exp_name="DemoExperiment")

    # Install fake yprov (no-ops)
    yprov = types.ModuleType("prov4ml")
    yprov.start_run = lambda **k: None
    yprov.log_param = lambda *a, **k: None
    yprov.log_metric = lambda *a, **k: None
    yprov.end_run = lambda **k: None
    sys.modules["prov4ml"] = yprov

    # Prepare export recorder
    called = {}
    def fake_export(run_id, out_dir, client=None):
        called["args"] = (run_id, Path(out_dir), client)
        # Return a fake path
        return out_dir / f"{run_id}.json"

    # Patch the prov_export module before importing tracking
    prov_export = types.ModuleType("yprov_mlflow_plugin.prov_export")
    prov_export.export_run_to_prov = fake_export
    sys.modules["yprov_mlflow_plugin.prov_export"] = prov_export

    # Import module
    mod = importlib.reload(importlib.import_module(PKG))

    # Ensure deterministic out dir via env
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    store = mod.YProvTrackingStore("file:///tmp/mlruns")

    # Call update_run_info with terminal status and end_time
    res = store.update_run_info("12345", run_status=3, end_time=1730000000000)
    
    # Verify delegate was called
    assert res[0] == "delegate.update_run_info"
    assert res[1] == "12345"

    # Verify export was called and path includes experiment name
    assert "args" in called, "export_run_to_prov was not called!"
    assert called["args"][0] == "12345"
    assert called["args"][1].as_posix() == (tmp_path / "DemoExperiment").as_posix()