
import sys, types, importlib
from pathlib import Path

PKG = "yprov_mlflow_plugin.tracking"

def _install_fake_mlflow_for_update(exp_name="DefaultExp"):
    # Minimal MlflowClient + structures for update_run_info path
    mlflow = types.ModuleType("mlflow")

    class RunInfo: 
        def __init__(self, run_id, exp_id="1"): 
            self.run_id = run_id; self.experiment_id = exp_id
    class Run: 
        def __init__(self, run_id): 
            self.info = RunInfo(run_id)
            self.data = types.SimpleNamespace(tags={})
    class Experiment: 
        def __init__(self, name): self.name = name

    class MlflowClient:
        def __init__(self): pass
        def get_run(self, run_id): return Run(run_id)
        def get_experiment(self, exp_id): return Experiment(exp_name)

    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = MlflowClient

    store = types.ModuleType("mlflow.store")
    tracking_store = types.ModuleType("mlflow.store.tracking")
    rest_store = types.ModuleType("mlflow.store.tracking.rest_store")
    file_store = types.ModuleType("mlflow.store.tracking.file_store")

    class Rest: 
        def __init__(self, uri): self.uri = uri
    class File: 
        def __init__(self, uri): self.uri = uri
        def update_run_info(self, *a, **k): return ("delegate.update_run_info", a, k)
        def set_terminated(self, *a, **k): return ("delegate.set_terminated", a, k)
        def log_param(self, *a, **k): return ("delegate.log_param", a, k)
        def log_metric(self, *a, **k): return ("delegate.log_metric", a, k)
        def log_batch(self, *a, **k): return ("delegate.log_batch", a, k)
        def get_run(self, run_id): return types.SimpleNamespace(info=types.SimpleNamespace(artifact_uri="file:///tmp"))

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

def test_update_run_info_triggers_export_when_end_time_set(tmp_path, monkeypatch):
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

    # Import module and patch export function
    mod = importlib.reload(importlib.import_module(PKG))
    mod.export_run_to_prov = fake_export

    # ensure deterministic out dir via env
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    store = mod.YProvTrackingStore("file:///tmp/mlruns")

    # call update_run_info with end_time set
    res = store.update_run_info("12345", "FINISHED", end_time=1730000000000)
    # delegate returns a tuple as per Fake FileStore impl
    assert res[0] == "delegate.update_run_info"

    # verify export called and path includes experiment name
    assert called["args"][0] == "12345"
    assert called["args"][1].as_posix() == (tmp_path / "DemoExperiment").as_posix()
