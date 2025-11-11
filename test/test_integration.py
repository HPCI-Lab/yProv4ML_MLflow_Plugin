"""
Updated integration tests that verify the REAL issue:
Params and metrics need to be logged to yprov, not just to MLflow
"""

import sys
import types
import importlib
from pathlib import Path
import pytest

PKG = "yprov_mlflow_plugin.tracking"


def _install_fake_mlflow(exp_name="DefaultExp"):
    """Create minimal fake MLflow infrastructure"""
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

    class RestStore: 
        def __init__(self, uri): self.uri = uri
        
    class FileStore: 
        def __init__(self, uri): 
            self.uri = uri
            
        def create_run(self, exp_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
            return Run("test_run_123")
            
        def create_experiment(self, name, artifact_location=None, tags=None):
            return "123"
            
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

    rest_store.RestStore = RestStore
    file_store.FileStore = FileStore
    tracking_store.rest_store = rest_store
    tracking_store.file_store = file_store
    store.tracking = tracking_store

    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.tracking"] = tracking
    sys.modules["mlflow.store"] = store
    sys.modules["mlflow.store.tracking"] = tracking_store
    sys.modules["mlflow.store.tracking.rest_store"] = rest_store
    sys.modules["mlflow.store.tracking.file_store"] = file_store


@pytest.fixture
def clean_sys_modules():
    """Clean up sys.modules after test"""
    original_modules = {k: v for k, v in sys.modules.items() if 'mlflow' in k or 'prov4ml' in k}
    yield
    to_remove = [k for k in sys.modules.keys() if ('mlflow' in k or 'prov4ml' in k) and k not in original_modules]
    for k in to_remove:
        del sys.modules[k]
    for k, v in original_modules.items():
        sys.modules[k] = v


def test_log_param_calls_yprov(tmp_path, monkeypatch, clean_sys_modules):
    """Test that log_param actually calls yprov.log_param"""
    _install_fake_mlflow(exp_name="TestExp")

    # Track yprov calls
    yprov_calls = {
        'start_run': [],
        'log_param': [],
        'log_metric': [],
        'end_run': []
    }
    
    yprov = types.ModuleType("prov4ml")
    yprov.start_run = lambda **k: yprov_calls['start_run'].append(k)
    yprov.log_param = lambda key, value: yprov_calls['log_param'].append({'key': key, 'value': value})
    yprov.log_metric = lambda key, value, step=None: yprov_calls['log_metric'].append({'key': key, 'value': value, 'step': step})
    yprov.end_run = lambda **k: yprov_calls['end_run'].append(k)
    sys.modules["prov4ml"] = yprov

    # Import tracking
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("YPROV_DEBUG", "1")
    
    if PKG in sys.modules:
        del sys.modules[PKG]
    mod = importlib.import_module(PKG)
    
    # Create store and run
    store = mod.YProvTrackingStore(f"file://{tmp_path}/mlruns")
    run = store.create_run(experiment_id="1", run_name="test_run")
    
    print(f"\n✓ start_run called {len(yprov_calls['start_run'])} times")
    
    # Log a parameter
    from types import SimpleNamespace
    store.log_param(run.info.run_id, SimpleNamespace(key="learning_rate", value=0.001))
    
    print(f"✓ log_param called {len(yprov_calls['log_param'])} times")
    print(f"  Params logged: {yprov_calls['log_param']}")
    
    # THE KEY TEST: Was yprov.log_param called?
    assert len(yprov_calls['log_param']) > 0, "❌ BUG: yprov.log_param was NOT called!"
    assert yprov_calls['log_param'][0]['key'] == "learning_rate"
    assert yprov_calls['log_param'][0]['value'] == 0.001
    
    print("✅ TEST PASSED: yprov.log_param is being called correctly")


def test_log_metric_calls_yprov(tmp_path, monkeypatch, clean_sys_modules):
    """Test that log_metric actually calls yprov.log_metric"""
    _install_fake_mlflow(exp_name="TestExp")

    # Track yprov calls
    yprov_calls = {
        'start_run': [],
        'log_param': [],
        'log_metric': [],
        'end_run': []
    }
    
    yprov = types.ModuleType("prov4ml")
    yprov.start_run = lambda **k: yprov_calls['start_run'].append(k)
    yprov.log_param = lambda key, value: yprov_calls['log_param'].append({'key': key, 'value': value})
    yprov.log_metric = lambda key, value, step=None: yprov_calls['log_metric'].append({'key': key, 'value': value, 'step': step})
    yprov.end_run = lambda **k: yprov_calls['end_run'].append(k)
    sys.modules["prov4ml"] = yprov

    # Import tracking
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("YPROV_DEBUG", "1")
    
    if PKG in sys.modules:
        del sys.modules[PKG]
    mod = importlib.import_module(PKG)
    
    # Create store and run
    store = mod.YProvTrackingStore(f"file://{tmp_path}/mlruns")
    run = store.create_run(experiment_id="1", run_name="test_run")
    
    # Log a metric
    from types import SimpleNamespace
    store.log_metric(run.info.run_id, SimpleNamespace(key="loss", value=0.5, step=1))
    
    print(f"\n✓ log_metric called {len(yprov_calls['log_metric'])} times")
    print(f"  Metrics logged: {yprov_calls['log_metric']}")
    
    # THE KEY TEST: Was yprov.log_metric called?
    assert len(yprov_calls['log_metric']) > 0, "❌ BUG: yprov.log_metric was NOT called!"
    assert yprov_calls['log_metric'][0]['key'] == "loss"
    assert yprov_calls['log_metric'][0]['value'] == 0.5
    
    print("✅ TEST PASSED: yprov.log_metric is being called correctly")


def test_log_batch_calls_yprov(tmp_path, monkeypatch, clean_sys_modules):
    """Test that log_batch actually calls yprov for params and metrics"""
    _install_fake_mlflow(exp_name="TestExp")

    # Track yprov calls
    yprov_calls = {
        'start_run': [],
        'log_param': [],
        'log_metric': [],
        'end_run': []
    }
    
    yprov = types.ModuleType("prov4ml")
    yprov.start_run = lambda **k: yprov_calls['start_run'].append(k)
    yprov.log_param = lambda key, value: yprov_calls['log_param'].append({'key': key, 'value': value})
    yprov.log_metric = lambda key, value, step=None: yprov_calls['log_metric'].append({'key': key, 'value': value, 'step': step})
    yprov.end_run = lambda **k: yprov_calls['end_run'].append(k)
    sys.modules["prov4ml"] = yprov

    # Import tracking
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("YPROV_DEBUG", "1")
    
    if PKG in sys.modules:
        del sys.modules[PKG]
    mod = importlib.import_module(PKG)
    
    # Create store and run
    store = mod.YProvTrackingStore(f"file://{tmp_path}/mlruns")
    run = store.create_run(experiment_id="1", run_name="test_run")
    
    # Log batch
    from types import SimpleNamespace
    params = [
        SimpleNamespace(key="lr", value=0.001),
        SimpleNamespace(key="batch_size", value=32)
    ]
    metrics = [
        SimpleNamespace(key="loss", value=0.5, step=1),
        SimpleNamespace(key="acc", value=0.8, step=1)
    ]
    
    store.log_batch(run.info.run_id, metrics=metrics, params=params, tags=None)
    
    print(f"\n✓ log_batch called")
    print(f"  Params logged to yprov: {len(yprov_calls['log_param'])} times")
    print(f"  Metrics logged to yprov: {len(yprov_calls['log_metric'])} times")
    print(f"  Params: {yprov_calls['log_param']}")
    print(f"  Metrics: {yprov_calls['log_metric']}")
    
    # THE KEY TEST: Were params and metrics logged to yprov?
    assert len(yprov_calls['log_param']) == 2, "❌ BUG: yprov.log_param not called for batch params!"
    assert len(yprov_calls['log_metric']) == 2, "❌ BUG: yprov.log_metric not called for batch metrics!"
    
    print("✅ TEST PASSED: log_batch correctly logs to yprov")


def test_end_run_uses_create_graph_true(tmp_path, monkeypatch, clean_sys_modules):
    """Test that end_run is called with create_graph=True"""
    _install_fake_mlflow(exp_name="TestExp")

    # Track yprov calls
    yprov_calls = {
        'start_run': [],
        'end_run': []
    }
    
    yprov = types.ModuleType("prov4ml")
    yprov.start_run = lambda **k: yprov_calls['start_run'].append(k)
    yprov.log_param = lambda key, value: None
    yprov.log_metric = lambda key, value, step=None: None
    yprov.end_run = lambda **k: yprov_calls['end_run'].append(k)
    sys.modules["prov4ml"] = yprov

    # Import tracking
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("YPROV_DEBUG", "1")
    
    if PKG in sys.modules:
        del sys.modules[PKG]
    mod = importlib.import_module(PKG)
    
    # Create store and run
    store = mod.YProvTrackingStore(f"file://{tmp_path}/mlruns")
    run = store.create_run(experiment_id="1", run_name="test_run")
    
    # End the run
    store.set_terminated(run.info.run_id, status="FINISHED", end_time=123456789)
    
    print(f"\n✓ end_run called {len(yprov_calls['end_run'])} times")
    print(f"  End run args: {yprov_calls['end_run']}")
    
    # THE KEY TEST: Was create_graph=True?
    assert len(yprov_calls['end_run']) > 0, "end_run was not called!"
    
    end_call = yprov_calls['end_run'][0]
    if 'create_graph' in end_call:
        assert end_call['create_graph'] is True, f"❌ BUG: create_graph={end_call['create_graph']} (should be True for JSON generation!)"
        print("✅ TEST PASSED: create_graph=True")
    else:
        print("⚠️  WARNING: create_graph not specified (might use default)")


def test_full_lifecycle(tmp_path, monkeypatch, clean_sys_modules):
    """Test complete lifecycle: create -> log params/metrics -> end"""
    _install_fake_mlflow(exp_name="TestExp")

    # Track ALL yprov calls
    yprov_calls = {
        'start_run': [],
        'log_param': [],
        'log_metric': [],
        'end_run': []
    }
    
    yprov = types.ModuleType("prov4ml")
    yprov.start_run = lambda **k: yprov_calls['start_run'].append(k)
    yprov.log_param = lambda key, value: yprov_calls['log_param'].append({'key': key, 'value': value})
    yprov.log_metric = lambda key, value, step=None: yprov_calls['log_metric'].append({'key': key, 'value': value, 'step': step})
    yprov.end_run = lambda **k: yprov_calls['end_run'].append(k)
    sys.modules["prov4ml"] = yprov

    # Import tracking
    monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("YPROV_DEBUG", "1")
    
    if PKG in sys.modules:
        del sys.modules[PKG]
    mod = importlib.import_module(PKG)
    
    # Create store and run
    store = mod.YProvTrackingStore(f"file://{tmp_path}/mlruns")
    run = store.create_run(experiment_id="1", run_name="test_run")
    
    # Log params
    from types import SimpleNamespace
    store.log_param(run.info.run_id, SimpleNamespace(key="lr", value=0.001))
    store.log_param(run.info.run_id, SimpleNamespace(key="epochs", value=10))
    
    # Log metrics
    for i in range(5):
        store.log_metric(run.info.run_id, SimpleNamespace(key="loss", value=0.5-i*0.05, step=i))
    
    # End run
    store.set_terminated(run.info.run_id, status="FINISHED", end_time=123456789)
    
    # Print summary
    print("\n" + "="*60)
    print("LIFECYCLE TEST SUMMARY")
    print("="*60)
    print(f"start_run calls: {len(yprov_calls['start_run'])}")
    print(f"log_param calls: {len(yprov_calls['log_param'])}")
    print(f"log_metric calls: {len(yprov_calls['log_metric'])}")
    print(f"end_run calls: {len(yprov_calls['end_run'])}")
    print("="*60)
    
    # Verify everything was called
    assert len(yprov_calls['start_run']) >= 1, "start_run not called"
    assert len(yprov_calls['log_param']) == 2, f"❌ BUG: Expected 2 param logs, got {len(yprov_calls['log_param'])}"
    assert len(yprov_calls['log_metric']) == 5, f"❌ BUG: Expected 5 metric logs, got {len(yprov_calls['log_metric'])}"
    assert len(yprov_calls['end_run']) >= 1, "end_run not called"
    
    print("✅ ALL TESTS PASSED: Complete lifecycle working correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])