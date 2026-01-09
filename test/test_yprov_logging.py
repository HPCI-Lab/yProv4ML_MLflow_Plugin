"""
Working test to verify the bug and fix in your actual project.
Place this in your test/ directory and run with pytest.
"""
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import importlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def fresh_tracking_module():
    """Force a clean import of yprov_mlflow_plugin.tracking so sys.modules is correct."""
    sys.modules.pop("yprov_mlflow_plugin.tracking", None)
    mod = importlib.import_module("yprov_mlflow_plugin.tracking")
    sys.modules["yprov_mlflow_plugin.tracking"] = mod
    return mod

def _normalize_file_path(raw: str) -> str:
    """
    Convert a file://... or platform-specific path into a local filesystem path string
    suitable for mlflow.store.tracking.file_store.FileStore.
    """
    # If it's a file:// URI, extract the path
    if raw.startswith("file://"):
        p = urlparse(raw).path
    else:
        p = raw

    # Handle Windows drive letters that may appear as /C:/... or C:/...
    m = re.match(r"^/[A-Za-z]:/.*", p)  # e.g. /C:/Users/...
    if m:
        p = p.lstrip("/")  # -> C:/Users/...

    # If it still looks like a Windows drive (C:/...), keep as-is; otherwise absolutize
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return p.replace("\\", "/")  # normalize slashes

    # On WSL, urlparse(...).path is already a Linux path like /mnt/c/...
    return os.path.abspath(p)


class TestYProvLogging:
    """Test that tracking.py actually logs to yprov"""
    
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        """Set up test environment"""
        monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path / "prov"))
        monkeypatch.setenv("YPROV_DEBUG", "1")
        
    def test_log_param_calls_yprov(self, tmp_path, monkeypatch):
        """Verify that log_param actually calls yprov.log_param"""
        fresh_tracking_module()

        yprov_calls = []
        with patch('yprov_mlflow_plugin.tracking.yprov') as mock_yprov:
            mock_yprov.start_run = MagicMock()
            mock_yprov.log_param = MagicMock(side_effect=lambda k, v: yprov_calls.append(('param', k, v)))
            mock_yprov.log_metric = MagicMock()
            mock_yprov.end_run = MagicMock()
            
            from yprov_mlflow_plugin.tracking import YProvTrackingStore
            store = YProvTrackingStore(f"file://{tmp_path}/mlruns")
            run = store.create_run(experiment_id="1", run_name="test")
            
            store.log_param(run.info.run_id, SimpleNamespace(key="learning_rate", value=0.001))
            
            if len(yprov_calls) == 0:
                pytest.fail("❌ BUG CONFIRMED: yprov.log_param was NOT called!\n"
                            "The log_param method only logs to MLflow, not to yprov.")
            
            assert len(yprov_calls) == 1
            assert yprov_calls[0] == ('param', 'learning_rate', 0.001)
            print("✅ PASS: yprov.log_param is being called")
    
    def test_log_metric_calls_yprov(self, tmp_path, monkeypatch):
        """Verify that log_metric actually calls yprov.log_metric"""
        fresh_tracking_module()

        yprov_calls = []
        with patch('yprov_mlflow_plugin.tracking.yprov') as mock_yprov:
            mock_yprov.start_run = MagicMock()
            mock_yprov.log_param = MagicMock()
            mock_yprov.log_metric = MagicMock(side_effect=lambda k, v, **kw: yprov_calls.append(('metric', k, v)))
            mock_yprov.end_run = MagicMock()
            
            from yprov_mlflow_plugin.tracking import YProvTrackingStore
            store = YProvTrackingStore(f"file://{tmp_path}/mlruns")
            run = store.create_run(experiment_id="1", run_name="test")
            
            store.log_metric(run.info.run_id, SimpleNamespace(key="loss", value=0.5, step=1))
            
            if len(yprov_calls) == 0:
                pytest.fail("❌ BUG CONFIRMED: yprov.log_metric was NOT called!\n"
                            "The log_metric method only logs to MLflow, not to yprov.")
            
            assert len(yprov_calls) == 1
            assert yprov_calls[0] == ('metric', 'loss', 0.5)
            print("✅ PASS: yprov.log_metric is being called")
    
    def test_log_batch_calls_yprov(self, tmp_path, monkeypatch):
        """Verify that log_batch actually calls yprov for params and metrics"""
        fresh_tracking_module()

        yprov_calls = {'params': [], 'metrics': []}
        with patch('yprov_mlflow_plugin.tracking.yprov') as mock_yprov:
            mock_yprov.start_run = MagicMock()
            mock_yprov.log_param = MagicMock(side_effect=lambda k, v: yprov_calls['params'].append((k, v)))
            mock_yprov.log_metric = MagicMock(side_effect=lambda k, v, **kw: yprov_calls['metrics'].append((k, v)))
            mock_yprov.end_run = MagicMock()
            
            from yprov_mlflow_plugin.tracking import YProvTrackingStore
            store = YProvTrackingStore(f"file://{tmp_path}/mlruns")
            run = store.create_run(experiment_id="1", run_name="test")
            
            params = [
                SimpleNamespace(key="lr", value=0.001),
                SimpleNamespace(key="batch_size", value=32)
            ]
            metrics = [
                SimpleNamespace(key="loss", value=0.5, step=1),
                SimpleNamespace(key="acc", value=0.8, step=1)
            ]
            store.log_batch(run.info.run_id, metrics=metrics, params=params, tags=None)
            
            if len(yprov_calls['params']) == 0:
                pytest.fail("❌ BUG CONFIRMED: log_batch does not call yprov.log_param!\n"
                            "Batch parameters are only logged to MLflow, not to yprov.")
            
            if len(yprov_calls['metrics']) == 0:
                pytest.fail("❌ BUG CONFIRMED: log_batch does not call yprov.log_metric!\n"
                            "Batch metrics are only logged to MLflow, not to yprov.")
            
            assert len(yprov_calls['params']) == 2
            assert len(yprov_calls['metrics']) == 2
            print("✅ PASS: log_batch correctly logs to yprov")
    
    def test_end_run_called_with_create_graph(self, tmp_path, monkeypatch):
        """Verify that end_run is called (checking create_graph is optional)"""
        fresh_tracking_module()

        end_run_calls = []
        with patch('yprov_mlflow_plugin.tracking.yprov') as mock_yprov:
            mock_yprov.start_run = MagicMock()
            mock_yprov.log_param = MagicMock()
            mock_yprov.log_metric = MagicMock()
            mock_yprov.end_run = MagicMock(side_effect=lambda **kw: end_run_calls.append(kw))
            
            from yprov_mlflow_plugin.tracking import YProvTrackingStore
            store = YProvTrackingStore(f"file://{tmp_path}/mlruns")
            run = store.create_run(experiment_id="1", run_name="test")
            store.set_terminated(run.info.run_id, status="FINISHED", end_time=123456)
            
            assert len(end_run_calls) > 0, "end_run was not called!"
            
            if end_run_calls and 'create_graph' in end_run_calls[0]:
                create_graph = end_run_calls[0]['create_graph']
                if create_graph is False:
                    pytest.fail(f"⚠️ WARNING: create_graph={create_graph}\n"
                                "This might prevent PROV JSON generation. Should be True.")
                print(f"✅ PASS: end_run called with create_graph={create_graph}")
            else:
                print("✅ PASS: end_run called (create_graph not specified, using default)")


class TestFullLifecycle:
    """Test complete run lifecycle"""
    
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        """Set up test environment"""
        monkeypatch.setenv("YPROV_OUT_DIR", str(tmp_path / "prov"))
        monkeypatch.setenv("YPROV_DEBUG", "1")
    
    def test_complete_lifecycle(self, tmp_path, monkeypatch):
        """Test: create run -> log params/metrics -> end run"""
        fresh_tracking_module()

        calls = {'start_run': [], 'log_param': [], 'log_metric': [], 'end_run': []}
        with patch('yprov_mlflow_plugin.tracking.yprov') as mock_yprov:
            mock_yprov.start_run = MagicMock(side_effect=lambda **kw: calls['start_run'].append(kw))
            mock_yprov.log_param = MagicMock(side_effect=lambda k, v: calls['log_param'].append((k, v)))
            mock_yprov.log_metric = MagicMock(side_effect=lambda k, v, **kw: calls['log_metric'].append((k, v)))
            mock_yprov.end_run = MagicMock(side_effect=lambda **kw: calls['end_run'].append(kw))
            
            from yprov_mlflow_plugin.tracking import YProvTrackingStore
            store = YProvTrackingStore(f"file://{tmp_path}/mlruns")
            
            run = store.create_run(experiment_id="1", run_name="test")
            store.log_param(run.info.run_id, SimpleNamespace(key="lr", value=0.001))
            store.log_param(run.info.run_id, SimpleNamespace(key="epochs", value=10))
            for i in range(3):
                store.log_metric(run.info.run_id, SimpleNamespace(key="loss", value=0.5 - i * 0.1, step=i))
            store.set_terminated(run.info.run_id, status="FINISHED", end_time=123456)
            
            print("\n" + "=" * 60)
            print("LIFECYCLE TEST RESULTS")
            print("=" * 60)
            print(f"start_run calls: {len(calls['start_run'])}")
            print(f"log_param calls: {len(calls['log_param'])}")
            print(f"log_metric calls: {len(calls['log_metric'])}")
            print(f"end_run calls: {len(calls['end_run'])}")
            print("=" * 60)
            
            assert len(calls['start_run']) >= 1, "start_run not called"
            assert len(calls['end_run']) >= 1, "end_run not called"
            if len(calls['log_param']) == 0:
                pytest.fail("❌ BUG CONFIRMED: yprov.log_param never called during lifecycle!\n"
                            "Parameters are only logged to MLflow, PROV files will be empty.")
            if len(calls['log_metric']) == 0:
                pytest.fail("❌ BUG CONFIRMED: yprov.log_metric never called during lifecycle!\n"
                            "Metrics are only logged to MLflow, PROV files will be empty.")
            
            assert len(calls['log_param']) == 2, f"Expected 2 params, got {len(calls['log_param'])}"
            assert len(calls['log_metric']) == 3, f"Expected 3 metrics, got {len(calls['log_metric'])}"
            print("✅ ALL TESTS PASSED: Full lifecycle working correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
