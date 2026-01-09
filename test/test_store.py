
import os
import sys
from pathlib import Path
import types

import importlib

PKG = "yprov_mlflow_plugin.tracking"

def _install_fake_mlflow(rest_cls=None, file_cls=None):
    # Create a fake mlflow module tree with rest_store and file_store
    mlflow = types.ModuleType("mlflow")
    store = types.ModuleType("mlflow.store")
    tracking = types.ModuleType("mlflow.store.tracking")
    rest_store = types.ModuleType("mlflow.store.tracking.rest_store")
    file_store = types.ModuleType("mlflow.store.tracking.file_store")

    class _DefaultRest:
        def __init__(self, uri): self.uri = uri
    class _DefaultFile:
        def __init__(self, uri): self.uri = uri

    rest_store.RestStore = rest_cls or _DefaultRest
    file_store.FileStore = file_cls or _DefaultFile

    tracking.rest_store = rest_store
    tracking.file_store = file_store
    store.tracking = tracking
    mlflow.store = store

    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.store"] = store
    sys.modules["mlflow.store.tracking"] = tracking
    sys.modules["mlflow.store.tracking.rest_store"] = rest_store
    sys.modules["mlflow.store.tracking.file_store"] = file_store

def test_strip_yprov_prefix():
    _install_fake_mlflow()
    mod = importlib.reload(importlib.import_module(PKG))
    strip = getattr(mod, "_strip_yprov")
    assert strip("yprov+http://x") == "http://x"
    assert strip("yprov+file:///tmp") == "file:///tmp"
    assert strip("file:///tmp") == "file:///tmp"

def test_delegate_for_http_uses_reststore():
    calls = {}
    class Rest:
        def __init__(self, uri): calls["uri"] = uri
    _install_fake_mlflow(rest_cls=Rest)
    mod = importlib.reload(importlib.import_module(PKG))
    d = mod._delegate_for("yprov+https://example.com/mlflow")
    assert isinstance(d, Rest)
    assert calls["uri"] == "https://example.com/mlflow"

def test_delegate_for_file_normalizes_and_uses_filestore(tmp_path, monkeypatch):
    # record the URI passed to FileStore
    rec = {}
    class File:
        def __init__(self, uri): rec["uri"] = uri

    _install_fake_mlflow(file_cls=File)
    mod = importlib.reload(importlib.import_module(PKG))

    # relative path -> absolute file://
    rel = tmp_path / "mlruns"
    uri = f"yprov+{rel}"
    d = mod._delegate_for(uri)
    assert isinstance(d, File)
    assert rec["uri"].startswith("file://")
    assert rec["uri"].endswith(str(rel))

    # explicit file:// stays file:// and normalized
    d = mod._delegate_for(f"yprov+file://{rel}")
    assert isinstance(d, File)
    assert rec["uri"].startswith("file://")

def test_yprov_out_dir_env(monkeypatch):
    _install_fake_mlflow()
    monkeypatch.setenv("YPROV_OUT_DIR", "/tmp/prov_out")
    mod = importlib.reload(importlib.import_module(PKG))
    # monkeypatch delegate to avoid importing real mlflow
    class FakeDelegate:
        pass
    mod._delegate_for = lambda uri: FakeDelegate()
    store = mod.YProvTrackingStore("file:///tmp/mlruns")
    assert Path(store._prov_out) == Path("/tmp/prov_out")
