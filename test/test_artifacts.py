
import sys, types, importlib
from pathlib import Path

PKG = "yprov_mlflow_plugin.artifacts"

def _install_fake_mlflow(local_repo_cls=None):
    mlflow = types.ModuleType("mlflow")
    store = types.ModuleType("mlflow.store")
    artifact = types.ModuleType("mlflow.store.artifact")
    local_mod = types.ModuleType("mlflow.store.artifact.local_artifact_repo")

    class _LocalDefault:
        def __init__(self, base_uri): self.base_uri = base_uri

    local_mod.LocalArtifactRepository = local_repo_cls or _LocalDefault
    artifact.local_artifact_repo = local_mod
    store.artifact = artifact
    mlflow.store = store

    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.store"] = store
    sys.modules["mlflow.store.artifact"] = artifact
    sys.modules["mlflow.store.artifact.local_artifact_repo"] = local_mod

def test_strip_yprov():
    _install_fake_mlflow()
    mod = importlib.reload(importlib.import_module(PKG))
    strip = getattr(mod, "_strip_yprov")
    assert strip("yprov+file:///tmp") == "file:///tmp"
    assert strip("/tmp") == "/tmp"

def test_init_sets_delegate_with_stripped_uri(tmp_path):
    rec = {}
    class LocalRepo:
        def __init__(self, base_uri): rec["base"] = base_uri
    _install_fake_mlflow(local_repo_cls=LocalRepo)
    mod = importlib.reload(importlib.import_module(PKG))
    repo = mod.YProvArtifactRepo(f"yprov+{tmp_path.as_posix()}")
    assert isinstance(repo._delegate, LocalRepo)
    assert rec["base"] == tmp_path.as_posix()

def test_log_artifact_mirrors_and_delegates(monkeypatch, tmp_path):
    calls = {"yprov": [], "delegate": []}

    # Fake yprov module
    import types
    yprov = types.ModuleType("prov4ml")
    def log_artifact(path, artifact_path=None):
        calls["yprov"].append((Path(path).name, artifact_path))
    yprov.log_artifact = log_artifact
    sys.modules["prov4ml"] = yprov

    # Fake local artifact repo
    class LocalRepo:
        def __init__(self, base_uri): pass
        def log_artifact(self, local_file, artifact_path=None):
            calls["delegate"].append((Path(local_file).name, artifact_path))
            return "ok"

    _install_fake_mlflow(local_repo_cls=LocalRepo)
    mod = importlib.reload(importlib.import_module(PKG))
    repo = mod.YProvArtifactRepo(tmp_path.as_posix())

    # create dummy file
    f = tmp_path / "model.bin"
    f.write_bytes(b"abc")

    out = repo.log_artifact(str(f), artifact_path="models")
    assert out == "ok"
    assert calls["yprov"] == [("model.bin", "models")]
    assert calls["delegate"] == [("model.bin", "models")]
