# yprov_mlflow_plugin/artifacts.py
from __future__ import annotations
from typing import Optional

# Try all known locations; if none are present (e.g., in tests), use a stub.
try:
    from mlflow.store.artifact_repo import ArtifactRepository as _BaseArtifactRepo
except Exception:
    try:
        from mlflow.store.artifact.artifact_repo import ArtifactRepository as _BaseArtifactRepo
    except Exception:
        class _BaseArtifactRepo:  # minimal stub for tests/fakes
            def __init__(self, artifact_uri: str):
                self.artifact_uri = artifact_uri

# Optional yProv
try:
    import prov4ml as yprov
except Exception:
    yprov = None

def _strip_yprov(uri: str) -> str:
    return uri.replace("yprov+", "", 1) if uri.startswith("yprov+") else uri

class YProvArtifactRepo(_BaseArtifactRepo):
    """
    Delegates to a real LocalArtifactRepository while mirroring artifact logs to yProv.
    """
    def __init__(self, artifact_uri: str):
        super().__init__(artifact_uri)
        base = _strip_yprov(artifact_uri)

        # Lazy import so tests can inject a fake local repo at:
        # mlflow.store.artifact.local_artifact_repo.LocalArtifactRepository
        from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
        self._delegate = LocalArtifactRepository(base)

    def log_artifact(self, local_file, artifact_path: Optional[str] = None):
        if yprov:
            try:
                yprov.log_artifact(local_file, artifact_path=artifact_path)
            except Exception:
                pass
        return self._delegate.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path: Optional[str] = None):
        if yprov:
            try:
                # if prov4ml exposes only log_artifact, calling it with a dir is fine
                yprov.log_artifact(local_dir, artifact_path=artifact_path)
            except Exception:
                pass
        return self._delegate.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, path: Optional[str] = None):
        return self._delegate.list_artifacts(path)

    def download_artifacts(self, artifact_path, dst_path: Optional[str] = None):
        return self._delegate.download_artifacts(artifact_path, dst_path)
