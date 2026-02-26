# yprov_mlflow_plugin/artifacts.py
"""
YProvArtifactRepo – delegates to the appropriate MLflow ArtifactRepository
implementation while mirroring artifact logs to yProv4ML / prov4ml.

Scheme routing:
  yprov+file://   -> LocalArtifactRepository
  yprov+http(s):// -> HttpArtifactRepository (when mlflow-server is used)
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

# Try all known base class locations across MLflow versions.
try:
    from mlflow.store.artifact_repo import ArtifactRepository as _BaseArtifactRepo
except Exception:
    try:
        from mlflow.store.artifact.artifact_repo import ArtifactRepository as _BaseArtifactRepo
    except Exception:
        class _BaseArtifactRepo:  # minimal stub for isolated tests
            def __init__(self, artifact_uri: str):
                self.artifact_uri = artifact_uri

# Optional yProv integration
try:
    import prov4ml as yprov
except Exception:
    yprov = None  # type: ignore[assignment]


def _strip_yprov(uri: str) -> str:
    """Remove the 'yprov+' scheme prefix, leaving the underlying URI."""
    return uri.replace("yprov+", "", 1) if uri.startswith("yprov+") else uri


def _make_delegate(base_uri: str):
    """
    Construct the right MLflow ArtifactRepository for *base_uri*.

    ``base_uri`` has already had the ``yprov+`` prefix stripped.
    """
    scheme = urlparse(base_uri).scheme

    if scheme in ("http", "https"):
        # Remote MLflow tracking server – use the HTTP artifact repo.
        try:
            from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
            return HttpArtifactRepository(base_uri)
        except ImportError:
            # Older MLflow versions don't ship this class; fall back gracefully.
            from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
            return LocalArtifactRepository(base_uri)

    # Default: local filesystem (file:// or bare path)
    from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
    return LocalArtifactRepository(base_uri)


class YProvArtifactRepo(_BaseArtifactRepo):
    """
    Artifact repository that:
      1. Delegates all real storage to the appropriate MLflow backend.
      2. Mirrors artifact events to prov4ml for PROV provenance tracking.
    """

    def __init__(self, artifact_uri: str):
        super().__init__(artifact_uri)
        base = _strip_yprov(artifact_uri)
        self._delegate = _make_delegate(base)

    # ------------------------------------------------------------------
    # Write operations – mirror to yprov, then delegate
    # ------------------------------------------------------------------

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None):
        if yprov:
            try:
                yprov.log_artifact(local_file, artifact_path=artifact_path)
            except Exception:
                pass  # provenance failure must never block the actual log
        return self._delegate.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        if yprov:
            try:
                # prov4ml may only expose log_artifact; passing a directory is fine.
                yprov.log_artifact(local_dir, artifact_path=artifact_path)
            except Exception:
                pass
        return self._delegate.log_artifacts(local_dir, artifact_path)

    # ------------------------------------------------------------------
    # Read operations – pure delegation
    # ------------------------------------------------------------------

    def list_artifacts(self, path: Optional[str] = None):
        return self._delegate.list_artifacts(path)

    def download_artifacts(self, artifact_path: str, dst_path: Optional[str] = None):
        return self._delegate.download_artifacts(artifact_path, dst_path)

    def _download_file(self, remote_file_path: str, local_path: str):
        """Required by some MLflow versions."""
        return self._delegate._download_file(remote_file_path, local_path)
