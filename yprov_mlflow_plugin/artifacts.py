"""
YProvArtifactRepo — MLflow ArtifactRepository plugin for yProv4ML.

Mirrors mlflow.log_artifact() / log_artifacts() to yprov4ml.log_artifact()
while delegating the actual file storage to the real backend
(LocalArtifactRepository or HttpArtifactRepository) based on the URI scheme.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# MLflow ArtifactRepository base class
# ---------------------------------------------------------------------------
try:
    from mlflow.store.artifact.artifact_repo import ArtifactRepository as _Base
except ImportError:
    try:
        from mlflow.store.artifact_repo import ArtifactRepository as _Base  # type: ignore
    except ImportError:
        class _Base:                                           # type: ignore
            def __init__(self, artifact_uri: str):
                self.artifact_uri = artifact_uri

# ---------------------------------------------------------------------------
# Import yProv4ML — same logic as tracking.py
# ---------------------------------------------------------------------------
import yprov4ml               # pip install yprov4ml  (HPCI-Lab main)

DEBUG = os.getenv("YPROV_DEBUG", "").lower() in ("1", "true", "yes", "on")

def _debug(msg: str) -> None:
    if DEBUG:
        print(f"[yProv Plugin] {msg}", flush=True)

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def _strip_yprov(uri: str) -> str:
    return uri.replace("yprov+", "", 1) if isinstance(uri, str) and uri.startswith("yprov+") else uri

def _make_delegate(artifact_uri: str):
    """
    Build the real MLflow artifact backend for the given URI:
      yprov+file://...  → LocalArtifactRepository
      yprov+http://...  → HttpArtifactRepository
      yprov+https://... → HttpArtifactRepository
      (anything else)   → LocalArtifactRepository
    """
    base   = _strip_yprov(artifact_uri)
    scheme = urlparse(base).scheme

    if scheme in ("http", "https"):
        try:
            from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
            _debug(f"  delegate → HttpArtifactRepository({base!r})")
            return HttpArtifactRepository(base)
        except Exception as exc:
            _debug(f"  ⚠ HttpArtifactRepository failed: {exc}")
            # Fallback to local
            base = "."

    # file:// or bare path → LocalArtifactRepository
    from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
    _debug(f"  delegate → LocalArtifactRepository({base!r})")
    return LocalArtifactRepository(base)

# ---------------------------------------------------------------------------
# Main repo class
# ---------------------------------------------------------------------------

class YProvArtifactRepo(_Base):
    """
    MLflow ArtifactRepository that:
      1. Delegates file storage to LocalArtifactRepository or HttpArtifactRepository.
      2. Calls yprov4ml.log_artifact() so provenance tracks the artifact.
    """

    def __init__(self, artifact_uri: str, tracking_uri: str = None):
        super().__init__(artifact_uri)
        self._delegate = _make_delegate(artifact_uri)
        _debug(f"🟢 YProvArtifactRepo: uri={artifact_uri!r}")

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None):
        """
        Called by mlflow.log_artifact(local_path, artifact_path=...).
        Stores the file via the real backend AND records it in yprov4ml.
        """
        _debug(f"log_artifact: {local_file!r} → {artifact_path!r}")
        # 1. Call yprov4ml first (if available)
        try:
            yprov4ml.log_artifact(artifact_path, local_file, log_copy_in_prov_directory=True)
            _debug(f"  ✓ yprov4ml.log_artifact({local_file}, {artifact_path}, log_copy_in_prov_directory=True)")
        except Exception as exc:
            _debug(f"  ✗ yprov4ml.log_artifact() error: {exc}")
            # 2. Always delegate to real storage
        return self._delegate.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Called by mlflow.log_artifacts(local_dir, artifact_path=...).
        Iterates the directory and logs each file individually to yprov4ml.
        """
        _debug(f"log_artifacts: {local_dir!r} → {artifact_path!r}")
        # Walk all files and log each one individually
        base = Path(local_dir)
        for f in base.rglob("*"):
            if f.is_file():
                # Preserve sub-directory structure inside artifact_path
                rel = f.relative_to(base)
                sub = str(Path(artifact_path) / rel.parent) if artifact_path else str(rel.parent)
                sub = sub.rstrip("/.")
                try:
                    yprov4ml.log_artifact(artifact_path, sub, log_copy_in_prov_directory=True)
                    _debug(f"  ✓ yprov4ml.log_artifact({sub}, {artifact_path}, log_copy_in_prov_directory=True)")
                except Exception as exc:
                    _debug(f"  ✗ yprov4ml.log_artifact() error: {exc}")

        return self._delegate.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, path: Optional[str] = None):
        return self._delegate.list_artifacts(path)

    def download_artifacts(self, artifact_path: str, dst_path: Optional[str] = None):
        return self._delegate.download_artifacts(artifact_path, dst_path)

    def _download_file(self, remote_file_path: str, local_path: str):
        """Required by some MLflow versions."""
        if hasattr(self._delegate, "_download_file"):
            return self._delegate._download_file(remote_file_path, local_path)
        raise NotImplementedError(f"_download_file not supported by {type(self._delegate).__name__}")
