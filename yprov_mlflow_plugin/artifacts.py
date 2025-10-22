from __future__ import annotations
from mlflow.store.artifact.artifact_repo import ArtifactRepository

# choose delegate based on final uri; here we support local only for brevity
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository

try:
    import prov4ml as yprov
except Exception:
    yprov = None


def _strip_yprov(uri: str) -> str:
    return uri.replace("yprov+", "", 1) if uri.startswith("yprov+") else uri


class YProvArtifactRepo(ArtifactRepository):
    """
    Mirrors artifact logs to yProv4ML, delegates to real repo.
    Register via entry points with schemes: yprov+file, yprov+http, etc.
    """

    def __init__(self, artifact_uri: str):
        super().__init__(artifact_uri)
        base = _strip_yprov(artifact_uri)
        # minimal: local delegate; extend for s3/http as needed
        self._delegate = LocalArtifactRepository(base)

    def log_artifact(self, local_file, artifact_path=None):
        if yprov:
            try:
                yprov.log_artifact(local_file, artifact_path=artifact_path)
            except Exception:
                pass
        return self._delegate.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        if yprov:
            try:
                yprov.log_artifact(local_dir, artifact_path=artifact_path)
            except Exception:
                pass
        return self._delegate.log_artifacts(local_dir, artifact_path)

    # delegate everything else
    def list_artifacts(self, path=None):
        return self._delegate.list_artifacts(path)

    def download_artifacts(self, artifact_path, dst_path=None):
        return self._delegate.download_artifacts(artifact_path, dst_path)
