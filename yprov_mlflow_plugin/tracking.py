# yprov_mlflow_plugin/tracking.py
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from urllib.parse import urlparse
import os
import re
import sys


# Avoid importing mlflow at module import time to prevent circular import
# issues when MLflow registers entry points. Static type checkers still
# get the real types via TYPE_CHECKING.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from mlflow.store.tracking.abstract_store import AbstractStore  # type: ignore
else:
    # Minimal runtime stub so the module can be imported even when mlflow
    # is not present; actual store classes are imported lazily in
    # `_delegate_for` when needed.
    class AbstractStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


# Try to import yProv4ML; keep plugin usable even if it's missing
try:
    import prov4ml as yprov  # function-style API expected
except Exception:  # pragma: no cover
    yprov = None


def _strip_yprov(uri: str) -> str:
    """
    Convert yprov+http://... -> http://..., and yprov+file:///... -> file:///...
    """
    return uri.replace("yprov+", "", 1) if uri.startswith("yprov+") else uri


def _delegate_for(tracking_uri: str) -> AbstractStore:
    """
    Build the real MLflow tracking store (REST or File) that we delegate to.
    """
    base = _strip_yprov(tracking_uri)
    parsed = urlparse(base)

    if parsed.scheme in ("http", "https"):
        try:
            # Lazy import to avoid module-level mlflow import
            from mlflow.store.tracking.rest_store import RestStore  # type: ignore

            return RestStore(base)
        except Exception as exc:  # pragma: no cover - runtime error path
            raise ImportError("mlflow is required to use YProvTrackingStore (rest).") from exc

    # Treat "" and "file" as FileStore
    if parsed.scheme in ("", "file"):
        # Normalize to a well-formed file:// URI for the current OS. This
        # handles Windows drive-letter paths when running under WSL by
        # mapping e.g. C:\foo -> /mnt/c/foo so the FileStore writes to a
        # valid path.
        def _normalize_file_uri(raw: str) -> str:
            # If already a file:// URI, extract path
            if raw.startswith("file://"):
                p = urlparse(raw).path
            else:
                p = raw

            # Windows drive-letter like C:\ or C:/ at start
            m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
            if m:
                drive = m.group(1).lower()
                rest = m.group(2).replace("\\", "/")
                # If running under WSL/Posix but path is Windows-style,
                # map to /mnt/<drive>/...
                if sys.platform.startswith("linux") and os.path.isdir("/mnt"):
                    return f"file:///mnt/{drive}/{rest}"
                # On Windows, return file:///C:/path
                return f"file:///{drive.upper()}:{'/' + rest if rest else ''}"

            # If path starts with / and looks like /C:/... (file:// parsed)
            m2 = re.match(r"^/[A-Za-z]:/.*", p)
            if m2:
                # strip leading slash
                p2 = p.lstrip("/")
                drive = p2[0].lower()
                rest = p2[3:]
                if sys.platform.startswith("linux") and os.path.isdir("/mnt"):
                    return f"file:///mnt/{drive}/{rest.lstrip('/') }"
                return f"file:///{p2}"

            # Absolute POSIX path
            if p.startswith("/"):
                return f"file://{p}"

            # Relative path -> make absolute
            abs_path = os.path.abspath(p)
            return f"file://{abs_path}"

        base_uri = _normalize_file_uri(base if parsed.scheme == "file" else base)
        try:
            from mlflow.store.tracking.file_store import FileStore  # type: ignore

            return FileStore(base_uri)
        except Exception as exc:  # pragma: no cover - runtime error path
            raise ImportError("mlflow is required to use YProvTrackingStore (file).") from exc

    # Fallback to REST for any other scheme
    try:
        from mlflow.store.tracking.rest_store import RestStore  # type: ignore

        return RestStore(base)
    except Exception as exc:  # pragma: no cover - runtime error path
        raise ImportError("mlflow is required to use YProvTrackingStore.") from exc


class YProvTrackingStore(AbstractStore):
    """
    A delegating Tracking Store that mirrors MLflow calls to yProv4ML.
    Select via URI schemes:
      - yprov+file:///path/to/mlruns
      - yprov+http://host:5000
    """

    # NOTE: MLflow calls this constructor with (store_uri, artifact_uri=None)
    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        super().__init__()
        self._store_uri = store_uri
        self._delegate = _delegate_for(store_uri)
    def shut_down_async_logging(self):
        if hasattr(self._delegate, "shut_down_async_logging"):
            return self._delegate.shut_down_async_logging()
        # no-op if delegate doesn't have it

    # optional, but future-proof (MLflow sometimes probes this)
    def get_artifact_uri(self, run_id):
        if hasattr(self._delegate, "get_artifact_uri"):
            return self._delegate.get_artifact_uri(run_id)
        # fall back to delegate's behavior via get_run
        return self._delegate.get_run(run_id).info.artifact_uri

    # ------------- Experiment APIs (delegate) -------------

    def create_experiment(self, name, artifact_location=None, tags=None):
        return self._delegate.create_experiment(name, artifact_location, tags)

    def get_experiment(self, exp_id):
        return self._delegate.get_experiment(exp_id)

    def get_experiment_by_name(self, name):
        return self._delegate.get_experiment_by_name(name)

    # ------------- Run lifecycle -------------

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        # mirror start to yProv4ML
        if yprov:
            try:
                yprov.start_run(run_name=run_name or (tags or {}).get("mlflow.runName"))
                for k, v in (tags or {}).items():
                    try:
                        yprov.log_param(f"mlflow.tag.{k}", v)
                    except Exception:
                        pass
            except Exception:
                pass

        # IMPORTANT: forward run_name (and any future kwargs) to the real store
        return self._delegate.create_run(
            experiment_id, user_id, start_time, tags, run_name=run_name, **kwargs
        )

    def set_terminated(self, run_id, status, end_time):
        # Mirror end to yProv4ML
        if yprov:
            try:
                yprov.end_run(create_graph=False)  # set True if you want graph by default
            except Exception:
                pass

        return self._delegate.set_terminated(run_id, status, end_time)

    # Some clients close via update rather than set_terminated; mirror defensively
    def update_run_info(self, run_id, run_status, end_time, **kwargs):
        # Accept and forward extra kwargs (e.g. run_name) that newer MLflow
        # versions may pass to be forward-compatible with changing APIs.
        if yprov and str(run_status).upper() in {"FINISHED", "FAILED", "KILLED"}:
            try:
                yprov.end_run(create_graph=False)
            except Exception:
                pass
        return self._delegate.update_run_info(run_id, run_status, end_time, **kwargs)

    # ------------- Logging -------------

    def log_param(self, run_id, param):
        try:
            if yprov:
                # Coerce to str to be safe
                yprov.log_param(param.key, str(param.value))
        finally:
            return self._delegate.log_param(run_id, param)

    def log_metric(self, run_id, metric):
        try:
            if yprov:
                # Be robust to missing step/value types
                step = getattr(metric, "step", None)
                yprov.log_metric(metric.key, float(metric.value), step=step)
        finally:
            return self._delegate.log_metric(run_id, metric)

    def log_batch(self, run_id, metrics, params, tags):
        try:
            if yprov:
                for p in params or []:
                    yprov.log_param(p.key, str(p.value))
                for m in metrics or []:
                    yprov.log_metric(m.key, float(m.value), step=getattr(m, "step", None))
                # mirror tags to yProv as params (namespaced)
                for t in tags or []:
                    try:
                        yprov.log_param(f"mlflow.tag.{t.key}", t.value)
                    except Exception:
                        pass
        finally:
            return self._delegate.log_batch(run_id, metrics, params, tags)

    # ------------- Tags / queries / listings (delegate) -------------

    def set_tag(self, run_id, tag):
        return self._delegate.set_tag(run_id, tag)

    def delete_tag(self, run_id, tag_name):
        return self._delegate.delete_tag(run_id, tag_name)

    def get_run(self, run_id):
        return self._delegate.get_run(run_id)

    def search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results,
        order_by,
        page_token,
    ):
        return self._delegate.search_runs(
            experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
        )

    def list_run_infos(
        self,
        experiment_id,
        run_view_type,
        max_results,
        order_by,
        page_token,
    ):
        return self._delegate.list_run_infos(
            experiment_id, run_view_type, max_results, order_by, page_token
        )
