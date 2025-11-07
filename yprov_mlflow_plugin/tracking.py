from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from urllib.parse import urlparse
from pathlib import Path
import os, re, sys

if TYPE_CHECKING:
    from mlflow.store.tracking.abstract_store import AbstractStore  # type: ignore
else:
    class AbstractStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

# yprov (prov4ml)
try:
    import prov4ml as yprov
except Exception:
    yprov = None

# Try eager import of export_run_to_prov; tests may monkeypatch module var later
export_run_to_prov = None
try:
    from .prov_export import export_run_to_prov  # type: ignore
except ImportError as e:
    print(f"[yProv Plugin] ⚠ WARNING: Cannot import prov_export: {e}", file=sys.stderr, flush=True)
    print(f"[yProv Plugin]   PROV JSON export will not work!", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[yProv Plugin] ⚠ WARNING: Unexpected error importing prov_export: {e}", file=sys.stderr, flush=True)

DEBUG = os.getenv("YPROV_DEBUG", "").lower() in ("1", "true", "yes")

def _debug(msg: str):
    if DEBUG:
        print(f"[yProv Plugin] {msg}", flush=True)

def _env_flag(name: str, default=False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _strip_yprov(uri: str) -> str:
    return uri.replace("yprov+", "", 1) if uri.startswith("yprov+") else uri

def _normalize_file_uri(raw: str) -> str:
    if raw.startswith("file://"):
        p = urlparse(raw).path
    else:
        p = raw
    m = re.match(r"^([A-Za-z]):[\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        if sys.platform.startswith("linux") and os.path.isdir("/mnt"):
            return f"file:///mnt/{drive}/{rest}"
        return f"file:///{drive.upper()}:{'/' + rest if rest else ''}"
    if re.match(r"^/[A-Za-z]:/.*", p):
        p2 = p.lstrip("/")
        drive = p2[0].lower()
        rest = p2[3:]
        if sys.platform.startswith("linux") and os.path.isdir("/mnt"):
            return f"file:///mnt/{drive}/{rest.lstrip('/')}"
        return f"file:///{p2}"
    if p.startswith("/"):
        return f"file://{p}"
    abs_path = os.path.abspath(p)
    return f"file://{abs_path}"

def _delegate_for(tracking_uri: str):
    base = _strip_yprov(tracking_uri)
    parsed = urlparse(base)
    if parsed.scheme in ("http", "https"):
        from mlflow.store.tracking.rest_store import RestStore  # type: ignore
        return RestStore(base)
    if parsed.scheme in ("", "file"):
        base_uri = _normalize_file_uri(base if parsed.scheme == "file" else base)
        from mlflow.store.tracking.file_store import FileStore  # type: ignore
        return FileStore(base_uri)
    from mlflow.store.tracking.rest_store import RestStore  # type: ignore
    return RestStore(base)

class YProvTrackingStore(AbstractStore):
    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        super().__init__()
        self._store_uri = store_uri
        self._delegate = _delegate_for(store_uri)

        # tests expect _prov_out to exist and reflect YPROV_OUT_DIR
        self._prov_out = Path(os.getenv("YPROV_OUT_DIR", "data/prov"))
        self._prov_out.mkdir(parents=True, exist_ok=True)

        _debug(f"🟢 YProvTrackingStore initialized with URI: {store_uri}")
        _debug(f"   Delegate type: {type(self._delegate).__name__}")
        _debug(f"   PROV output: {self._prov_out}")
        _debug(f"   export_run_to_prov available: {export_run_to_prov is not None}")
        if export_run_to_prov is None:
            _debug(f"   ⚠️ WARNING: PROV export will NOT work - prov_export module failed to import!")

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def _exp_name(self, experiment_id: str) -> str:
        try:
            exp = self._delegate.get_experiment(experiment_id)
            return (exp.name if exp else None) or "Default"
        except Exception:
            return "Default"

    def _export_prov(self, run_id: str):
        """Export PROV for a run (fallback path used by tests)."""
        _debug(f"  _export_prov called for run_id={run_id}")

        # Allow tests to monkeypatch module var
        export_func = export_run_to_prov
        if export_func is None:
            try:
                from yprov_mlflow_plugin.prov_export import export_run_to_prov as export_func  # type: ignore
                _debug(f"  ✓ Lazy-loaded export_run_to_prov successfully")
            except Exception as e:
                _debug(f"  ⚠ Cannot import export_run_to_prov: {e}")
                if DEBUG:
                    import traceback; traceback.print_exc()
                return

        try:
            from mlflow.tracking import MlflowClient  # late import to avoid circulars
            client = MlflowClient()
            run = client.get_run(run_id)
            exp_id = run.info.experiment_id
            exp = client.get_experiment(exp_id)
            exp_name = exp.name if exp else "Default"

            out = self._prov_out / exp_name
            out.mkdir(parents=True, exist_ok=True)

            prov_file = export_func(run_id, out, client=client)
            _debug(f"  ✓ PROV exported to: {prov_file}")
        except Exception as e:
            _debug(f"  ⚠ PROV export failed: {e}")
            if DEBUG:
                import traceback; traceback.print_exc()

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        _debug(f"create_run called: exp={experiment_id}, run_name={run_name}")

        # Create the MLflow run first (lets us query experiment safely)
        result = self._delegate.create_run(experiment_id, user_id, start_time, tags, run_name=run_name, **kwargs)
        run_id = result.info.run_id

        exp_name = self._exp_name(experiment_id)
        exp_dir = (self._prov_out / exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Unify config
        unify = _env_flag("YPROV_UNIFY", False)
        merged_path = os.getenv("YPROV_MERGED_PATH")
        if merged_path:
            try:
                # Set before yprov.start_run so end_run uses it
                from prov4ml import prov4ml as _p  # type: ignore
                Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
                _p.PROV4ML_DATA.PROV_MERGED_PATH = str(merged_path)
                _debug(f"   Using merged path: {merged_path}")
            except Exception as e:
                _debug(f"   ⚠ Failed to set PROV_MERGED_PATH: {e}")

        # Start yprov (fix: pass required args)
        if yprov:
            try:
                # Prefer CSV metrics to avoid zarr/netcdf deps by default
                try:
                    from prov4ml.datamodel.metric_type import MetricsType  # type: ignore
                    metrics_type = MetricsType.CSV
                except Exception:
                    metrics_type = None

                kwargs_sr = dict(
                    prov_user_namespace=os.getenv("YPROV_USER_NAMESPACE", "yProv4ML"),
                    experiment_name=exp_name,
                    provenance_save_dir=str(exp_dir),
                    collect_all_processes=_env_flag("YPROV_COLLECT_ALL", False),
                    csv_separator=",",
                    unify_experiments=unify,
                )
                if metrics_type is not None:
                    kwargs_sr["metrics_file_type"] = metrics_type

                yprov.start_run(**kwargs_sr)
                _debug(f"  ✓ prov4ml.start_run(): dir={exp_dir} exp={exp_name} unify={unify}")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.start_run() failed: {e}")
        else:
            _debug("  ⚠ prov4ml not available")

        _debug(f"  Created run_id: {run_id}")
        return result

    def set_terminated(self, run_id, status, end_time):
        _debug(f"set_terminated called: run_id={run_id}, status={status}")

        if yprov:
            try:
                yprov.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)
                _debug("  ✓ prov4ml.end_run() called")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.end_run() failed: {e}")
        else:
            _debug("  ⚠ prov4ml not available")

        # Keep export fallback (tests rely on it)
        if end_time is not None:
            self._export_prov(run_id)

        return self._delegate.set_terminated(run_id, status, end_time)

    def update_run_info(self, run_id, run_status, end_time, **kwargs):
        _debug(f"update_run_info called: run_id={run_id}, status={run_status}, end_time={end_time}")

        # Terminal?
        is_terminal = False
        try:
            status_str = str(run_status).upper()
            status_int = int(run_status) if str(run_status).isdigit() else -1
            is_terminal = status_str in {"FINISHED", "FAILED", "KILLED"} or status_int in {3, 4, 5}
        except Exception:
            pass
        _debug(f"  is_terminal={is_terminal}, end_time={end_time}")

        if yprov and is_terminal:
            try:
                yprov.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)
                _debug("  ✓ prov4ml.end_run() called")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.end_run() failed: {e}")

        # Export fallback when terminal OR when explicit end_time present (matches tests)
        if is_terminal or (end_time is not None):
            self._export_prov(run_id)

        return self._delegate.update_run_info(run_id, run_status, end_time, **kwargs)

    # Passthroughs
    def log_param(self, run_id, param):
        _debug(f"log_param: {param.key}={param.value}")
        return self._delegate.log_param(run_id, param)

    def log_metric(self, run_id, metric):
        if DEBUG:
            self._metric_log_count = getattr(self, "_metric_log_count", 0) + 1
            if self._metric_log_count <= 3 or self._metric_log_count % 10 == 0:
                _debug(f"log_metric: {metric.key}={metric.value}")
        return self._delegate.log_metric(run_id, metric)

    def log_batch(self, run_id, metrics, params, tags):
        _debug(f"log_batch: {len(params or [])} params, {len(metrics or [])} metrics, {len(tags or [])} tags")
        return self._delegate.log_batch(run_id, metrics, params, tags)

    def get_artifact_uri(self, run_id):
        if hasattr(self._delegate, "get_artifact_uri"):
            return self._delegate.get_artifact_uri(run_id)
        return self._delegate.get_run(run_id).info.artifact_uri

    def shut_down_async_logging(self):
        _debug("shut_down_async_logging called")
        if hasattr(self._delegate, "shut_down_async_logging"):
            return self._delegate.shut_down_async_logging()

    def create_experiment(self, name, artifact_location=None, tags=None):
        _debug(f"create_experiment: {name}")
        return self._delegate.create_experiment(name, artifact_location, tags)

    def get_experiment(self, exp_id):
        return self._delegate.get_experiment(exp_id)

    def get_experiment_by_name(self, name):
        return self._delegate.get_experiment_by_name(name)

    def get_run(self, run_id):
        return self._delegate.get_run(run_id)

    def search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):
        return self._delegate.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by, page_token)

    def list_run_infos(self, experiment_id, run_view_type, max_results, order_by, page_token):
        return self._delegate.list_run_infos(experiment_id, run_view_type, max_results, order_by, page_token)
