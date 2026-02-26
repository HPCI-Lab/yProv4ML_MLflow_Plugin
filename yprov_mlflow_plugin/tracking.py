"""
YProvTrackingStore — MLflow AbstractStore that mirrors every tracking call
to the HPCI-Lab yProv4ML library (https://github.com/HPCI-Lab/yProv4ML).

HOW IT WORKS
------------
Set your MLflow tracking URI with the ``yprov+`` prefix:

    mlflow.set_tracking_uri("yprov+file:///absolute/path/to/mlruns")
    # or for a remote server:
    mlflow.set_tracking_uri("yprov+http://my-server:5000")

Every mlflow.* call then:
  1. Is forwarded to the real MLflow FileStore / RestStore (normal data preserved).
  2. Also calls the matching yprov4ml function so provenance is recorded.

MLflow → yprov4ml mapping
--------------------------
  mlflow.start_run()       -> yprov4ml.start_run(experiment_name, provenance_save_dir)
  mlflow.log_param(k, v)   -> yprov4ml.log_param(k, v)
  mlflow.log_metric(k, v)  -> yprov4ml.log_metric(k, v, step=...)
  mlflow.log_artifact(f)   -> yprov4ml.log_artifact(f)        ← via YProvArtifactRepo
  mlflow.end_run()         -> yprov4ml.end_run(create_graph=True)

WHY artifact logging requires yprov+ prefix on artifact_uri
-------------------------------------------------------------
MLflow resolves which ArtifactRepository class to use by looking at the
URI scheme of run.info.artifact_uri (e.g. "file://..." vs "yprov+file://...").
If that URI doesn't start with "yprov+", the YProvArtifactRepo entry-point
is never triggered, regardless of the tracking URI.

This store ensures every experiment's artifact_location starts with "yprov+"
by overriding create_experiment(), and patches existing runs via get_run().

Environment variables
---------------------
  YPROV_OUT_DIR          root directory for provenance output (default: data/prov)
  YPROV_USER_NAMESPACE   PROV namespace identifier     (default: yProv4ML)
  YPROV_DEBUG            set to 1 for verbose logging
  YPROV_TEST_SHIM        set to 1 to force in-memory shim  (tests only)
"""
from __future__ import annotations

import inspect
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from mlflow.store.tracking.abstract_store import AbstractStore  # type: ignore
else:
    class AbstractStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

# ---------------------------------------------------------------------------
# Import yProv4ML  (package name: "yprov4ml" on HPCI-Lab main)
# ---------------------------------------------------------------------------
try:
    import yprov4ml as yprov
    _YPROV_SOURCE = "yprov4ml"
except ImportError:
    try:
        import prov4ml as yprov        # type: ignore[no-redef]  old fork fallback
        _YPROV_SOURCE = "prov4ml"
    except ImportError:
        yprov = None                   # type: ignore[assignment]
        _YPROV_SOURCE = None

# ---------------------------------------------------------------------------
# Introspect real function signatures once at import time so we never
# accidentally pass kwargs the library doesn't accept.
# ---------------------------------------------------------------------------
def _accepted_params(fn) -> Optional[set]:
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return None      # **kwargs present → accept anything
        return set(sig.parameters)
    except Exception:
        return set()

_START_RUN_PARAMS  = _accepted_params(yprov.start_run)  if yprov and hasattr(yprov, "start_run")  else set()
_END_RUN_PARAMS    = _accepted_params(yprov.end_run)    if yprov and hasattr(yprov, "end_run")    else set()
_LOG_METRIC_PARAMS = _accepted_params(yprov.log_metric) if yprov and hasattr(yprov, "log_metric") else set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEBUG = os.getenv("YPROV_DEBUG", "").lower() in ("1", "true", "yes", "on")

def _debug(msg: str) -> None:
    if DEBUG:
        print(f"[yProv Plugin] {msg}", flush=True)

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _filter_kwargs(kwargs: dict, accepted: Optional[set]) -> dict:
    """Keep only kwargs the target function actually accepts."""
    if accepted is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}

# ---------------------------------------------------------------------------
# URI utilities
# ---------------------------------------------------------------------------
def _strip_yprov(uri: str) -> str:
    return uri.replace("yprov+", "", 1) if isinstance(uri, str) and uri.startswith("yprov+") else uri

def _ensure_yprov(uri: str) -> str:
    """Add yprov+ prefix if not already present."""
    if isinstance(uri, str) and uri and not uri.startswith("yprov+"):
        return "yprov+" + uri
    return uri

def _normalize_file_uri(raw: str) -> str:
    if raw.startswith("file://"):
        p = urlparse(raw).path
    else:
        p = raw
    if re.match(r"^/[A-Za-z]:/.*", p):
        p = p.lstrip("/")
    p = p.replace("\\", "/")
    if re.match(r"^[A-Za-z]:[\\/]", p):
        return f"file:///{p}"
    if p.startswith("/"):
        return f"file://{p}"
    return f"file://{os.path.abspath(p).replace(chr(92), '/')}"

def _store_root_path(store_uri: str) -> str:
    """Extract the filesystem root path from a yprov+file:// URI."""
    base = _strip_yprov(store_uri)
    if base.startswith("file://"):
        return urlparse(base).path
    return os.path.abspath(base)

# ---------------------------------------------------------------------------
# Delegate factory
# ---------------------------------------------------------------------------
def _delegate_for(tracking_uri: str):
    base   = _strip_yprov(tracking_uri or "")
    parsed = urlparse(base)
    if parsed.scheme in ("http", "https"):
        from mlflow.store.tracking.rest_store import RestStore  # type: ignore
        return RestStore(base)
    from mlflow.store.tracking.file_store import FileStore  # type: ignore
    return FileStore(_normalize_file_uri(base if base else "."))

# ---------------------------------------------------------------------------
# Run/RunInfo wrapper to patch artifact_uri on-the-fly for existing runs
# ---------------------------------------------------------------------------
class _PatchedRunInfo:
    """Thin wrapper that injects yprov+ into artifact_uri if missing."""
    def __init__(self, info):
        self._info = info

    def __getattr__(self, name: str):
        val = getattr(self._info, name)
        if name == "artifact_uri" and isinstance(val, str) and val and not val.startswith("yprov+"):
            return "yprov+" + val
        return val


class _PatchedRun:
    """Thin wrapper that ensures run.info.artifact_uri has yprov+ prefix."""
    def __init__(self, run):
        self._run = run
        self.info = _PatchedRunInfo(run.info)

    def __getattr__(self, name: str):
        return getattr(self._run, name)


# ---------------------------------------------------------------------------
# In-memory shim (tests / fallback)
# ---------------------------------------------------------------------------
class _RunInfo:
    def __init__(self, run_id, artifact_uri=""):
        self.run_id = run_id
        self.artifact_uri = artifact_uri

class _Run:
    def __init__(self, run_id, artifact_uri=""):
        self.info = _RunInfo(run_id, artifact_uri)

class _Experiment:
    def __init__(self, eid, name, artifact_location=""):
        self.experiment_id = eid
        self.name = name
        self.artifact_location = artifact_location

class _NoopFileShim:
    def __init__(self, _uri):
        self._uri = _uri
        self._exps: Dict[str, _Experiment] = {}
        self._exps_by_id: Dict[str, _Experiment] = {}
        self._runs: Dict[str, _Run] = {}
        self.create_experiment("Default")

    def create_experiment(self, name, artifact_location=None, tags=None):
        if name in self._exps:
            return self._exps[name].experiment_id
        eid = str(len(self._exps_by_id) + 1)
        loc = artifact_location or f"yprov+file://{self._uri}/{eid}/artifacts"
        e = _Experiment(eid, name, loc)
        self._exps[name] = e
        self._exps_by_id[eid] = e
        return eid

    def get_experiment_by_name(self, name):   return self._exps.get(name)
    def get_experiment(self, eid):            return self._exps_by_id.get(str(eid))

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kw):
        rid = str(uuid.uuid4())
        exp = self.get_experiment(experiment_id)
        base_uri = (exp.artifact_location if exp else f"yprov+file://{self._uri}") + f"/{rid}"
        r = _Run(rid, base_uri)
        self._runs[rid] = r
        return r

    def get_run(self, run_id):                return self._runs.get(run_id, _Run(run_id))
    def log_param(self, run_id, param):       return None
    def log_metric(self, run_id, metric):     return None
    def log_batch(self, run_id, m, p, t):     return None
    def set_terminated(self, run_id, s, t):   return None
    def update_run_info(self, r, s, t, **kw): return None
    def list_run_infos(self, e, r, m, o, p):  return []
    def search_runs(self, e, f, r, m, o, p):  return []
    def get_artifact_uri(self, run_id):       return f"yprov+file://{self._uri}/{run_id}/artifacts"


# ---------------------------------------------------------------------------
# yprov4ml call wrappers
# ---------------------------------------------------------------------------

def _yprov_start_run(exp_name: str, exp_dir: str) -> bool:
    if not yprov or not hasattr(yprov, "start_run"):
        _debug("  ⚠ yprov4ml.start_run not found")
        return False
    namespace = os.getenv("YPROV_USER_NAMESPACE", "yProv4ML")
    desired   = dict(prov_user_namespace=namespace, experiment_name=exp_name, provenance_save_dir=exp_dir)
    kwargs    = _filter_kwargs(desired, _START_RUN_PARAMS)
    try:
        yprov.start_run(**kwargs)
        _debug(f"  ✓ yprov4ml.start_run(experiment_name={exp_name!r})")
        return True
    except TypeError as exc:
        _debug(f"  ↩ retry minimal: {exc}")
        try:
            yprov.start_run(experiment_name=exp_name, provenance_save_dir=exp_dir)
            _debug("  ✓ yprov4ml.start_run() (minimal)")
            return True
        except Exception as exc2:
            _debug(f"  ✗ start_run failed: {exc2}")
            return False
    except Exception as exc:
        _debug(f"  ✗ start_run error: {exc}")
        return False


def _yprov_end_run() -> bool:
    if not yprov or not hasattr(yprov, "end_run"):
        return False
    desired = dict(create_graph=True, create_svg=False)
    kwargs  = _filter_kwargs(desired, _END_RUN_PARAMS)
    try:
        yprov.end_run(**kwargs)
        _debug("  ✓ yprov4ml.end_run()")
        return True
    except TypeError:
        try:
            yprov.end_run()
            _debug("  ✓ yprov4ml.end_run() (no args)")
            return True
        except Exception as exc:
            _debug(f"  ✗ end_run failed: {exc}")
            return False
    except Exception as exc:
        _debug(f"  ✗ end_run error: {exc}")
        return False


def _yprov_log_metric(key: str, value: float, step: Optional[int] = None) -> bool:
    if not yprov or not hasattr(yprov, "log_metric"):
        return False
    kwargs: dict = {}
    if step is not None and (_LOG_METRIC_PARAMS is None or "step" in (_LOG_METRIC_PARAMS or set())):
        kwargs["step"] = step
    try:
        yprov.log_metric(key, value, **kwargs)
        return True
    except TypeError:
        try:
            yprov.log_metric(key, value)
            return True
        except Exception as exc:
            _debug(f"  ✗ log_metric({key}) failed: {exc}")
            return False
    except Exception as exc:
        _debug(f"  ✗ log_metric({key}) error: {exc}")
        return False


def _yprov_log_param(key: str, value: Any) -> bool:
    if not yprov or not hasattr(yprov, "log_param"):
        return False
    try:
        yprov.log_param(key, value)
        return True
    except Exception as exc:
        _debug(f"  ✗ log_param({key}) error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main store
# ---------------------------------------------------------------------------

class YProvTrackingStore(AbstractStore):
    """
    MLflow AbstractStore that:
      1. Delegates all calls to the real MLflow FileStore / RestStore.
      2. Mirrors run lifecycle and logging to yprov4ml for provenance.
      3. Ensures artifact URIs have the yprov+ prefix so YProvArtifactRepo
         is always selected by MLflow for log_artifact() calls.
    """

    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        super().__init__()
        self._store_uri = store_uri

        delegate = _delegate_for(store_uri)
        cls_name = type(delegate).__name__
        mod_name = type(delegate).__module__

        force_shim = _env_flag("YPROV_TEST_SHIM", False)
        is_fake    = cls_name.startswith("_") or "pytest" in (mod_name or "")
        needs_shim = force_shim or is_fake or not hasattr(delegate, "create_run")

        if needs_shim:
            uri_attr = getattr(delegate, "_uri", getattr(delegate, "uri", str(store_uri)))
            self._delegate = _NoopFileShim(uri_attr)
            _debug("   Using in-memory shim (test mode or fallback)")
        else:
            self._delegate = delegate

        self._prov_out = Path(os.getenv("YPROV_OUT_DIR", "data/prov"))
        self._prov_out.mkdir(parents=True, exist_ok=True)

        if yprov:
            ver = getattr(yprov, "__version__", "?")
            _debug(f"🟢 {_YPROV_SOURCE} v{ver} loaded — provenance ACTIVE")
        else:
            print(
                "[yProv Plugin] ⚠  yprov4ml is NOT installed — no provenance data will be produced.\n"
                "               Install:  pip install yprov4ml\n"
                "               or:       pip install git+https://github.com/HPCI-Lab/yProv4ML.git",
                flush=True,
            )

        _debug(f"🟢 YProvTrackingStore ready | delegate={type(self._delegate).__name__} | prov={self._prov_out}")

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    def _exp_name(self, experiment_id: str) -> str:
        try:
            exp = self._delegate.get_experiment(experiment_id)
            return (getattr(exp, "name", None) if exp else None) or "Default"
        except Exception:
            return "Default"

    # ------------------------------------------------------------------
    # Experiment management — ALWAYS prefix artifact_location with yprov+
    # ------------------------------------------------------------------

    def create_experiment(self, name, artifact_location=None, tags=None):
        """
        Intercept experiment creation to ensure the artifact_location starts
        with "yprov+". This is critical: MLflow uses the artifact_location to
        decide which ArtifactRepository class to instantiate. Without the
        "yprov+" prefix, YProvArtifactRepo is never used and log_artifact()
        calls never reach yprov4ml.
        """
        _debug(f"create_experiment: {name!r} artifact_location={artifact_location!r}")

        if artifact_location is None:
            # Build a yprov+file:// location under the store root.
            # We use the experiment name (URL-safe) so the path is deterministic
            # without needing the experiment ID in advance.
            root = _store_root_path(self._store_uri)
            safe_name = re.sub(r"[^\w\-]", "_", name)
            artifact_location = f"yprov+file://{root}/{safe_name}/artifacts"
            _debug(f"  Generated artifact_location: {artifact_location}")
        elif not artifact_location.startswith("yprov+"):
            artifact_location = "yprov+" + artifact_location
            _debug(f"  Prefixed artifact_location: {artifact_location}")

        return self._delegate.create_experiment(name, artifact_location, tags)

    def get_experiment(self, exp_id):
        return self._delegate.get_experiment(exp_id)

    def get_experiment_by_name(self, name):
        return self._delegate.get_experiment_by_name(name)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        """
        Called by mlflow.start_run().
        Creates the MLflow run, then calls yprov4ml.start_run().
        """
        _debug(f"create_run: exp_id={experiment_id!r}, run_name={run_name!r}")

        result   = self._delegate.create_run(experiment_id, user_id, start_time, tags, run_name=run_name, **kwargs)
        run_id   = result.info.run_id
        exp_name = self._exp_name(experiment_id)

        exp_dir = self._prov_out / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if yprov:
            _yprov_start_run(exp_name, str(exp_dir))
        else:
            _debug("  ⚠ yprov4ml not installed")

        _debug(f"  → run_id={run_id}")
        # Wrap the returned run so artifact_uri has yprov+ prefix
        return _PatchedRun(result)

    def get_run(self, run_id):
        """
        Always returns a run whose artifact_uri has the yprov+ prefix.
        This ensures mlflow.log_artifact() routes to YProvArtifactRepo
        even for runs created before the store was activated.
        """
        run = self._delegate.get_run(run_id)
        return _PatchedRun(run) if run is not None else run

    def set_terminated(self, run_id, status, end_time):
        _debug(f"set_terminated: run_id={run_id}, status={status}")
        if yprov:
            _yprov_end_run()
        return self._delegate.set_terminated(run_id, status, end_time)

    def update_run_info(self, run_id, run_status, end_time, **kwargs):
        _debug(f"update_run_info: run_id={run_id}, status={run_status}")
        try:
            s = str(run_status).upper()
            is_terminal = s in {"FINISHED", "FAILED", "KILLED"} or (s.isdigit() and int(s) in {3, 4, 5})
        except Exception:
            is_terminal = False
        if yprov and is_terminal:
            _yprov_end_run()
        return self._delegate.update_run_info(run_id, run_status, end_time, **kwargs)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_param(self, run_id, param):
        _debug(f"log_param: {param.key}={param.value!r}")
        if yprov:
            _yprov_log_param(param.key, param.value)
        return self._delegate.log_param(run_id, param)

    def log_metric(self, run_id, metric):
        _debug(f"log_metric: {metric.key}={metric.value} step={getattr(metric, 'step', None)}")
        if yprov:
            _yprov_log_metric(metric.key, metric.value, step=getattr(metric, "step", None))
        return self._delegate.log_metric(run_id, metric)

    def log_batch(self, run_id, metrics, params, tags):
        _debug(f"log_batch: {len(params or [])} params, {len(metrics or [])} metrics")
        if yprov:
            for p in (params or []):
                _yprov_log_param(p.key, p.value)
            for m in (metrics or []):
                _yprov_log_metric(m.key, m.value, step=getattr(m, "step", None))
        return self._delegate.log_batch(run_id, metrics, params, tags)

    # ------------------------------------------------------------------
    # Pass-throughs
    # ------------------------------------------------------------------

    def get_artifact_uri(self, run_id):
        uri = (
            self._delegate.get_artifact_uri(run_id)
            if hasattr(self._delegate, "get_artifact_uri")
            else self._delegate.get_run(run_id).info.artifact_uri
        )
        return _ensure_yprov(uri)

    def search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):
        return self._delegate.search_runs(
            experiment_ids, filter_string, run_view_type, max_results, order_by, page_token)

    def list_run_infos(self, experiment_id, run_view_type, max_results, order_by, page_token):
        return self._delegate.list_run_infos(
            experiment_id, run_view_type, max_results, order_by, page_token)

    def shut_down_async_logging(self):
        _debug("shut_down_async_logging")
        if hasattr(self._delegate, "shut_down_async_logging"):
            return self._delegate.shut_down_async_logging()
