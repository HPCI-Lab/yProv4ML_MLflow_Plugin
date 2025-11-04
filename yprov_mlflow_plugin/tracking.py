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

try:
    import prov4ml as yprov
except Exception:
    yprov = None

try:
    from .prov_export import export_run_to_prov
except Exception:
    export_run_to_prov = None

# Debug flag - set via environment variable
DEBUG = os.getenv('YPROV_DEBUG', '').lower() in ('1', 'true', 'yes')

def _debug(msg: str):
    if DEBUG:
        print(f"[yProv Plugin] {msg}", flush=True)

def _strip_yprov(uri: str) -> str:
    return uri.replace('yprov+', '', 1) if uri.startswith('yprov+') else uri

def _delegate_for(tracking_uri: str):
    base = _strip_yprov(tracking_uri)
    parsed = urlparse(base)

    if parsed.scheme in ('http', 'https'):
        from mlflow.store.tracking.rest_store import RestStore  # type: ignore
        return RestStore(base)

    if parsed.scheme in ('', 'file'):
        def _normalize_file_uri(raw: str) -> str:
            if raw.startswith('file://'):
                p = urlparse(raw).path
            else:
                p = raw
            m = re.match(r'^([A-Za-z]):[\/](.*)$', p)
            if m:
                drive = m.group(1).lower()
                rest = m.group(2).replace('\\', '/')
                if sys.platform.startswith('linux') and os.path.isdir('/mnt'):
                    return f'file:///mnt/{drive}/{rest}'
                return f'file:///{drive.upper()}:{"/" + rest if rest else ""}'
            m2 = re.match(r'^/[A-Za-z]:/.*', p)
            if m2:
                p2 = p.lstrip('/')
                drive = p2[0].lower()
                rest = p2[3:]
                if sys.platform.startswith('linux') and os.path.isdir('/mnt'):
                    return f'file:///mnt/{drive}/{rest.lstrip("/") }'
                return f'file:///{p2}'
            if p.startswith('/'):
                return f'file://{p}'
            abs_path = os.path.abspath(p)
            return f'file://{abs_path}'

        base_uri = _normalize_file_uri(base if parsed.scheme == 'file' else base)
        from mlflow.store.tracking.file_store import FileStore  # type: ignore
        return FileStore(base_uri)

    from mlflow.store.tracking.rest_store import RestStore  # type: ignore
    return RestStore(base)

class YProvTrackingStore(AbstractStore):
    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        super().__init__()
        self._store_uri = store_uri
        self._delegate = _delegate_for(store_uri)
        self._prov_out = Path(os.getenv('YPROV_OUT_DIR', 'data/prov'))
        _debug(f"🟢 YProvTrackingStore initialized with URI: {store_uri}")
        _debug(f"   Delegate type: {type(self._delegate).__name__}")
        _debug(f"   PROV output: {self._prov_out}")
    
    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        _debug(f"create_run called: exp={experiment_id}, run_name={run_name}")
        if yprov:
            try:
                yprov.start_run(run_name=run_name or (tags or {}).get('mlflow.runName'))
                _debug("  ✓ prov4ml.start_run() called")
                for k, v in (tags or {}).items():
                    try:
                        yprov.log_param(f'mlflow.tag.{k}', v)
                    except Exception as e:
                        _debug(f"  ⚠ Failed to log tag {k}: {e}")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.start_run() failed: {e}")
        else:
            _debug("  ⚠ prov4ml not available")
        
        result = self._delegate.create_run(experiment_id, user_id, start_time, tags, run_name=run_name, **kwargs)
        _debug(f"  Created run_id: {result.info.run_id}")
        return result

    def _export_prov(self, run_id: str):
        """Helper to export PROV document for a run."""
        if not export_run_to_prov:
            _debug(f"  ⚠ export_run_to_prov not available")
            return
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            exp_id = client.get_run(run_id).info.experiment_id
            name = client.get_experiment(exp_id).name
            out = self._prov_out / name
            out.mkdir(parents=True, exist_ok=True)
            prov_file = export_run_to_prov(run_id, out, client=client)
            _debug(f"  ✓ PROV exported to: {prov_file}")
        except Exception as e:
            _debug(f"  ⚠ PROV export failed: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()

    def set_terminated(self, run_id, status, end_time):
        _debug(f"set_terminated called: run_id={run_id}, status={status}")
        
        if yprov:
            try:
                yprov.end_run(create_graph=False)
                _debug("  ✓ prov4ml.end_run() called")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.end_run() failed: {e}")
        else:
            _debug("  ⚠ prov4ml not available")
        
        # Always try to export PROV document
        if end_time is not None:
            self._export_prov(run_id)
        
        return self._delegate.set_terminated(run_id, status, end_time)

    def update_run_info(self, run_id, run_status, end_time, **kwargs):
        _debug(f"update_run_info called: run_id={run_id}, status={run_status}")
        
        if yprov and str(run_status).upper() in {'FINISHED', 'FAILED', 'KILLED'}:
            try:
                yprov.end_run(create_graph=False)
                _debug("  ✓ prov4ml.end_run() called")
            except Exception as e:
                _debug(f"  ⚠ prov4ml.end_run() failed: {e}")
        else:
            if not yprov:
                _debug("  ⚠ prov4ml not available")
        
        # Always try to export PROV document for terminal states
        if end_time is not None and str(run_status).upper() in {'FINISHED', 'FAILED', 'KILLED'}:
            self._export_prov(run_id)
        
        return self._delegate.update_run_info(run_id, run_status, end_time, **kwargs)

    def log_param(self, run_id, param):
        _debug(f"log_param: {param.key}={param.value}")
        try:
            if yprov:
                yprov.log_param(param.key, str(param.value))
        except Exception as e:
            _debug(f"  ⚠ prov4ml.log_param failed: {e}")
        finally:
            return self._delegate.log_param(run_id, param)

    def log_metric(self, run_id, metric):
        if DEBUG and hasattr(self, '_metric_log_count'):
            self._metric_log_count = getattr(self, '_metric_log_count', 0) + 1
            if self._metric_log_count <= 3 or self._metric_log_count % 10 == 0:
                _debug(f"log_metric: {metric.key}={metric.value}")
        try:
            if yprov:
                step = getattr(metric, 'step', None)
                yprov.log_metric(metric.key, float(metric.value), step=step)
        except Exception as e:
            if DEBUG:
                _debug(f"  ⚠ prov4ml.log_metric failed: {e}")
        finally:
            return self._delegate.log_metric(run_id, metric)

    def log_batch(self, run_id, metrics, params, tags):
        _debug(f"log_batch: {len(params or [])} params, {len(metrics or [])} metrics, {len(tags or [])} tags")
        try:
            if yprov:
                for p in params or []:
                    yprov.log_param(p.key, str(p.value))
                for m in metrics or []:
                    yprov.log_metric(m.key, float(m.value), step=getattr(m, 'step', None))
                for t in tags or []:
                    try:
                        yprov.log_param(f'mlflow.tag.{t.key}', t.value)
                    except Exception:
                        pass
        except Exception as e:
            _debug(f"  ⚠ prov4ml.log_batch failed: {e}")
        finally:
            return self._delegate.log_batch(run_id, metrics, params, tags)

    def get_artifact_uri(self, run_id):
        if hasattr(self._delegate, 'get_artifact_uri'):
            return self._delegate.get_artifact_uri(run_id)
        return self._delegate.get_run(run_id).info.artifact_uri

    def shut_down_async_logging(self):
        _debug("shut_down_async_logging called")
        if hasattr(self._delegate, 'shut_down_async_logging'):
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