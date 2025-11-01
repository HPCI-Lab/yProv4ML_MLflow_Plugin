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
    

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def create_run(self, experiment_id, user_id=None, start_time=None, tags=None, run_name=None, **kwargs):
        if yprov:
            try:
                yprov.start_run(run_name=run_name or (tags or {}).get('mlflow.runName'))
                for k, v in (tags or {}).items():
                    try:
                        yprov.log_param(f'mlflow.tag.{k}', v)
                    except Exception:
                        pass
            except Exception:
                pass
        return self._delegate.create_run(experiment_id, user_id, start_time, tags, run_name=run_name, **kwargs)

    def set_terminated(self, run_id, status, end_time):
        if yprov:
            try:
                yprov.end_run(create_graph=False)
            except Exception:
                pass
        if export_run_to_prov and end_time is not None:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                exp_id = client.get_run(run_id).info.experiment_id
                name = client.get_experiment(exp_id).name
                out = self._prov_out / name
                export_run_to_prov(run_id, out, client=client)
            except Exception:
                pass
        return self._delegate.set_terminated(run_id, status, end_time)

    def update_run_info(self, run_id, run_status, end_time, **kwargs):
        if yprov and str(run_status).upper() in {'FINISHED', 'FAILED', 'KILLED'}:
            try:
                yprov.end_run(create_graph=False)
            except Exception:
                pass
        if export_run_to_prov and end_time is not None:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                exp_id = client.get_run(run_id).info.experiment_id
                name = client.get_experiment(exp_id).name
                out = self._prov_out / name
                export_run_to_prov(run_id, out, client=client)
            except Exception:
                pass
        return self._delegate.update_run_info(run_id, run_status, end_time, **kwargs)

    def log_param(self, run_id, param):
        try:
            if yprov:
                yprov.log_param(param.key, str(param.value))
        finally:
            return self._delegate.log_param(run_id, param)

    def log_metric(self, run_id, metric):
        try:
            if yprov:
                step = getattr(metric, 'step', None)
                yprov.log_metric(metric.key, float(metric.value), step=step)
        finally:
            return self._delegate.log_metric(run_id, metric)

    def log_batch(self, run_id, metrics, params, tags):
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
        finally:
            return self._delegate.log_batch(run_id, metrics, params, tags)

    def get_artifact_uri(self, run_id):
        if hasattr(self._delegate, 'get_artifact_uri'):
            return self._delegate.get_artifact_uri(run_id)
        return self._delegate.get_run(run_id).info.artifact_uri

    def shut_down_async_logging(self):
        if hasattr(self._delegate, 'shut_down_async_logging'):
            return self._delegate.shut_down_async_logging()

    def create_experiment(self, name, artifact_location=None, tags=None):
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