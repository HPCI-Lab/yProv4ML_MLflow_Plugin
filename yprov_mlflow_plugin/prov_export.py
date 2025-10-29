from pathlib import Path
from datetime import datetime, timezone
from prov.model import ProvDocument
from mlflow.tracking import MlflowClient

def _ms_to_iso(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

def _list_all_artifacts(client: MlflowClient, run_id: str, path: str = ''):
    stack = [path]
    files = []
    while stack:
        cur = stack.pop()
        for it in client.list_artifacts(run_id, cur):
            if it.is_dir:
                stack.append(it.path)
            else:
                files.append(it)
    return files

def export_run_to_prov(run_id: str, out_dir: Path, client: MlflowClient | None = None) -> Path:
    client = client or MlflowClient()
    run = client.get_run(run_id)
    exp = client.get_experiment(run.info.experiment_id)

    d = ProvDocument()
    ns_ctx   = d.add_namespace('context', 'urn:context')
    ns_param = d.add_namespace('param',   'urn:param')
    ns_metric= d.add_namespace('metric',  'urn:metric')
    ns_art   = d.add_namespace('art',     'urn:artifact')
    ns_user  = d.add_namespace('user',    'urn:user')

    act = d.activity(ns_ctx[f"{exp.name}_{run.info.run_id}"])
    s = _ms_to_iso(run.info.start_time)
    e = _ms_to_iso(run.info.end_time)
    if s: act.add_attributes({ns_ctx['start']: s})
    if e: act.add_attributes({ns_ctx['end']: e})
    act.add_attributes({
        ns_ctx['status']: run.info.status,
        ns_ctx['run_id']: run.info.run_id,
        ns_ctx['experiment_id']: run.info.experiment_id,
        ns_ctx['experiment_name']: exp.name,
        ns_ctx['artifact_uri']: run.info.artifact_uri,
    })

    user = run.data.tags.get('mlflow.user') or 'unknown'
    ag = d.agent(ns_user[user])
    d.wasAssociatedWith(act, ag)

    for k, v in run.data.params.items():
        act.add_attributes({ns_param[k]: v})

    for k, v in run.data.metrics.items():
        act.add_attributes({ns_metric[k]: v})

    for k, v in run.data.tags.items():
        act.add_attributes({ns_param[f"mlflow.tag.{k}"]: v})

    arts = _list_all_artifacts(client, run.info.run_id)
    for a in arts:
        ent = d.entity(ns_art[a.path], {
            ns_ctx['artifact_path']: a.path,
            ns_ctx['artifact_size']: getattr(a, 'file_size', None),
        })
        d.wasGeneratedBy(ent, act)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run.info.run_id}.json"
    d.serialize(destination=str(out_path), format='json', indent=2)
    return out_path