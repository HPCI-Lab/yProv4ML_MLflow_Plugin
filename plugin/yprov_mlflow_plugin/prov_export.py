"""
prov_export.py – Convert a finished MLflow run into a W3C PROV JSON document.

This module is intentionally standalone: it reads data directly from the
MLflow FileStore so it can be used without the tracking plugin being active
(e.g. to re-export historical runs).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def export_run_to_prov(
    run_id: str,
    tracking_uri: Optional[str] = None,
    out_dir: Optional[str] = None,
    user_namespace: str = "yProv4ML",
) -> Optional[Path]:
    """Export an MLflow run to a W3C PROV-compliant JSON file.

    Parameters
    ----------
    run_id:
        The MLflow run ID to export.
    tracking_uri:
        MLflow tracking URI (defaults to ``MLFLOW_TRACKING_URI`` env var or
        ``"mlruns"``).
    out_dir:
        Directory where the PROV JSON will be written.  Defaults to
        ``YPROV_OUT_DIR`` env var or ``"data/prov"``.
    user_namespace:
        PROV namespace prefix (e.g. your organisation or project name).

    Returns
    -------
    Path to the generated JSON file, or ``None`` if the export failed.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        raise ImportError("mlflow is required for prov_export. Install it with: pip install mlflow")

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    client = MlflowClient(tracking_uri=uri)

    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise ValueError(f"Could not retrieve run '{run_id}' from '{uri}': {exc}") from exc

    exp = client.get_experiment(run.info.experiment_id)
    exp_name = exp.name if exp else "unknown"

    prov_dir = Path(out_dir or os.getenv("YPROV_OUT_DIR", "data/prov")) / exp_name
    prov_dir.mkdir(parents=True, exist_ok=True)

    doc = _build_prov_document(run, exp_name, user_namespace, client)

    out_path = prov_dir / f"prov_{run_id}.json"
    out_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _ts(epoch_ms: Optional[int]) -> Optional[str]:
    """Convert an MLflow epoch-millisecond timestamp to ISO-8601."""
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).isoformat()


def _build_prov_document(run, exp_name: str, namespace: str, client) -> Dict[str, Any]:
    """Construct a minimal W3C PROV-JSON document from an MLflow Run object."""

    run_id = run.info.run_id
    ns_prefix = f"{namespace}:"

    # -- Entities ------------------------------------------------------------
    entities: Dict[str, Any] = {}

    # One entity per param
    for key, value in (run.data.params or {}).items():
        eid = f"{ns_prefix}param_{key}"
        entities[eid] = {
            "prov:type": "prov:Entity",
            f"{ns_prefix}name": key,
            f"{ns_prefix}value": str(value),
            f"{ns_prefix}kind": "parameter",
        }

    # One entity per metric (last value)
    for key, value in (run.data.metrics or {}).items():
        eid = f"{ns_prefix}metric_{key}"
        entities[eid] = {
            "prov:type": "prov:Entity",
            f"{ns_prefix}name": key,
            f"{ns_prefix}value": value,
            f"{ns_prefix}kind": "metric",
        }

    # Artifacts
    try:
        artifacts = client.list_artifacts(run_id)
    except Exception:
        artifacts = []

    for art in artifacts:
        eid = f"{ns_prefix}artifact_{art.path.replace('/', '_')}"
        entities[eid] = {
            "prov:type": "prov:Entity",
            f"{ns_prefix}path": art.path,
            f"{ns_prefix}is_dir": art.is_dir,
            f"{ns_prefix}file_size": getattr(art, "file_size", None),
            f"{ns_prefix}kind": "artifact",
        }

    # -- Activities ----------------------------------------------------------
    activity_id = f"{ns_prefix}run_{run_id}"
    activities: Dict[str, Any] = {
        activity_id: {
            "prov:type": "prov:Activity",
            f"{ns_prefix}run_id": run_id,
            f"{ns_prefix}experiment": exp_name,
            f"{ns_prefix}run_name": run.info.run_name or "",
            f"{ns_prefix}status": run.info.status,
            "prov:startTime": _ts(run.info.start_time),
            "prov:endTime": _ts(run.info.end_time),
        }
    }

    # Add hyperparams as attributes on the activity too (convenient for queries)
    for key, value in (run.data.params or {}).items():
        activities[activity_id][f"{ns_prefix}param_{key}"] = value

    # -- Agents --------------------------------------------------------------
    user = run.info.user_id or "unknown"
    agent_id = f"{ns_prefix}agent_{user.replace(' ', '_')}"
    agents: Dict[str, Any] = {
        agent_id: {
            "prov:type": "prov:Agent",
            f"{ns_prefix}user_id": user,
        }
    }

    # -- Relations -----------------------------------------------------------
    was_associated_with: List[Dict[str, str]] = [
        {"prov:activity": activity_id, "prov:agent": agent_id}
    ]
    was_generated_by: List[Dict[str, str]] = [
        {"prov:entity": eid, "prov:activity": activity_id}
        for eid in entities
    ]
    used: List[Dict[str, str]] = [
        {"prov:activity": activity_id, "prov:entity": eid}
        for eid in entities
        if ":param_" in eid
    ]

    return {
        "prefix": {
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "prov": "http://www.w3.org/ns/prov#",
            namespace: f"https://w3id.org/prov4ml/{namespace}#",
        },
        "entity": entities,
        "activity": activities,
        "agent": agents,
        "wasAssociatedWith": was_associated_with,
        "wasGeneratedBy": was_generated_by,
        "used": used,
        # Metadata
        "_meta": {
            "run_id": run_id,
            "experiment": exp_name,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "schema": "W3C PROV-JSON 1.0",
        },
    }
