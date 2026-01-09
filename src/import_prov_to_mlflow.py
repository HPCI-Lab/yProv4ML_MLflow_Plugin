#!/usr/bin/env python3
"""
Import yProv4ML provenance runs (prov JSON + metrics) as MLflow runs,
so you can view them in the MLflow UI.

Assumes structure like:

data/prov/usecase/
  <run_dir>/                    # e.g. 1e-05_5_64_0.5_small_0
    artifacts/
    metrics/   or metrics_GR0/
    prov_*.json
"""

import os
import argparse
import json
import glob

import mlflow

# optional: xarray to read .nc files
try:
    import xarray as xr
    _HAS_XARRAY = True
except Exception:
    _HAS_XARRAY = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prov-root",
        type=str,
        required=True,
        help="Directory that contains per-run folders (each with metrics/ and prov_*.json)",
    )
    p.add_argument(
        "--experiment-name",
        type=str,
        default="usecase_import",
        help="MLflow experiment name to import into",
    )
    return p.parse_args()


def _safe_cast(val, cast_fn):
    """Try to cast val; if it fails, just return original."""
    # Flatten {"$": X, "type": "..."} pattern
    if isinstance(val, dict) and "$" in val:
        val = val["$"]
    try:
        return cast_fn(val)
    except Exception:
        return val


def read_params_from_prov_json(prov_path: str, run_dir_name: str):
    """
    Extract params from:
      - the run directory name (LR, EPOCHS, BATCH_SIZE, DROPOUT, MODEL_SIZE, RUN_IDX)
      - the PROV JSON 'activity' section (TRAINING hyperparams + TESTING accuracy)

    Expected dir pattern:
        LR_EPOCHS_BATCHSIZE_DROPOUT_MODEL_SIZE_RUNIDX
    e.g.:
        1e-05_5_64_0.5_small_0
        0.001_5_16_0.1_large_0
    """
    # baseline: always store which prov file we used
    params = {
        "prov_file": os.path.basename(prov_path)
    }

    # ---------- 1) Parse hyperparams from run_dir_name ----------
    base = os.path.basename(run_dir_name)  # e.g. "1e-05_5_64_0.5_small_0"
    parts = base.split("_")

    if len(parts) == 6:
        lr_str, epochs_str, bs_str, dropout_str, model_size, run_idx_str = parts

        params["lr"] = _safe_cast(lr_str, float)
        params["epochs"] = _safe_cast(epochs_str, int)
        params["batch_size"] = _safe_cast(bs_str, int)
        params["dropout"] = _safe_cast(dropout_str, float)
        params["model_size"] = model_size
        params["run_idx"] = _safe_cast(run_idx_str, int)
    else:
        print(f"⚠️ Could not parse hyperparams from run_dir_name='{base}' (got {len(parts)} parts)")

    # ---------- 2) Read more precise info from PROV JSON ----------
    try:
        with open(prov_path, "r") as f:
            prov = json.load(f)
    except Exception as e:
        print(f"⚠️ Could not parse JSON in {prov_path}: {e}")
        return params

    activity = prov.get("activity", {})

    # TRAINING context: MODEL_SIZE, DROPOUT, BATCH_SIZE, EPOCHS, LR
    training_ctx = activity.get("context:Context.TRAINING", {})
    if training_ctx:
        # mapping: JSON key -> (param_name, cast_fn)
        mapping = {
            "MODEL_SIZE": ("model_size", str),
            "DROPOUT": ("dropout", float),
            "BATCH_SIZE": ("batch_size", int),
            "EPOCHS": ("epochs", int),
            "LR": ("lr", float),
        }
        for json_key, (param_name, fn) in mapping.items():
            if json_key in training_ctx:
                val = training_ctx[json_key]
                params[param_name] = _safe_cast(val, fn)

    # TESTING context: single accuracy value
    testing_ctx = activity.get("context:Context.TESTING", {})
    if testing_ctx and "accuracy" in testing_ctx:
        acc_val = testing_ctx["accuracy"]
        params["accuracy"] = _safe_cast(acc_val, float)

    return params


def log_nc_metrics(metrics_dir: str):
    """
    Log each *_Context...nc file as a single scalar metric (last value in 'values').

    - Metric name is taken from ds.attrs["_name"] when available (e.g. 'cpu_energy').
    - Fallback: file name prefix before '_Context'.
    - Value is taken from the 'values' variable, last time step.
    """
    if not _HAS_XARRAY:
        print("⚠️ xarray not available; skipping detailed .nc metric import.")
        return

    nc_files = sorted(glob.glob(os.path.join(metrics_dir, "*.nc")))
    if not nc_files:
        print(f"⚠️ No .nc files found under {metrics_dir}")
        return

    for nc_path in nc_files:
        base = os.path.basename(nc_path)

        try:
            with xr.open_dataset(nc_path) as ds:
                # metric name
                metric_name = ds.attrs.get("_name")
                if not metric_name:
                    metric_name = base.split("_Context")[0]

                # value from 'values'
                if "values" not in ds:
                    print(f"⚠️ No 'values' variable in {base}, skipping")
                    continue

                values = ds["values"].values
                if values.size == 0:
                    print(f"⚠️ Empty 'values' in {base}, skipping")
                    continue

                value = float(values[-1])

        except Exception as e:
            print(f"⚠️ Could not open or parse {nc_path}: {e}")
            continue

        mlflow.log_metric(metric_name, value, step=0)
        print(f"  • logged metric {metric_name} = {value} from {base}")


def main():
    args = parse_args()

    # Same backend store as your yProv+MLflow setup (without the yprov+ scheme)
    mlflow.set_tracking_uri(
        "file:///mnt/c/Users/admin/Documents/GitHub/yProv4ML_MLflow_Plugin/mlruns"
    )
    mlflow.set_experiment(args.experiment_name)

    run_root = args.prov_root

    for run_dir_name in sorted(os.listdir(run_root)):
        run_dir = os.path.join(run_root, run_dir_name)
        if not os.path.isdir(run_dir):
            continue

        print(f"\n=== Importing {run_dir_name} ===")

        # find prov_*.json
        prov_jsons = glob.glob(os.path.join(run_dir, "prov_*.json"))
        prov_json = prov_jsons[0] if prov_jsons else None

        # support both metrics/ and metrics_GR0/
        metrics_dir = None
        for candidate in ["metrics", "metrics_GR0"]:
            cand_path = os.path.join(run_dir, candidate)
            if os.path.isdir(cand_path):
                metrics_dir = cand_path
                break

        with mlflow.start_run(run_name=run_dir_name):
            # log params & prov file as artifact
            if prov_json:
                params = read_params_from_prov_json(
                    prov_json, run_dir_name=run_dir_name
                )
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # If we have accuracy param, also log as metric
                acc = params.get("accuracy")
                if acc is not None:
                    try:
                        mlflow.log_metric("accuracy", float(acc), step=0)
                    except (TypeError, ValueError):
                        print(f"⚠️ Could not cast accuracy={acc} to float for metrics")

                mlflow.log_artifact(prov_json, artifact_path="prov")
            else:
                print("⚠️ No prov_*.json found in", run_dir)

            # log metrics from .nc files if available (carbon/system metrics)
            if metrics_dir:
                log_nc_metrics(metrics_dir)
            else:
                print("⚠️ No metrics/ or metrics_GR0/ folder in", run_dir)

            # optional: log whole run dir as raw_prov
            # mlflow.log_artifacts(run_dir, artifact_path="raw_prov")

        print(f"✓ Imported {run_dir_name} into MLflow")


if __name__ == "__main__":
    main()
