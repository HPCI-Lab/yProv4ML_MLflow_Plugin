"""
Demo: MLflow + yProv4ML plugin
===============================
Shows that every standard mlflow call automatically produces provenance data.
Run with:  python examples/demo.py
"""
import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Optional heavy deps
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",   default="demo_yprov_plugin")
    p.add_argument("--run_name",   default="hello-yprov")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    # -----------------------------------------------------------------------
    # CRITICAL: set tracking URI with "yprov+" prefix to activate the plugin
    # -----------------------------------------------------------------------
    mlruns_path = Path("./mlruns").resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"yprov+file://{mlruns_path}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI : {tracking_uri}")
    print(f"Provenance   : {Path(os.getenv('YPROV_OUT_DIR', 'data/prov')).resolve()}")
    print()

    # Create / reuse experiment
    client = MlflowClient()
    exp    = client.get_experiment_by_name(args.exp_name)
    if exp is None:
        client.create_experiment(args.exp_name)
    mlflow.set_experiment(args.exp_name)

    # -----------------------------------------------------------------------
    # Every call below is 100% standard MLflow.
    # The plugin intercepts each call and forwards it to yprov4ml.
    # -----------------------------------------------------------------------
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow run   : {run_id}")

        # --- Parameters (mlflow.log_param → yprov4ml.log_param) ---
        mlflow.log_param("lr",         args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs",     args.epochs)
        mlflow.log_param("seed",       args.seed)

        # --- Config artifact ---
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        cfg = out / "config.json"
        cfg.write_text(json.dumps(vars(args), indent=2))
        mlflow.log_artifact(str(cfg), artifact_path="config")

        # --- Training loop (mlflow.log_metric → yprov4ml.log_metric) ---
        np.random.seed(args.seed)
        for epoch in range(args.epochs):
            t0   = time.time()
            loss = float(np.exp(-0.3 * epoch) + np.random.uniform(0, 0.05))
            acc  = float(1 - np.exp(-0.4 * epoch) + np.random.uniform(0, 0.02))

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_accuracy", acc,  step=epoch)
            mlflow.log_metric("epoch_time_ms", (time.time() - t0) * 1000, step=epoch)

            if PSUTIL_OK:
                mlflow.log_metric("cpu_pct",    float(psutil.cpu_percent()),         step=epoch)
                mlflow.log_metric("mem_pct",    float(psutil.virtual_memory().percent), step=epoch)

            print(f"  epoch {epoch+1}/{args.epochs}  loss={loss:.4f}  acc={acc:.4f}")

        print(f"\n✅ Run {run_id} complete.")
        print(f"   MLflow data  : {mlruns_path}")
        print(f"   Provenance   : {Path(os.getenv('YPROV_OUT_DIR', 'data/prov')).resolve()}")
        print(f"   MLflow UI    : mlflow ui --backend-store-uri {tracking_uri}")


if __name__ == "__main__":
    main()
