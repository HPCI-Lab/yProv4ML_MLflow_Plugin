"""
demo.py – Minimal end-to-end demo for the yProv4ML MLflow plugin.

Trains a small CNN on CIFAR-10 (or logs toy metrics when PyTorch is absent)
and records provenance via the plugin.

Usage
-----
    python examples/demo.py                       # default settings
    python examples/demo.py --epochs 10 --lr 5e-4
    python examples/demo.py --device cuda

Optional extras must be installed:
    pip install -e ".[examples]"
"""
import os
import time
import json
import csv
import random
import argparse
from pathlib import Path

import numpy as np

# --- Optional: PyTorch + torchvision ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# --- Optional: CodeCarbon (sustainability tracking) ---
try:
    from codecarbon import EmissionsTracker
    CODECARBON_OK = True
except ImportError:
    CODECARBON_OK = False

# --- Optional: psutil (system metrics) ---
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, out_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, 5), nn.ReLU(),
            nn.Conv2d(20, 64, 5), nn.ReLU(),
        )
        self.classifier = nn.Linear(36864, out_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class LRLogger:
    """CSV logger for the learning-rate schedule."""
    def __init__(self, out_csv: Path):
        self.path = out_csv
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lr_min, self.lr_max = float("inf"), float("-inf")
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["step_type", "index", "lr"])

    def log(self, step_type: str, index: int, optimizer):
        lr = float(optimizer.param_groups[0]["lr"])
        self.lr_min = min(self.lr_min, lr)
        self.lr_max = max(self.lr_max, lr)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([step_type, index, lr])


def _get_cifar10(root: str, train: bool, transform):
    cifar_root = os.path.join(root, "cifar-10-batches-py")
    download = not os.path.exists(cifar_root)
    return CIFAR10(root, train=train, download=download, transform=transform)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="yProv4ML MLflow Plugin demo")
    p.add_argument("--exp_name",     type=str,   default="demo_yprov_plugin")
    p.add_argument("--run_name",     type=str,   default="hello-yprov")
    p.add_argument("--data_dir",     type=str,   default="./data")
    p.add_argument("--artifact_dir", type=str,   default="./mlartifacts")
    p.add_argument("--device",       type=str,   default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seed",         type=int,   default=0)
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_OK:
        torch.manual_seed(args.seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Activate the yProv4ML plugin via yprov+ URI prefix
    mlruns_path = Path("./mlruns").resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"yprov+file://{mlruns_path}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔧 Tracking URI : {tracking_uri}")

    # Experiment setup
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    if client.get_experiment_by_name(args.exp_name) is None:
        client.create_experiment(args.exp_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(args.exp_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"🏃 Run ID: {run_id}")

        # --- Parameters ---
        mlflow.log_params({
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "epochs":     args.epochs,
            "seed":       args.seed,
        })

        # --- Config artifact ---
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True, parents=True)
        config_path = outputs_dir / "config.json"
        config_path.write_text(json.dumps({
            "device":     args.device,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "epochs":     args.epochs,
            "seed":       args.seed,
        }, indent=2))
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # --- Optional: carbon tracker ---
        tracker = None
        if CODECARBON_OK:
            try:
                tracker = EmissionsTracker(log_level="warning")
                tracker.start()
            except Exception:
                tracker = None

        # --- Training ---
        if TORCH_OK:
            device = torch.device(
                args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
            )
            tform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            os.makedirs(args.data_dir, exist_ok=True)

            train_ds = _get_cifar10(args.data_dir, train=True,  transform=tform)
            train_ds = Subset(train_ds, range(args.batch_size * 200))
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

            test_ds = _get_cifar10(args.data_dir, train=False, transform=tform)
            test_ds = Subset(test_ds, range(args.batch_size * 100))
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

            model  = CNN().to(device)
            optim  = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = nn.MSELoss()

            lr_csv  = outputs_dir / "lr_trace.csv"
            lrlog   = LRLogger(lr_csv)

            for epoch in range(args.epochs):
                t0 = time.time()
                model.train()
                last_batch_loss = None

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optim.zero_grad(set_to_none=True)
                    logits  = model(x)
                    y_hot   = F.one_hot(y, 10).float()
                    loss    = loss_fn(logits, y_hot)
                    loss.backward()
                    optim.step()
                    last_batch_loss = float(loss.item())

                model.eval()
                val_sum, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        val_sum += float(loss_fn(logits, F.one_hot(y, 10).float()).item())
                        correct += int((logits.argmax(1) == y).sum())
                        total   += int(y.size(0))

                avg_val = val_sum / max(1, len(test_loader))
                acc     = correct  / max(1, total)

                mlflow.log_metrics({
                    "train_epoch_time_ms": (time.time() - t0) * 1000.0,
                    "MSE_train":           last_batch_loss or float("nan"),
                    "MSE_val":             avg_val,
                    "ACC_val":             acc,
                }, step=epoch)

                if PSUTIL_OK:
                    mlflow.log_metrics({
                        "cpu_usage":    float(psutil.cpu_percent(interval=None)),
                        "memory_usage": float(psutil.virtual_memory().percent),
                        "disk_usage":   float(psutil.disk_usage("/").percent),
                    }, step=epoch)

                lrlog.log("epoch", epoch, optim)
                print(f"  Epoch {epoch:02d} | MSE_val={avg_val:.4f} | ACC_val={acc:.3f}")

            # --- Save model artifact ---
            model_path = outputs_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            if np.isfinite(lrlog.lr_min):
                mlflow.log_metric("lr_min", float(lrlog.lr_min))
            if np.isfinite(lrlog.lr_max):
                mlflow.log_metric("lr_max", float(lrlog.lr_max))
            mlflow.log_artifact(str(lr_csv), artifact_path="analysis")

        else:
            # Toy loop when PyTorch is not installed
            print("⚠️  PyTorch not found – logging toy metrics only.")
            mlflow.log_param("note", "PyTorch not available; toy metrics only")
            for step, (mse_train, mse_val, acc) in enumerate([
                (0.70, 0.68, 0.10),
                (0.42, 0.41, 0.25),
                (0.31, 0.30, 0.42),
                (0.27, 0.26, 0.55),
                (0.25, 0.24, 0.60),
            ], start=0):
                mlflow.log_metrics({
                    "MSE_train":           mse_train,
                    "MSE_val":             mse_val,
                    "ACC_val":             acc,
                    "train_epoch_time_ms": 1000.0,
                }, step=step)
                time.sleep(0.05)

            if PSUTIL_OK:
                mlflow.log_metrics({
                    "cpu_usage":    float(psutil.cpu_percent(interval=None)),
                    "memory_usage": float(psutil.virtual_memory().percent),
                    "disk_usage":   float(psutil.disk_usage("/").percent),
                })

        # --- Carbon footprint ---
        if tracker is not None:
            try:
                emissions = tracker.stop()
                if emissions is not None:
                    mlflow.log_metric("emissions_kgCO2e", float(emissions))

                    totals = getattr(tracker, "final_emissions_data", None) or \
                             getattr(tracker, "_emissions_data", None)
                    details = {}
                    if totals and hasattr(totals, "__dict__"):
                        details = {k: v for k, v in totals.__dict__.items() if not k.startswith("_")}
                    elif isinstance(totals, dict):
                        details = totals

                    _metric_map = {
                        "emissions_rate":      ["emissions_rate", "emissions_rate_kg_s"],
                        "energy_consumed_kwh": ["energy_consumed", "energy_kwh"],
                        "cpu_energy_kwh":      ["cpu_energy", "cpu_energy_kwh"],
                        "gpu_energy_kwh":      ["gpu_energy", "gpu_energy_kwh"],
                        "ram_energy_kwh":      ["ram_energy", "ram_energy_kwh"],
                        "cpu_power_watts":     ["cpu_average_power_watts"],
                        "gpu_power_watts":     ["gpu_average_power_watts"],
                        "ram_power_watts":     ["ram_power_watts"],
                    }
                    for mlflow_key, candidate_keys in _metric_map.items():
                        for ck in candidate_keys:
                            if ck in details and details[ck] is not None:
                                mlflow.log_metric(mlflow_key, float(details[ck]))
                                break
            except Exception as exc:
                print(f"⚠️  CodeCarbon error: {exc}")

        print(f"\n✅ Run {run_id} completed.")

    print(f"Artifacts : {artifact_dir}")
    print(f"MLflow UI : mlflow ui --backend-store-uri {tracking_uri}")
    print("PROV JSON : data/prov/<experiment>/ (generated by plugin)")


if __name__ == "__main__":
    main()
