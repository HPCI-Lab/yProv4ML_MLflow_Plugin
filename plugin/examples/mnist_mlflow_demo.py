#!/usr/bin/env python
"""
MNIST MLflow Demo with yProv4ML Integration (enhanced)
======================================================

This version ensures **all available** sustainability and system metrics are
captured and logged to MLflow, including robust CodeCarbon handling on WSL2.

Key upgrades:
- CodeCarbon tracker uses `tracking_mode="process"`, frequent sampling, and
  avoids file output (we log everything to MLflow and a JSON artifact).
- Defensive logging: only logs non-None metrics, prints what was found.
- Writes a `carbon_details.json` artifact with the full CodeCarbon payload.
- psutil metrics captured per epoch (and once in toy mode).
- Learning-rate trace persisted and min/max LRs logged.

Usage:
    python mnist_mlflow_demo.py --epochs 3 --lr 1e-3 --scheduler cosine
"""

import os
import sys
import time
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ============================================================================
# Optional PyTorch Dependencies
# ============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("⚠️  PyTorch not found. Install with: pip install torch torchvision")

# ============================================================================
# Optional Sustainability & System Metrics
# ============================================================================
try:
    from codecarbon import EmissionsTracker
    CODECARBON_OK = True
except ImportError:
    CODECARBON_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

# ============================================================================
# MLflow (Required)
# ============================================================================
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("❌ MLflow is required. Install with: pip install mlflow")
    sys.exit(1)


# ============================================================================
# Neural Network Architecture
# ============================================================================
class CNN(nn.Module):
    """
    Simple CNN for MNIST (1x28x28 grayscale images).
    """
    def __init__(self, out_classes: int = 10):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),   # 28->24
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),  # 24->20
            nn.ReLU(),
        )
        self.mlp = nn.Linear(64 * 20 * 20, out_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)


# ============================================================================
# Dataset Utilities
# ============================================================================
from torch.utils.data import Subset

def load_mnist_dataset(root: str, train: bool, transform) -> MNIST:
    processed_dir = os.path.join(root, "MNIST", "processed")
    expected_file = "training.pt" if train else "test.pt"
    download = not os.path.exists(os.path.join(processed_dir, expected_file))
    if download:
        print(f"📥 Downloading MNIST {'training' if train else 'test'} set...")
    return MNIST(root, train=train, download=download, transform=transform)


def _subset_first_n(ds, n: int | None):
    if n is None:
        return ds
    n = int(n)
    if n <= 0:
        return Subset(ds, [])
    n = min(n, len(ds))
    if n == len(ds):
        return ds
    return Subset(ds, range(n))


# ============================================================================
# Learning Rate Logger
# ============================================================================
class LRLogger:
    def __init__(self, csv_path: Path):
        self.path = csv_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lr_min = float("inf")
        self.lr_max = float("-inf")
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["step_type", "index", "lr"])

    def log(self, step_type: str, index: int, optimizer):
        lr = float(optimizer.param_groups[0]["lr"])
        self.lr_min = min(self.lr_min, lr)
        self.lr_max = max(self.lr_max, lr)
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([step_type, index, lr])
        mlflow.log_metric("lr", lr, step=int(index))


# ============================================================================
# Learning Rate Scheduler Factory
# ============================================================================

    def create_scheduler(optimizer, args, steps_per_epoch: Optional[int] = None) -> Tuple[Optional[object], str]:
        if args.scheduler == "none":
            return None, "epoch"

        if args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
            )
            return scheduler, "epoch"

        if args.scheduler == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_gamma
            )
            return scheduler, "epoch"

        if args.scheduler == "cosine":
            T_max = args.t_max if args.t_max is not None else args.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
            return scheduler, "epoch"

        if args.scheduler == "onecycle":
            if steps_per_epoch is None:
                raise ValueError("OneCycleLR requires steps_per_epoch")
            max_lr = args.max_lr if args.max_lr is not None else args.lr * 10.0
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=args.epochs,
                pct_start=args.pct_start,
            )
            return scheduler, "batch"

        if args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=args.lr_gamma, patience=args.plateau_patience
            )
            return scheduler, "plateau"

        raise ValueError(f"Unknown scheduler: {args.scheduler}")


# ============================================================================
# Training Loop
# ============================================================================


    def train_mnist(args):
        device = torch.device(
            args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        print(f"🖥️  Using device: {device}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
        ])

        os.makedirs(args.data_dir, exist_ok=True)
        train_dataset = load_mnist_dataset(args.data_dir, train=True,  transform=transform)
        test_dataset  = load_mnist_dataset(args.data_dir, train=False, transform=transform)

        # Clamp requested subset sizes to the dataset length
        train_n = args.batch_size * 200
        test_n  = args.batch_size * 100

        train_subset = _subset_first_n(train_dataset, train_n)
        test_subset  = _subset_first_n(test_dataset,  test_n)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=False)

        print(f"📊 Training samples: {len(train_subset)}")
        print(f"📊 Test samples: {len(test_subset)}")

        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        steps_per_epoch = len(train_loader)
        scheduler, schedule_mode = create_scheduler(optimizer, args, steps_per_epoch)
        if scheduler:
            print(f"📈 LR Scheduler: {args.scheduler} (mode: {schedule_mode})")

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        lr_csv = outputs_dir / "lr_trace.csv"
        lr_logger = LRLogger(lr_csv)

        global_step = 0
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            last_batch_loss = None

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                y_onehot = F.one_hot(y, num_classes=10).float()
                loss = loss_fn(logits, y_onehot)
                loss.backward()
                optimizer.step()

                last_batch_loss = float(loss.item())
                global_step += 1

                if schedule_mode == "batch" and scheduler is not None:
                    scheduler.step()
                    lr_logger.log("batch", global_step, optimizer)

            if schedule_mode == "epoch" and scheduler is not None:
                scheduler.step()
                lr_logger.log("epoch", epoch, optimizer)

            model.eval()
            val_loss_sum = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    y_onehot = F.one_hot(y, num_classes=10).float()
                    val_loss = loss_fn(logits, y_onehot)
                    val_loss_sum += float(val_loss.item())
                    pred = logits.argmax(dim=1)
                    correct += int((pred == y).sum().item())
                    total += int(y.size(0))

            avg_val_loss = val_loss_sum / max(1, len(test_loader))
            accuracy = correct / max(1, total)

            if schedule_mode == "plateau" and scheduler is not None:
                scheduler.step(avg_val_loss)
                lr_logger.log("epoch", epoch, optimizer)

            epoch_time_ms = (time.time() - epoch_start) * 1000.0
            mlflow.log_metric("train_epoch_time_ms", epoch_time_ms, step=epoch)
            mlflow.log_metric("MSE_train", last_batch_loss if last_batch_loss is not None else np.nan, step=epoch)
            mlflow.log_metric("MSE_val", avg_val_loss, step=epoch)
            mlflow.log_metric("ACC_val", accuracy, step=epoch)

            if PSUTIL_OK:
                try:
                    mlflow.log_metric("cpu_usage", float(psutil.cpu_percent(interval=None)), step=epoch)
                    vm = psutil.virtual_memory()
                    mlflow.log_metric("memory_usage", float(vm.percent), step=epoch)
                    du = psutil.disk_usage("/")
                    mlflow.log_metric("disk_usage", float(du.percent), step=epoch)
                except Exception:
                    pass

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {last_batch_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Time: {epoch_time_ms:.0f}ms"
            )

        model_path = outputs_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        if np.isfinite(lr_logger.lr_min):
            mlflow.log_metric("lr_min", float(lr_logger.lr_min))
        if np.isfinite(lr_logger.lr_max):
            mlflow.log_metric("lr_max", float(lr_logger.lr_max))

        mlflow.log_artifact(str(lr_csv), artifact_path="analysis")
        print(f"✅ Training complete. Model saved to {model_path}")


# ============================================================================
# Toy Training Loop (No PyTorch)
# ============================================================================

    def train_toy_mode(args):
        print("⚠️  Running in toy mode (PyTorch not available)")
        mlflow.log_param("note", "Torch not available; logging toy metrics only")
        for step, loss in enumerate([0.7, 0.42, 0.31, 0.27, 0.25], start=1):
            mlflow.log_metric("MSE_train", loss, step=step)
            time.sleep(0.05)
        mlflow.log_metric("MSE_val", 0.24)
        mlflow.log_metric("ACC_val", 0.10)
        mlflow.log_metric("train_epoch_time_ms", 1000.0)
        if PSUTIL_OK:
            mlflow.log_metric("cpu_usage", float(psutil.cpu_percent(interval=None)))
            vm = psutil.virtual_memory()
            mlflow.log_metric("memory_usage", float(vm.percent))
            du = psutil.disk_usage("/")
            mlflow.log_metric("disk_usage", float(du.percent))
        print("✅ Toy training complete")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MNIST classification with MLflow + yProv4ML tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--exp_name", type=str, default="mnist_yprov_plugin")
    parser.add_argument("--run_name", type=str, default="mnist-demo")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--artifact_dir", type=str, default="./mlartifacts")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "exp", "cosine", "onecycle", "plateau"])
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--plateau_patience", type=int, default=3)
    parser.add_argument("--run_id", type=int, default=None)

    args = parser.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_OK:
        torch.manual_seed(args.seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Activate yProv MLflow plugin
    mlruns_path = Path("./mlruns").resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"yprov+file://{mlruns_path}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔧 Using tracking URI: {tracking_uri}")

    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifact_dir.as_uri()

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.exp_name)
    if exp is None:
        client.create_experiment(args.exp_name, artifact_location=artifact_uri)
        exp = client.get_experiment_by_name(args.exp_name)
    mlflow.set_experiment(args.exp_name)

    run_display_name = (
        f"{args.run_name}-bs{args.batch_size}-lr{args.lr}-seed{args.seed}-sch{args.scheduler}"
    )
    if args.run_id is not None:
        run_display_name += f"-id{args.run_id}"

    with mlflow.start_run(run_name=run_display_name) as run:
        run_id = run.info.run_id
        print(f"🚀 Started MLflow run: {run_id}")
        print(f"📝 Experiment: {args.exp_name}")

        # Parameters
        mlflow.log_param("param_batch_size", args.batch_size)
        mlflow.log_param("param_lr", args.lr)
        mlflow.log_param("param_epochs", args.epochs)
        mlflow.log_param("param_seed", args.seed)
        mlflow.log_param("scheduler", args.scheduler)
        mlflow.log_param("lr_step_size", args.lr_step_size)
        mlflow.log_param("lr_gamma", args.lr_gamma)
        mlflow.log_param("t_max", args.t_max)
        mlflow.log_param("max_lr", args.max_lr)
        mlflow.log_param("pct_start", args.pct_start)
        mlflow.log_param("plateau_patience", args.plateau_patience)

        # Save config
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True, parents=True)
        config_path = outputs_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "device": args.device,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "seed": args.seed,
                "scheduler": args.scheduler,
            }, f, indent=2)
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # ---------------- CodeCarbon tracking (enhanced) ----------------
        tracker = None
        if CODECARBON_OK:
            try:
                tracker = EmissionsTracker(
                    log_level="warning",
                    tracking_mode="process",   # robust on WSL2
                    measure_power_secs=1,
                    save_to_file=False,
                    gpu_ids="all",
                )
                tracker.start()
                print("🌱 Carbon tracking enabled (CodeCarbon)")
            except Exception as e:
                print(f"⚠️  CodeCarbon initialization failed: {e}")
                tracker = None

        # ---------------- Training / Toy ----------------
        if TORCH_OK:
            train_mnist(args)
        else:
            train_toy_mode(args)

        # ---------------- Log CodeCarbon totals ----------------
        if tracker is not None:
            try:
                emissions = tracker.stop()  # kg CO2e (float)

                totals = (
                    getattr(tracker, "final_emissions_data", None)
                    or getattr(tracker, "_emissions_data", None)
                )
                details = {}
                if totals and hasattr(totals, "__dict__"):
                    details = {k: v for k, v in totals.__dict__.items() if not k.startswith("_")}
                elif isinstance(totals, dict):
                    details = totals

                # Persist full payload as artifact for provenance/debugging
                carbon_json = outputs_dir / "carbon_details.json"
                with open(carbon_json, "w") as f:
                    json.dump(details, f, indent=2, default=float)
                mlflow.log_artifact(str(carbon_json), artifact_path="carbon")

                def log_if_present(metric_name: str, *keys):
                    # Find first present key, cast to float, log metric
                    for k in keys:
                        if k in details and details[k] is not None:
                            try:
                                mlflow.log_metric(metric_name, float(details[k]))
                                return True
                            except Exception:
                                pass
                    return False

                # Always log total emissions (returned by stop())
                if emissions is not None:
                    mlflow.log_metric("emissions", float(emissions))

                # Try log the rest (skip silently if not available on the host)
                log_if_present("emissions_rate", "emissions_rate", "emissions_rate_kg_s")
                log_if_present("energy_consumed", "energy_consumed", "energy_kwh")
                log_if_present("cpu_energy", "cpu_energy", "cpu_energy_kwh")
                log_if_present("gpu_energy", "gpu_energy", "gpu_energy_kwh")
                log_if_present("ram_energy", "ram_energy", "ram_energy_kwh")
                log_if_present("cpu_power", "cpu_power", "cpu_average_power_watts")
                log_if_present("gpu_power", "gpu_power", "gpu_average_power_watts")
                log_if_present("ram_power", "ram_power", "ram_power_watts")

                # Log some environment metadata as params (useful context)
                for pkey in [
                    "country_name", "country_iso_code", "region", "os",
                    "python_version", "codecarbon_version", "cpu_count",
                    "cpu_model", "gpu_count", "gpu_model", "pue", "tracking_mode",
                ]:
                    if pkey in details and details[pkey] is not None:
                        try:
                            mlflow.log_param(pkey, str(details[pkey]))
                        except Exception:
                            pass

                print(f"🌱 Carbon emissions logged: {emissions if emissions is not None else 'n/a'} kg CO2e")
            except Exception as e:
                print(f"⚠️  Failed to log carbon metrics: {e}")

    mlflow.end_run()

    print("\n" + "=" * 60)
    print("✅ Run finished successfully")
    print(f"📁 Artifacts: {artifact_dir}")
    print(f"🔗 MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
