# examples/demo.py
import os, time, json, csv, random, argparse
from pathlib import Path
import numpy as np

# --- Optional dependencies (PyTorch / datasets) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# --- Optional CodeCarbon + psutil for sustainability/system metrics ---
try:
    from codecarbon import EmissionsTracker
    CODECARBON_OK = True
except Exception:
    CODECARBON_OK = False

try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False

import mlflow
from mlflow.tracking import MlflowClient

# ---- Optional PROV exporter (we’ll call this explicitly at run end) ----
try:
    from yprov_mlflow_plugin.prov_export import export_run_to_prov
except Exception:
    export_run_to_prov = None


# ----------------- Small CNN -----------------
class CNN(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU(),
        )
        self.mlp = nn.Linear(36864, out_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)


def _dataset(root: str, train: bool, tform):
    cifar_root = os.path.join(root, "cifar-10-batches-py")
    download = not os.path.exists(cifar_root)
    return CIFAR10(root, train=train, download=download, transform=tform)


class LRLogger:
    """Append-only CSV logger for learning rate trace + track min/max."""
    def __init__(self, out_csv: Path):
        self.path = out_csv
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


def main():
    p = argparse.ArgumentParser(description="Demo app: MLflow + (optional) CodeCarbon + system metrics")
    p.add_argument("--exp_name", type=str, default="demo_yprov_plugin")
    p.add_argument("--run_name", type=str, default="hello-yprov")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--artifact_dir", type=str, default="./mlartifacts")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_id", type=int, default=None)
    args = p.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_OK:
        torch.manual_seed(args.seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Prepare artifact location & experiment
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifact_dir.as_uri()

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.exp_name)
    if exp is None:
        client.create_experiment(args.exp_name, artifact_location=artifact_uri)
        exp = client.get_experiment_by_name(args.exp_name)
    mlflow.set_experiment(args.exp_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id

        # Params
        mlflow.log_param("param_batch_size", args.batch_size)
        mlflow.log_param("param_lr", args.lr)
        mlflow.log_param("param_epochs", args.epochs)
        mlflow.log_param("param_seed", args.seed)

        # Config artifact
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
            }, f)
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # CodeCarbon (optional)
        tracker = None
        if CODECARBON_OK:
            try:
                tracker = EmissionsTracker(log_level="warning")
                tracker.start()
            except Exception:
                tracker = None

        # Training or toy loop
        if TORCH_OK:
            device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

            tform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ])
            os.makedirs(args.data_dir, exist_ok=True)
            train_ds = _dataset(args.data_dir, train=True, tform=tform)
            train_ds = Subset(train_ds, range(args.batch_size * 200))
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

            test_ds = _dataset(args.data_dir, train=False, tform=tform)
            test_ds = Subset(test_ds, range(args.batch_size * 100))
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

            model = CNN().to(device)
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = nn.MSELoss().to(device)

            lr_csv = outputs_dir / "lr_trace.csv"
            lrlog = LRLogger(lr_csv)

            for epoch in range(args.epochs):
                t0 = time.time()
                model.train()
                last_batch_loss = None
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optim.zero_grad(set_to_none=True)
                    logits = model(x)
                    y_onehot = F.one_hot(y, 10).float()
                    loss = loss_fn(logits, y_onehot)
                    loss.backward()
                    optim.step()
                    last_batch_loss = float(loss.item())

                # Validation
                model.eval()
                val_sum, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        y_onehot = F.one_hot(y, 10).float()
                        vloss = loss_fn(logits, y_onehot)
                        val_sum += float(vloss.item())
                        pred = logits.argmax(1)
                        correct += int((pred == y).sum().item())
                        total += int(y.size(0))

                avg_val = val_sum / max(1, len(test_loader))
                acc = correct / max(1, total)

                mlflow.log_metric("train_epoch_time_ms", (time.time() - t0) * 1000.0, step=epoch)
                mlflow.log_metric("MSE_train", last_batch_loss if last_batch_loss is not None else np.nan, step=epoch)
                mlflow.log_metric("MSE_val", avg_val, step=epoch)
                mlflow.log_metric("ACC_val", acc, step=epoch)

                lrlog.log("epoch", epoch, optim)

                if PSUTIL_OK:
                    try:
                        mlflow.log_metric("cpu_usage", float(psutil.cpu_percent(interval=None)), step=epoch)
                        vm = psutil.virtual_memory()
                        mlflow.log_metric("memory_usage", float(vm.percent), step=epoch)
                        du = psutil.disk_usage("/")
                        mlflow.log_metric("disk_usage", float(du.percent), step=epoch)
                    except Exception:
                        pass

            # Artifacts
            model_path = outputs_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            if np.isfinite(lrlog.lr_min): mlflow.log_metric("lr_min", float(lrlog.lr_min))
            if np.isfinite(lrlog.lr_max): mlflow.log_metric("lr_max", float(lrlog.lr_max))
            mlflow.log_artifact(str(lr_csv), artifact_path="analysis")

        else:
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

        # CodeCarbon totals (optional)
        if tracker is not None:
            try:
                emissions = tracker.stop()  # total CO2e (kg)
                totals = getattr(tracker, "final_emissions_data", None) or getattr(tracker, "_emissions_data", None)
                details = {}
                if totals and hasattr(totals, "__dict__"):
                    details = {k: v for k, v in totals.__dict__.items() if not k.startswith("_")}
                elif isinstance(totals, dict):
                    details = totals

                def _maybe(*keys, default=None):
                    for k in keys:
                        if k in details and details[k] is not None:
                            return float(details[k])
                    return default

                mlflow.log_metric("emissions", float(emissions))
                mlflow.log_metric("emissions_rate", _maybe("emissions_rate", "emissions_rate_kg_s"))
                mlflow.log_metric("energy_consumed", _maybe("energy_consumed", "energy_kwh"))
                mlflow.log_metric("cpu_energy", _maybe("cpu_energy", "cpu_energy_kwh"))
                mlflow.log_metric("gpu_energy", _maybe("gpu_energy", "gpu_energy_kwh"))
                mlflow.log_metric("ram_energy", _maybe("ram_energy", "ram_energy_kwh"))
                mlflow.log_metric("cpu_power", _maybe("cpu_average_power_watts"))
                mlflow.log_metric("gpu_power", _maybe("gpu_average_power_watts"))
                mlflow.log_metric("ram_power", _maybe("ram_power_watts"))
            except Exception:
                pass

        # --------- NEW: always export PROV JSON for this run ---------
        try:
            if export_run_to_prov is not None:
                # Prefer env var; fallback to data/prov/<experiment_name>
                out_root = Path(os.getenv("YPROV_OUT_DIR", "data/prov"))
                # If we don’t trust exp.name yet, fetch it from the run (robust)
                exp_id = client.get_run(run_id).info.experiment_id
                exp_name = client.get_experiment(exp_id).name or args.exp_name
                out_dir = out_root / exp_name
                export_run_to_prov(run_id, out_dir, client=client)
                print(f"🟢 PROV JSON exported to: {out_dir}")
            else:
                print("ℹ️ yprov_mlflow_plugin.prov_export not available; skipping PROV export.")
        except Exception as e:
            print(f"⚠️ PROV export failed: {e}")

    print("✅ Run finished.")
    print("Artifacts directory:", artifact_dir)
    print("To launch MLflow UI:", "mlflow ui --backend-store-uri", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
