import os, time, json, mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

# Create a WSL/Windows/Linux-safe artifact folder and convert to file:// URI
artifact_dir = Path.cwd() / "mlartifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
artifact_uri = artifact_dir.resolve().as_uri()  

exp_name = "demo_yprov_plugin"
client = MlflowClient()

# Create experiment once with the correct artifact location
exp = client.get_experiment_by_name(exp_name)
if exp is None:
    client.create_experiment(exp_name, artifact_location=artifact_uri)

mlflow.set_experiment(exp_name)

with mlflow.start_run(run_name="hello-yprov"):
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("batch_size", 64)

    for step, loss in enumerate([0.7, 0.42, 0.31, 0.27, 0.25], start=1):
        mlflow.log_metric("loss", loss, step=step)
        time.sleep(0.1)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/config.json", "w") as f:
        json.dump({"hello": "yprov"}, f)

    mlflow.log_artifact("outputs/config.json", artifact_path="config")

print("✅ Run finished. Artifacts saved to:", artifact_dir)
print("You can now explore the run in MLflow UI.")
print("To launch MLflow UI, run: mlflow ui --backend-store-uri", mlflow.get_tracking_uri())