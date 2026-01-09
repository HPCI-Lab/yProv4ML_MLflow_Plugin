import mlflow

# 🔑 CRITICAL: Use yprov+ prefix to activate the plugin
mlflow.set_tracking_uri('yprov+file:tmp')

# Standard MLflow code - no changes needed!
mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="example_run"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    
    # Log metrics
    for epoch in range(5):
        mlflow.log_metric("loss", 0.5 - epoch * 0.05, step=epoch)
        mlflow.log_metric("accuracy", 0.7 + epoch * 0.05, step=epoch)
    
    # Log artifacts
    # mlflow.log_artifact("model.pt", artifact_path="models")

