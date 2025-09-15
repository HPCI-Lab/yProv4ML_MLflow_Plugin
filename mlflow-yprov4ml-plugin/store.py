# my_plugin/store.py
from mlflow.store.tracking.abstract_store import AbstractStore


class MyTrackingStore(AbstractStore):
    """Custom tracking store for scheme 'my-scheme://'"""

    def __init__(self, store_uri):
        super().__init__()
        self.store_uri = store_uri
        # Initialize your custom storage backend

    def create_experiment(self, name, artifact_location=None, tags=None):
        # Implement experiment creation logic
        pass

    def log_metric(self, run_id, metric):
        # Implement metric logging logic
        pass

    def log_param(self, run_id, param):
        # Implement parameter logging logic
        pass

    # Implement other required AbstractStore methods...