# setup.py
from setuptools import setup

setup(
    name="mlflow-yprov4ml-plugin",
    version="0.1.0",
    install_requires=["mlflow>=2.0.0"],
    entry_points={
        # Define plugin entry points
        "mlflow.tracking_store": "my-scheme=my_plugin.store:MyTrackingStore",
    },
)
