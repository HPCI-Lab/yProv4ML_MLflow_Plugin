
import mlflow

print(mlflow.__version__)

# Verify prov4ml is installed
import yprov4ml
print('prov4ml installed successfully')

# Check plugin entry points
from importlib.metadata import entry_points 

eps = [ep for ep in entry_points().get('mlflow.tracking_store', []) if 'yprov' in ep.name]
print(f'Found {len(eps)} yprov entry points')

# Test basic functionality
mlflow.set_tracking_uri('yprov+file:///tmp/test')
print('✅ Plugin activated successfully')