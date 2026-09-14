import pytest
import os
from testbook import testbook

# This integration test verifies that the pipeline runs end-to-end locally.
# It uses the dummy JSON data generated in the project.

@pytest.mark.integration
def test_pipeline_execution():
    # Set environment to local to avoid Databricks Unity Catalog registration attempts
    os.environ['ENVIRONMENT'] = 'local'
    os.environ['CATALOG_NAME'] = 'test_catalog'
    os.environ['SCHEMA_NAME'] = 'test_schema'
    
    # Ensure data exists for the test to run
    assert os.path.exists('data/sample_nested.json'), "Run generate_data.py to create the dummy dataset first."
    
    try:
        # Execute the entire notebook with a higher timeout for TensorFlow initialization
        with testbook('notebooks/model_training.ipynb', execute=True, timeout=300) as tb:
            # We can verify that certain variables were created and have expected types
            records = tb.ref("records")
            X_train = tb.ref("X_train")
            model = tb.ref("model")
            
            assert len(records) > 0, "No records loaded"
            assert len(X_train) > 0, "Training set is empty"
            assert model is not None, "Model was not built"
            
            # Verify MLflow tracking completed (the run ID should exist)
            run_id_exists = tb.value("run.info.run_id is not None")
            assert run_id_exists, "MLflow run was not created"
            
    except Exception as e:
        pytest.fail(f"Notebook execution failed: {e}")
