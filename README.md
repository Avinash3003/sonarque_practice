# ML Migration Project

## Objective
This project demonstrates a simple production-style ML lifecycle using a dummy nested JSON dataset, Databricks Asset Bundles (DAB), MLflow, TensorFlow, and CI/CD pipelines. It is designed for learning and demonstration.

## Architecture Flow
```text
          Git (GitHub)
                 ↓
          CI / Testing (GitHub Actions + Pytest)
                 ↓
            SonarQube (Static Analysis)
                 ↓
            DAB Validate (Syntax Check)
                 ↓
           DAB Deploy (dev/prod target)
                 ↓
         Databricks Job (Execution)
                 ↓
          .ipynb Notebook (model_training.ipynb)
                 ↓
          Nested JSON (data/sample_nested.json)
                 ↓
         Preprocessing (Flattening, Encoding)
                 ↓
         TensorFlow Model (Binary Classification)
                 ↓
          Train / Evaluate (Keras)
                 ↓
              MLflow (Tracking)
      ┌──────────┼──────────┐
      ↓          ↓          ↓
   Metrics    Artifacts    Model
                          ↓
                   Model Registry (Unity Catalog)
                          ↓
                   Version 1/2/3...
                          ↓
                   Model Serving (Databricks Serving)
                          ↓
                   JSON Prediction
```

## Repository Structure
- `notebooks/`: Contains the primary implementation, `model_training.ipynb`. Code is kept as a notebook per requirements.
- `data/`: Contains the dummy nested JSON dataset.
- `tests/`: Contains unit and integration tests. Unit tests validate notebook logic using `testbook`.
- `resources/`: Contains Databricks Job and Serving configurations.
- `databricks.yml`: Defines the Databricks Asset Bundle for deployment.
- `.github/workflows/`: Contains the CI pipeline.

## Implementation Details

### How the Notebook Works
The notebook (`model_training.ipynb`) is structured linearly:
1. Configuration (Catalogs, schemas, MLflow URIs)
2. Data Loading (JSON) & Preprocessing (Flattening nested dicts)
3. Model Definition (TensorFlow Keras Dense network)
4. Training and Evaluation
5. MLflow Logging (Parameters, Metrics, Artifacts, Model)
6. Unity Catalog Registration

### MLflow Lifecycle & Model Versioning
During each successful run of the notebook, an MLflow run is started. The model is trained and logged using `mlflow.tensorflow.log_model()`. The model is then registered to Unity Catalog using the configured catalog, schema, and model name. Every successful run creates a new version.

### DAB, CI/CD, and Testing Roles
- **DAB (Databricks Asset Bundles)**: Defines infrastructure-as-code for the job and the serving endpoint.
- **CI/CD**: GitHub Actions automatically runs tests, SonarQube analysis, and DAB validation on push.
- **Testing**: `pytest` and `testbook` are used to execute cells in the `.ipynb` file locally to verify preprocessing logic and pipeline flow without requiring a live Databricks environment.
- **SonarQube**: Evaluates the code quality and coverage.

## What Remains to be Configured in Phase 2
- Real Databricks Workspace URLs in `databricks.yml`
- Real GitHub Secrets for Databricks integration
- Real SonarQube tokens and configuration
- Cloud Storage (e.g., AWS S3) for data access instead of local JSON
- Unity Catalog and MLflow tracking URIs
