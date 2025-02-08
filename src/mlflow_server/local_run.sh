#!/usr/bin/env bash
source .venv/bin/activate

mlflow server \
  --backend-store-uri sqlite:////home/flavioramos/projects/fiap/FIAP-2MLET-Modulo-4/local_artifacts/mlflow_logs/mlflow.db \
  --default-artifact-root file:////home/flavioramos/projects/fiap/FIAP-2MLET-Modulo-4/local_artifacts/local_artifacts/training_artifacts \
  --host 0.0.0.0 \
  --port 5001

deactivate
