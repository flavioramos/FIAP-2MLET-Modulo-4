#!/usr/bin/env bash
source .venv/bin/activate

SCRIPT_PATH="$(readlink -f "$0")"

THREE_LEVELS_UP="$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")"

# Caminho para o arquivo SQLite
DB_PATH="$THREE_LEVELS_UP/local_storage/mlflow_logs/mlflow.db"

# Caminho para os artifacts do MLflow
ARTIFACT_PATH="$THREE_LEVELS_UP/local_storage/training_artifacts"

echo "DB_PATH: ${DB_PATH}"
echo "ARTIFACT_PATH: ${ARTIFACT_PATH}"

# Inicia o servidor MLflow utilizando os caminhos construídos.
mlflow server \
  --backend-store-uri sqlite:///"${DB_PATH}" \
  --default-artifact-root file:///"${ARTIFACT_PATH}" \
  --host 0.0.0.0 \
  --port 5001

deactivate
