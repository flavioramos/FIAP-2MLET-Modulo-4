# config.py
import os
import torch
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL = sys.argv[-1] == 'local' # sessão local ou remota (container)

print(f"Running locally: {LOCAL}")

# Diretório para armazenar os artefatos do treinamento (modelo, scaler, last_update, etc.)
if LOCAL:
    ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_artifacts/training_artifacts"))
else:
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "training_artifacts")

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")

# Diretório para armazenar os logs do MLflow
if LOCAL:
    LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_artifacts/mlflow_logs"))
else:
    LOGS_DIR = os.path.join(BASE_DIR, "mlflow_logs")

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

print(f"LOGS_DIR: {LOGS_DIR}")

# Caminhos para arquivos gerados (dentro do diretório de artefatos)
MODEL_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "model_lstm.pt")
SCALER_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
LAST_UPDATE_FILE = os.path.join(ARTIFACTS_DIR, "last_update.txt")
STEP_COUNT_FILE = os.path.join(ARTIFACTS_DIR, "step_count.txt")

# Configuração do dispositivo para PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
