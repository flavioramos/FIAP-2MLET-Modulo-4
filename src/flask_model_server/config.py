"""Configuration settings for the ML model training and serving.

This module contains all the configuration parameters, paths, and constants
used throughout the application for both local and containerized environments.
"""

import os
import sys
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL = sys.argv[-1] == 'local'  # sessão local ou remota (container)

print(f"Running locally: {LOCAL}")


# ARTIFACTS
# Diretório para armazenar os artefatos do treinamento (modelo, scaler, last_update, etc.)
if LOCAL:
    ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/training_artifacts"))
else:
    ARTIFACTS_DIR = os.path.abspath(os.path.join("/storage/", "training_artifacts"))

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")


# MLFLOW LOGS
# Diretório para armazenar os logs do MLflow
if LOCAL:
    LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/mlflow_logs"))
else:
    LOGS_DIR = os.path.abspath(os.path.join("/storage/", "mlflow_logs"))

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

print(f"LOGS_DIR: {LOGS_DIR}")


# PARAMS
# Diretório para armazenar os parâmetros
if LOCAL:
    PARAMS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../local_storage/parameters"))
else:
    PARAMS_DIR = os.path.abspath(os.path.join("/storage/", "parameters"))

if not os.path.exists(PARAMS_DIR):
    os.makedirs(PARAMS_DIR)

if not os.path.exists(os.path.join(PARAMS_DIR, "params.txt")):
    shutil.copyfile(
        os.path.join(BASE_DIR, "default_params.txt"),
        os.path.join(PARAMS_DIR, "params.txt")
    )
    print(f"Default params.txt copied to {PARAMS_DIR}")

print(f"PARAMS_DIR: {LOGS_DIR}")


# Caminhos para arquivos gerados (dentro do diretório de artefatos)
MODEL_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
SCALER_LOCAL_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
LAST_UPDATE_FILE = os.path.join(ARTIFACTS_DIR, "last_update.txt")
STEP_COUNT_FILE = os.path.join(ARTIFACTS_DIR, "step_count.txt")

STATUS_MAP = {
    "Encaminhado ao Requisitante":        0,
    "Contratado pela Decision":           1,
    "Desistiu":                           0,
    "Documentação PJ":                    1,
    "Não Aprovado pelo Cliente":          0,
    "Prospect":                           0,
    "Não Aprovado pelo RH":               0,
    "Aprovado":                           1,
    "Não Aprovado pelo Requisitante":     0,
    "Inscrito":                           0,
    "Entrevista Técnica":                 0,
    "Em avaliação pelo RH":               0,
    "Contratado como Hunting":            1,
    "Desistiu da Contratação":            0,
    "Entrevista com Cliente":             0,
    "Documentação CLT":                   1,
    "Recusado":                           0,
    "Documentação Cooperado":             1,
    "Sem interesse nesta vaga":           0,
    "Encaminhar Proposta":                1,
    "Proposta Aceita":                    1
}

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF parameters
TFIDF_JOB_DESCRIPTION_MAX_FEATURES = 1000
TFIDF_JOB_DESCRIPTION_NGRAM_RANGE = (1, 2)

TFIDF_JOB_REQUIREMENTS_MAX_FEATURES = 1000
TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE = (1, 2)

TFIDF_CANDIDATE_CV_MAX_FEATURES = 5000
TFIDF_CANDIDATE_CV_NGRAM_RANGE = (1, 2)

# Logistic Regression parameters
LOGISTIC_REGRESSION_MAX_ITER = 1000

# Grid Search parameters
GRID_SEARCH_CV = 5
GRID_SEARCH_SCORING = "roc_auc"
GRID_SEARCH_N_JOBS = -1
GRID_SEARCH_C_VALUES = [0.1, 1, 10]

# Data paths
APPLICANTS_PATH = "../../data/raw/applicants.json"
VAGAS_PATH = "../../data/raw/vagas.json"
PROSPECTS_PATH = "../../data/raw/prospects.json"