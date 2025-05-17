import os
import json
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.config_loader import load_parameters
from config import (
    LOGS_DIR, MODEL_LOCAL_PATH, STEP_COUNT_FILE, STATUS_MAP,
    TEST_SIZE, RANDOM_STATE,
    TFIDF_JOB_DESCRIPTION_MAX_FEATURES, TFIDF_JOB_DESCRIPTION_NGRAM_RANGE,
    TFIDF_JOB_REQUIREMENTS_MAX_FEATURES, TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE,
    TFIDF_CANDIDATE_CV_MAX_FEATURES, TFIDF_CANDIDATE_CV_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER,
    GRID_SEARCH_CV, GRID_SEARCH_SCORING, GRID_SEARCH_N_JOBS, GRID_SEARCH_C_VALUES,
    APPLICANTS_PATH, VAGAS_PATH, PROSPECTS_PATH
)
from models.job_matching_model import train_model

mlflow.set_tracking_uri("sqlite:///" + os.path.join(LOGS_DIR, "mlflow.db"))
mlflow.set_experiment("job_matching")

def get_step_count():
    if os.path.exists(STEP_COUNT_FILE):
        try:
            with open(STEP_COUNT_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 0

def set_step_count(step):
    with open(STEP_COUNT_FILE, "w") as f:
        f.write(str(step))

def read_jsons():
    print(f"\n=== Loading JSON files ===")
    print(f"Loading applicants from: {APPLICANTS_PATH}")
    with open(APPLICANTS_PATH, encoding='utf-8') as f:
        applicants = json.load(f)
    print(f"Loading vagas from: {VAGAS_PATH}")
    with open(VAGAS_PATH, encoding='utf-8') as f:
        vagas = json.load(f)
    print(f"Loading prospects from: {PROSPECTS_PATH}")
    with open(PROSPECTS_PATH, encoding='utf-8') as f:
        prospects = json.load(f)

    return applicants, vagas, prospects


def load_and_consolidate_jsons():
    applicants, vagas, prospects = read_jsons()

    rows = []
    statuses = []
    for vaga_idx in prospects:
        if vaga_idx in vagas:
            for prospect_candidate in prospects[vaga_idx]["prospects"]:
                if prospect_candidate["codigo"] in applicants:
                    rows.append({
                        "job_description": vagas[vaga_idx]["perfil_vaga"]["principais_atividades"],
                        "job_requirements": vagas[vaga_idx]["perfil_vaga"]["competencia_tecnicas_e_comportamentais"],
                        "candidate_cv": applicants[prospect_candidate["codigo"]]["cv_pt"],
                        "status": prospect_candidate["situacao_candidado"]
                    })
                    if (prospect_candidate["situacao_candidado"] not in statuses):
                        statuses.append(prospect_candidate["situacao_candidado"])

    return pd.DataFrame(rows)

def run_training():
    params = load_parameters()

    step = get_step_count()

    with mlflow.start_run():
        # Log parameters
        # mlflow.log_param("data_path", base_path)

        # Load and process data
        df = load_and_consolidate_jsons()

        if df.empty:
            return {"error": "DataFrame vazio após ETL. Verifique os JSONs em base_path."}

        auc, grid = train_model(df)

        # Log metrics
        mlflow.log_metric("auc", auc, step=step)
        mlflow.log_metric("best_c", grid.best_params_["clf__C"], step=step)

        # Save model
        os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
        joblib.dump(grid.best_estimator_, MODEL_LOCAL_PATH)
        print(f"Modelo salvo em: {MODEL_LOCAL_PATH}")

        set_step_count(step + 1)
        
        return {
            "status": "Treinamento concluído com sucesso!",
            "auc": float(auc),
            "best_c": float(grid.best_params_["clf__C"])
        }
