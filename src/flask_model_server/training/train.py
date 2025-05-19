"""Training module for job matching model.

This module provides functionality to train the job matching model using MLflow
for experiment tracking and model management.
"""

import os
import json
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.config_loader import load_parameters
from utils.model_versioning import ModelVersioning
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

# Set up MLflow
mlflow.set_tracking_uri("sqlite:///" + os.path.join(LOGS_DIR, "mlflow.db"))
mlflow.set_experiment("job_matching")

def get_step_count():
    """Get the current training step count.
    
    Returns:
        int: Current step count, or 0 if not found
    """
    if os.path.exists(STEP_COUNT_FILE):
        try:
            with open(STEP_COUNT_FILE, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 0

def set_step_count(step):
    """Set the current training step count.
    
    Args:
        step (int): Step count to save
    """
    with open(STEP_COUNT_FILE, "w") as f:
        f.write(str(step))

def read_jsons():
    """Read and load JSON data files.
    
    Returns:
        tuple: (applicants, vagas, prospects) dictionaries
    """
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
    """Load and consolidate data from JSON files into a DataFrame.
    
    Returns:
        pd.DataFrame: Consolidated data with job descriptions, requirements,
                     candidate CVs, and status
    """
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
    """Run the model training process.
    
    Returns:
        dict: Training results including metrics and model information
    """
    params = load_parameters()
    step = get_step_count()

    with mlflow.start_run():
        # Load and process data
        df = load_and_consolidate_jsons()

        if df.empty:
            return {"error": "Empty DataFrame after ETL. Check JSONs in base_path."}

        # Log data statistics
        mlflow.log_metric("total_samples", len(df), step=step)
        # Log each class distribution separately
        class_dist = df["status"].value_counts(normalize=True)
        for class_name, proportion in class_dist.items():
            mlflow.log_metric(f"class_distribution_{class_name}", float(proportion), step=step)

        auc, grid, X_test, y_test = train_model(df)

        # Get predictions for additional metrics
        y_pred_proba = grid.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("auc", auc, step=step)
        mlflow.log_metric("best_c", grid.best_params_["clf__C"], step=step)
        mlflow.log_metric("best_cv_score", grid.best_score_, step=step)
        mlflow.log_metric("mean_cv_score", grid.cv_results_["mean_test_score"].mean(), step=step)
        mlflow.log_metric("std_cv_score", grid.cv_results_["std_test_score"].mean(), step=step)
        
        # Log classification metrics
        mlflow.log_metric("precision", precision, step=step)
        mlflow.log_metric("recall", recall, step=step)
        mlflow.log_metric("f1_score", f1, step=step)
        mlflow.log_metric("accuracy", accuracy, step=step)
        
        # Log confusion matrix metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        mlflow.log_metric("true_negatives", tn, step=step)
        mlflow.log_metric("false_positives", fp, step=step)
        mlflow.log_metric("false_negatives", fn, step=step)
        mlflow.log_metric("true_positives", tp, step=step)

        # Log model parameters
        mlflow.log_params({
            "tfidf_job_desc_max_features": TFIDF_JOB_DESCRIPTION_MAX_FEATURES,
            "tfidf_job_req_max_features": TFIDF_JOB_REQUIREMENTS_MAX_FEATURES,
            "tfidf_candidate_cv_max_features": TFIDF_CANDIDATE_CV_MAX_FEATURES,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "grid_search_cv": GRID_SEARCH_CV,
            "grid_search_scoring": GRID_SEARCH_SCORING,
            "grid_search_n_jobs": GRID_SEARCH_N_JOBS,
            "logistic_regression_max_iter": LOGISTIC_REGRESSION_MAX_ITER
        })

        # Increment model version and save model
        model_version = ModelVersioning.increment_version()
        versioned_model_path = ModelVersioning.get_versioned_path()
        os.makedirs(os.path.dirname(versioned_model_path), exist_ok=True)
        joblib.dump(grid.best_estimator_, versioned_model_path)
        print(f"Model version {model_version} saved at: {versioned_model_path}")

        # Update the latest model symlink
        ModelVersioning.update_latest_symlink(versioned_model_path)

        set_step_count(step + 1)
        
        return {
            "status": "Training completed successfully!",
            "model_version": model_version,
            "auc": float(auc),
            "best_c": float(grid.best_params_["clf__C"]),
            "best_cv_score": float(grid.best_score_),
            "mean_cv_score": float(grid.cv_results_["mean_test_score"].mean()),
            "std_cv_score": float(grid.cv_results_["std_test_score"].mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "class_distribution": class_dist.to_dict()
        }
