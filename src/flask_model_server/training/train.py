import os
import json
import joblib
import mlflow
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from utils.config_loader import load_parameters
from config import LOGS_DIR, MODEL_LOCAL_PATH, STEP_COUNT_FILE

mlflow.set_tracking_uri("sqlite:///" + os.path.join(LOGS_DIR, "mlflow.db"))
mlflow.set_experiment("job_matching")

applicants_path = "../../data/raw/applicants.json"
vagas_path = "../../data/raw/vagas.json"
prospects_path = "../../data/raw/prospects.json"
status_map = {
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
    print(f"Loading applicants from: {applicants_path}")
    with open(applicants_path, encoding='utf-8') as f:
        applicants = json.load(f)
    print(f"Loading vagas from: {vagas_path}")
    with open(vagas_path, encoding='utf-8') as f:
        vagas = json.load(f)
    print(f"Loading prospects from: {prospects_path}")
    with open(prospects_path, encoding='utf-8') as f:
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

        x = df[["job_description","job_requirements","candidate_cv"]]
        y = df["status"].map(status_map)

        X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer([
            ("tfidf_ativ", TfidfVectorizer(max_features=1000, ngram_range=(1,2)),
                "job_description"),
            ("tfidf_comp", TfidfVectorizer(max_features=1000, ngram_range=(1,2)),
                "job_requirements"),
            ("tfidf_cv",   TfidfVectorizer(max_features=5000, ngram_range=(1,2)),
                "candidate_cv"),
        ], remainder="drop")

        # Create full pipeline
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        grid = GridSearchCV(
            pipeline, {"clf__C": [0.1, 1, 10]},
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)

        # Final evaluation
        y_pred = grid.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"AUC no conjunto de teste: {auc:.4f}")

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
