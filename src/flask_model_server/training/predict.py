"""Prediction module for job matching model.

This module provides functionality to make predictions using the trained job matching model.
"""

import os
import joblib
import pandas as pd
from config import MODEL_LOCAL_PATH, STATUS_MAP
from utils.model_versioning import ModelVersioning


def run_prediction(principais_atividades, competencia_tecnicas_e_comportamentais, cv_pt):
    """Predict the probability of a candidate being hired for a job.
    
    Args:
        principais_atividades (str): Main activities of the job
        competencia_tecnicas_e_comportamentais (str): Technical and behavioral requirements
        cv_pt (str): Candidate's CV text
    
    Returns:
        dict: Prediction results containing:
            - probability: float, Probability of being hired
            - prediction: str, "Contratado" or "Não contratado"
            - confidence: float, Confidence score between 0 and 1
            - model_version: int, Version of the model used for prediction
    """
    if not os.path.exists(MODEL_LOCAL_PATH):
        return {"error": "Model not found. Run /train first."}

    try:
        # Load the trained model
        model = joblib.load(MODEL_LOCAL_PATH)
        model_version = ModelVersioning.get_version()

        # Prepare input data
        input_data = pd.DataFrame([{
            "job_description": principais_atividades,
            "job_requirements": competencia_tecnicas_e_comportamentais,
            "candidate_cv": cv_pt
        }])

        # Make prediction
        probability = model.predict_proba(input_data)[0, 1]
        
        return {
            "probability": float(probability),
            "prediction": "Contratado" if probability > 0.5 else "Não contratado",
            "confidence": float(abs(probability - 0.5) * 2),  # Convert to 0-1 scale
            "model_version": model_version
        }

    except Exception as e:
        return {"error": f"Error making prediction: {str(e)}"}
