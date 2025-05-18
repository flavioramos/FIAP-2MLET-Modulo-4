"""Prediction module for job matching model.

This module provides functionality to make predictions using the trained job matching model.
"""

import os
import joblib
import pandas as pd
from config import MODEL_LOCAL_PATH, STATUS_MAP


def run_prediction(job_description, candidate_cv):
    """Predict the probability of a candidate being hired for a job.
    
    Args:
        job_description (dict): Dictionary containing job information
            {
                "principais_atividades": str,
                "competencia_tecnicas_e_comportamentais": str
            }
        candidate_cv (str): Candidate's CV text
    
    Returns:
        dict: Prediction results containing:
            - probability: float, Probability of being hired
            - prediction: str, "Contratado" or "Não contratado"
            - confidence: float, Confidence score between 0 and 1
    """
    if not os.path.exists(MODEL_LOCAL_PATH):
        return {"error": "Model not found. Run /train first."}

    try:
        # Load the trained model
        model = joblib.load(MODEL_LOCAL_PATH)

        # Prepare input data
        input_data = pd.DataFrame([{
            "job_description": job_description.get("principais_atividades", ""),
            "job_requirements": job_description.get("competencia_tecnicas_e_comportamentais", ""),
            "candidate_cv": candidate_cv
        }])

        # Make prediction
        probability = model.predict_proba(input_data)[0, 1]
        
        return {
            "probability": float(probability),
            "prediction": "Contratado" if probability > 0.5 else "Não contratado",
            "confidence": float(abs(probability - 0.5) * 2)  # Convert to 0-1 scale
        }

    except Exception as e:
        return {"error": f"Error making prediction: {str(e)}"}
