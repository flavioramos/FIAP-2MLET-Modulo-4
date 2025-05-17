# app.py
import os
from flask import Flask, request, jsonify
from training.train import run_training
from training.predict import run_prediction

app = Flask(__name__)

@app.route("/train", methods=["GET"])
def train():
    result = run_training()
    return jsonify(result), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    job_description = data.get("job_description")
    candidate_cv = data.get("candidate_cv")
    
    if not job_description or not candidate_cv:
        return jsonify({"error": "Both job_description and candidate_cv are required"}), 400
    
    try:
        result = run_prediction(job_description, candidate_cv)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
