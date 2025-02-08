# app.py
import os
from flask import Flask, request, jsonify, send_from_directory
from training.train import run_training
from training.predict import run_prediction
from config import ARTIFACTS_DIR, LOGS_DIR

app = Flask(__name__)

@app.route("/train", methods=["GET"])
def train():
    reset_flag = request.args.get("reset", "false").lower() in ("true", "1", "yes")
    result = run_training(reset=reset_flag)
    return jsonify(result), 200

@app.route("/predict", methods=["GET"])
def predict():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "A data deve ser fornecida no formato YYYY-MM-DD"}), 400
    result = run_prediction(date_str)
    return jsonify(result), 200

@app.route("/artifacts", methods=["GET"])
def list_artifacts():
    artifacts = []
    for root, dirs, files in os.walk(ARTIFACTS_DIR):
        for file in files:
            rel_dir = os.path.relpath(root, ARTIFACTS_DIR)
            rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
            artifacts.append(rel_file)
    return jsonify({"artifacts": artifacts})

@app.route("/artifact/<path:filename>", methods=["GET"])
def get_artifact(filename):
    return send_from_directory(ARTIFACTS_DIR, filename, as_attachment=True)

@app.route("/logs", methods=["GET"])
def list_logs():
    logs = []
    for root, dirs, files in os.walk(LOGS_DIR):
        for file in files:
            rel_dir = os.path.relpath(root, LOGS_DIR)
            rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
            logs.append(rel_file)
    return jsonify({"logs": logs})

@app.route("/log/<path:filename>", methods=["GET"])
def get_log(filename):
    return send_from_directory(LOGS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
