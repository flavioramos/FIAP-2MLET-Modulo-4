# app.py
import os
from flask import Flask, request, jsonify
from training.train import run_training
from training.predict import run_prediction

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
