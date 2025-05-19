"""Flask application for job matching model training and prediction.

This module provides REST API endpoints for training the job matching model
and making predictions using the trained model.
"""

import os
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from training.train import run_training
from training.predict import run_prediction
import configparser

app = Flask(__name__)

# Load configuration
config = configparser.ConfigParser()
config.read('default_params.txt')

# JWT Configuration
app.config['JWT_SECRET_KEY'] = config.get('Authentication', 'JWT_SECRET_KEY', fallback='your-super-secret-key-change-in-production')
jwt = JWTManager(app)

# Default credentials
DEFAULT_USERNAME = config.get('Authentication', 'DEFAULT_USERNAME', fallback='user')
DEFAULT_PASSWORD = config.get('Authentication', 'DEFAULT_PASSWORD', fallback='password')

@app.route("/login", methods=["POST"])
def login():
    """Login endpoint to get JWT token.
    
    Returns:
        tuple: JSON response with access token and HTTP status code
    """
    auth = request.get_json()
    if not auth:
        return jsonify({"error": "No JSON data provided"}), 400
    
    username = auth.get("username")
    password = auth.get("password")
    
    if not username or not password:
        return jsonify({"error": "Both username and password are required"}), 400
    
    if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
        access_token = create_access_token(identity=username)
        return jsonify({"access_token": access_token}), 200
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/train", methods=["GET"])
@jwt_required()
def train():
    """Train the job matching model.
    
    Returns:
        tuple: JSON response with training results and HTTP status code
    """
    result = run_training()
    return jsonify(result), 200

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """Make a prediction using the trained model.
    
    Returns:
        tuple: JSON response with prediction results and HTTP status code
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    principais_atividades = data.get("principais_atividades")
    competencia_tecnicas_e_comportamentais = data.get("competencia_tecnicas_e_comportamentais")
    cv_pt = data.get("cv_pt")
    
    if not principais_atividades or not competencia_tecnicas_e_comportamentais or not cv_pt:
        return jsonify({"error": "All fields (principais_atividades, competencia_tecnicas_e_comportamentais, and cv_pt) are required"}), 400
    
    try:
        result = run_prediction(principais_atividades, competencia_tecnicas_e_comportamentais, cv_pt)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
