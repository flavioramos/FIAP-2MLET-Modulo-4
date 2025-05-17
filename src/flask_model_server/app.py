# app.py
import os
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from training.train import run_training
from training.predict import run_prediction

def load_auth_properties():
    properties = {}
    try:
        with open('auth.properties', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    properties[key.strip()] = value.strip()
    except FileNotFoundError:
        print("Warning: auth.properties file not found. Using default values.")
        properties = {
            'ADMIN_USERNAME': 'admin',
            'ADMIN_PASSWORD': 'password',
            'JWT_SECRET_KEY': 'your-secret-key'
        }
    return properties

app = Flask(__name__)

# Load auth properties
auth_properties = load_auth_properties()

# Configure JWT
app.config["JWT_SECRET_KEY"] = auth_properties.get('JWT_SECRET_KEY')
jwt = JWTManager(app)

# Admin credentials from properties
ADMIN_USERNAME = auth_properties.get('ADMIN_USERNAME')
ADMIN_PASSWORD = auth_properties.get('ADMIN_PASSWORD')

@app.route("/login", methods=["POST"])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid credentials"}), 401
    
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200

@app.route("/train", methods=["GET"])
# @jwt_required()
def train():
    result = run_training()
    return jsonify(result), 200

@app.route("/predict", methods=["POST"])
# @jwt_required()
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    job_description = data.get("job_description")
    candidate_cv = data.get("candidate_cv")
    
    if not job_description or not candidate_cv:
        return jsonify({"error": "Both job_description and candidate_cv are required"}), 400
    
    result = run_prediction(job_description, candidate_cv)
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
