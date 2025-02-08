import os
import pickle
import mlflow
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from models.lstm_model import LSTMModel
from utils.config_loader import load_parameters
from utils.data_utils import download_data_from_yfinance, preprocess_data
from config import MODEL_LOCAL_PATH, SCALER_LOCAL_PATH, LAST_UPDATE_FILE, DEVICE

# Configura o tracking URI do MLflow para usar um banco SQLite localizado no diretório de logs.
# Importante: Para caminhos absolutos no Linux, o esquema 'sqlite' exige 4 barras.
mlflow.set_tracking_uri("sqlite:////app/mlflow_logs/mlflow.db")

def get_last_update():
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, "r") as f:
                date_str = f.read().strip()
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass
    # Se não houver last_update, usa DATE_ZERO de params.txt
    params = load_parameters()
    date_zero_str = params.get("DATE_ZERO", "2010-01-01")
    return datetime.strptime(date_zero_str, "%Y-%m-%d").date()

def set_last_update(date_obj):
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(date_obj.strftime("%Y-%m-%d"))

def run_training(reset=False):
    params = load_parameters()
    ticker = params.get("TICKER", "AAPL")
    sequence_length = int(params.get("SEQUENCE_LENGTH", 5))
    epochs = int(params.get("EPOCHS", 100))
    learning_rate = float(params.get("LEARNING_RATE", 0.0005))
    hidden_size = int(params.get("HIDDEN_SIZE", 100))
    num_layers = int(params.get("NUM_LAYERS", 3))
    date_zero_str = params.get("DATE_ZERO", "2010-01-01")

    if reset:
        try:
            date_zero = datetime.strptime(date_zero_str, "%Y-%m-%d").date()
        except Exception:
            date_zero = datetime(2010, 1, 1).date()
        set_last_update(date_zero)
        print(f"Reset solicitado: last_update definido para {date_zero}")

    # Loga os parâmetros do treinamento no MLflow
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("sequence_length", sequence_length)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("ticker", ticker)

    last_update = get_last_update()
    df = download_data_from_yfinance(last_update, ticker)
    if df is None:
        return {"error": "Erro ao baixar dados do Yahoo Finance"}
    
    X_np, y_np, scaler = preprocess_data(df, sequence_length, DEVICE)
    X = torch.tensor(X_np).to(DEVICE)
    y = torch.tensor(y_np).to(DEVICE)
    
    model = LSTMModel(input_size=1, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 0:
        model.load_state_dict(torch.load(MODEL_LOCAL_PATH))
        print("Modelo carregado para re-treinamento.")
    else:
        print("Treinando modelo do zero.")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        train_preds = model(X)
    train_preds_np = train_preds.cpu().numpy()
    y_np = y.cpu().numpy()
    mae = float(np.mean(np.abs(train_preds_np - y_np)))
    rmse = float(np.sqrt(np.mean((train_preds_np - y_np) ** 2)))
    epsilon = 1e-8  # Para evitar divisão por zero
    mape = float(np.mean(np.abs((train_preds_np - y_np) / (y_np + epsilon))) * 100)

    mlflow.log_metric("loss", float(loss.item()))
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

    # Salva os artefatos (modelo e scaler)
    torch.save(model.state_dict(), MODEL_LOCAL_PATH)
    with open(SCALER_LOCAL_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return {
        "status": "Treinamento concluído com sucesso!",
        "loss": float(loss.item()),
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
