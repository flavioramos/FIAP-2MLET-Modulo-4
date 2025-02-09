import os
import pickle
import mlflow
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from models.lstm_model import LSTMModel
from utils.config_loader import load_parameters
from utils.data_utils import download_data_from_yfinance, preprocess_data
from config import LOGS_DIR, MODEL_LOCAL_PATH, SCALER_LOCAL_PATH, LAST_UPDATE_FILE, STEP_COUNT_FILE, DEVICE

mlflow.set_tracking_uri("sqlite:///" + os.path.join(LOGS_DIR, "mlflow.db"))
mlflow.set_experiment("stock_prediction")

def get_last_update(params):
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, "r") as f:
                date_str = f.read().strip()
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass
    # Se não houver last_update, usa DATE_ZERO de params.txt
    date_zero_str = params.get("DATE_ZERO", "2010-01-01")
    return datetime.strptime(date_zero_str, "%Y-%m-%d").date()

def set_last_update(date_obj):
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(date_obj.strftime("%Y-%m-%d"))

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

def run_training(reset=False):
    params = load_parameters()
    ticker = params.get("TICKER")
    sequence_length = int(params.get("SEQUENCE_LENGTH"))
    epochs = int(params.get("EPOCHS"))
    learning_rate = float(params.get("LEARNING_RATE"))
    hidden_size = int(params.get("HIDDEN_SIZE"))
    num_layers = int(params.get("NUM_LAYERS"))
    date_zero_str = params.get("DATE_ZERO")

    step = get_step_count()

    if reset:
        try:
            date_zero = datetime.strptime(date_zero_str, "%Y-%m-%d").date()
        except Exception:
            date_zero = datetime(2010, 1, 1).date()
        set_last_update(date_zero)
        print(f"Reset solicitado: last_update definido para {date_zero}")

    with mlflow.start_run():
        # Loga os parâmetros do treinamento no MLflow
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("ticker", ticker)

        last_update = get_last_update(params)
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

        mlflow.log_metric("loss", float(loss.item()), step=step)
        mlflow.log_metric("mae", mae, step=step)
        mlflow.log_metric("rmse", rmse, step=step)
        mlflow.log_metric("mape", mape, step=step)

    # Salva os artefatos (modelo e scaler)
    torch.save(model.state_dict(), MODEL_LOCAL_PATH)
    with open(SCALER_LOCAL_PATH, "wb") as f:
        pickle.dump(scaler, f)

    set_step_count(step + 1)
    
    return {
        "status": "Treinamento concluído com sucesso!",
        "loss": float(loss.item()),
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
