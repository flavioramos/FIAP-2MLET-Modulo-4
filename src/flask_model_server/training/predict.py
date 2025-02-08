import os
import pickle
import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
from models.lstm_model import LSTMModel
from utils.config_loader import load_parameters
from config import MODEL_LOCAL_PATH, SCALER_LOCAL_PATH, LAST_UPDATE_FILE, DEVICE

def get_last_update():
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, "r") as f:
                date_str = f.read().strip()
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass
    params = load_parameters()
    date_zero_str = params.get("DATE_ZERO", "2010-01-01")
    return datetime.strptime(date_zero_str, "%Y-%m-%d").date()

def run_prediction(date_str):
    params = load_parameters()
    ticker = params.get("TICKER", "AAPL")
    sequence_length = int(params.get("SEQUENCE_LENGTH", 5))
    hidden_size = int(params.get("HIDDEN_SIZE", 100))
    num_layers = int(params.get("NUM_LAYERS", 3))

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Formato de data inválido. Use YYYY-MM-DD."}

    if not os.path.exists(MODEL_LOCAL_PATH) or not os.path.exists(SCALER_LOCAL_PATH):
        return {"error": "Modelo ou Scaler não encontrados. Execute /train primeiro."}

    model = LSTMModel(input_size=1, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH))
    model.eval()

    with open(SCALER_LOCAL_PATH, "rb") as f:
        scaler = pickle.load(f)

    last_date = get_last_update()

    if target_date > last_date:
        start_date_seq = (last_date - timedelta(days=sequence_length * 3)).strftime("%Y-%m-%d")
        end_date_seq = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        df_seq = yf.download(ticker, start=start_date_seq, end=end_date_seq, progress=False)
        if df_seq.empty or len(df_seq) < sequence_length:
            return {"error": "Histórico insuficiente para formar uma sequência de entrada."}

        last_sequence = df_seq["Close"].values[-sequence_length:]
        last_sequence = last_sequence.reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        current_sequence = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        num_days = (target_date - last_date).days
        predictions_scaled = []
        for _ in range(num_days):
            with torch.no_grad():
                pred = model(current_sequence)
            predictions_scaled.append(pred.item())
            pred_tensor = pred.view(1, 1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], pred_tensor), dim=1)
        
        final_pred_scaled = np.array([[predictions_scaled[-1]]])
        predicted_value = scaler.inverse_transform(final_pred_scaled)[0, 0]
        return {
            "date": date_str,
            "predicted_close_price": predicted_value,
            "note": "Prediction for a future date (untrained)"
        }
    else:
        start_date_seq = (target_date - timedelta(days=sequence_length * 3)).strftime("%Y-%m-%d")
        end_date_seq = target_date.strftime("%Y-%m-%d")
        df_seq = yf.download(ticker, start=start_date_seq, end=end_date_seq, progress=False)
        if df_seq.empty or len(df_seq) < sequence_length:
            return {"error": "Histórico insuficiente para formar uma sequência de entrada para a data informada."}
        
        df_seq = df_seq[df_seq.index < pd.Timestamp(target_date)]
        if len(df_seq) < sequence_length:
            return {"error": "Histórico insuficiente para formar uma sequência de entrada para a data fornecida."}
        
        last_sequence = df_seq["Close"].values[-sequence_length:]
        last_sequence = last_sequence.reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        current_sequence = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred = model(current_sequence)
        predicted_scaled = pred.item()
        predicted_value = scaler.inverse_transform(np.array([[predicted_scaled]]))[0, 0]
        
        df_actual = yf.download(
            ticker,
            start=target_date.strftime("%Y-%m-%d"),
            end=(target_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False
        )
        actual_value = None
        if not df_actual.empty:
            actual_value = float(df_actual["Close"].iloc[0])
        
        return {
            "date": date_str,
            "predicted_close_price": predicted_value,
            "actual_close_price": actual_value,
            "note": "Prediction for a trained date (for accuracy comparison)"
        }
