import os
import pickle
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


def load_parameters(file_path="params.txt"):
    """
    Load parameters from a txt file.
    Empty lines and lines starting with '#' are ignored.
    Attempts to convert values to int or float when possible.
    """
    params = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            params[key] = int(value)
                        except ValueError:
                            try:
                                params[key] = float(value)
                            except ValueError:
                                params[key] = value
    else:
        print(f"File {file_path} not found. Using default parameters.")
    return params


# Diretório onde serão salvos os arquivos do modelo.
MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_LOCAL_PATH = os.path.join(MODEL_DIR, "model_lstm.pt")
SCALER_LOCAL_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LAST_UPDATE_FILE = os.path.join(MODEL_DIR, "last_update.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flask app initialization.
app = Flask(__name__)


def get_last_update():
    """
    Retrieve the last update date from a local file.
    If the file does not exist, return the DATE_ZERO value from params.txt,
    or default to 5 years ago.
    """
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, "r") as f:
                date_str = f.read().strip()
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass

    # Se o arquivo não existir, tenta usar a DATE_ZERO definida em params.txt.
    params = load_parameters("params.txt")
    date_zero_str = params.get("DATE_ZERO", None)
    if date_zero_str is not None:
        try:
            return datetime.strptime(date_zero_str, "%Y-%m-%d").date()
        except Exception:
            pass
    return datetime.today().date() - timedelta(days=5 * 365)


def set_last_update(date_obj):
    """
    Save the last update date to a local file.
    """
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(date_obj.strftime("%Y-%m-%d"))


def download_data_from_yfinance(start_date, ticker):
    """
    Download historical data from Yahoo Finance starting from the given date.
    """
    end_date = datetime.today().date()
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if not df.empty:
        set_last_update(end_date)
    return df if not df.empty else None


def preprocess_data(df, sequence_length):
    """
    Normalize 'Close' prices and create sequences for the LSTM.
    Returns tensors X, y and the scaler.
    """
    df = df[["Close"]].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(df_scaled)):
        X.append(df_scaled[i - sequence_length:i])
        y.append(df_scaled[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return torch.tensor(X).to(DEVICE), torch.tensor(y).to(DEVICE), scaler


class LSTMModel(nn.Module):
    """
    Define the LSTM model architecture.
    Hyperparameters can be provided during instantiation.
    """

    def __init__(self, input_size=1, hidden_size=100, num_layers=1,
                 output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


@app.route("/train", methods=["GET"])
def train():
    """
    Route to train (or retrain) the model.
    Parameters are loaded from the file at call time,
    allowing dynamic updates.
    Se o parâmetro "reset" for passado (por exemplo, /train?reset=true),
    a data de última atualização é redefinida para DATE_ZERO.
    """
    params = load_parameters("params.txt")
    ticker = params.get("TICKER", "AAPL")
    sequence_length = int(params.get("SEQUENCE_LENGTH", 5))
    epochs = int(params.get("EPOCHS", 100))
    batch_size = int(params.get("BATCH_SIZE", 32))
    learning_rate = float(params.get("LEARNING_RATE", 0.0005))
    hidden_size = int(params.get("HIDDEN_SIZE", 100))
    num_layers = int(params.get("NUM_LAYERS", 3))

    # Verifica se o parâmetro de reset foi passado.
    reset_flag = request.args.get("reset", "false").lower() in (
        "true", "1", "yes"
    )
    if reset_flag:
        date_zero_str = params.get("DATE_ZERO", "2010-01-01")
        try:
            date_zero = datetime.strptime(date_zero_str, "%Y-%m-%d").date()
        except Exception:
            date_zero = datetime(2010, 1, 1).date()
        set_last_update(date_zero)
        print(f"Reset requested: last_update set to {date_zero}")

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("sequence_length", sequence_length)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("ticker", ticker)

    last_update = get_last_update()
    df = download_data_from_yfinance(last_update, ticker)
    if df is None:
        return jsonify(
            {"error": "Error downloading data from Yahoo Finance"}
        ), 500

    X, y, scaler = preprocess_data(df, sequence_length)
    model = LSTMModel(
        input_size=1, hidden_size=hidden_size, num_layers=num_layers,
        output_size=1
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if os.path.exists(MODEL_LOCAL_PATH):
        model.load_state_dict(torch.load(MODEL_LOCAL_PATH))
        print("Model loaded for retraining.")
    else:
        print("Training model from scratch.")

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
    mae = np.mean(np.abs(train_preds_np - y_np))
    rmse = np.sqrt(np.mean((train_preds_np - y_np) ** 2))
    mape = np.mean(np.abs((train_preds_np - y_np) / y_np)) * 100

    mlflow.log_metric("loss", loss.item())
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

    torch.save(model.state_dict(), MODEL_LOCAL_PATH)
    with open(SCALER_LOCAL_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return jsonify({
        "status": "Training completed successfully!",
        "loss": float(loss.item()),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }), 200


@app.route("/predict", methods=["GET"])
def predict():
    """
    Route to predict the closing price for a given date.
    Parameters are loaded at call time for dynamic tuning.
    """
    params = load_parameters("params.txt")
    ticker = params.get("TICKER", "AAPL")
    sequence_length = int(params.get("SEQUENCE_LENGTH", 5))
    hidden_size = int(params.get("HIDDEN_SIZE", 100))
    num_layers = int(params.get("NUM_LAYERS", 3))

    date_str = request.args.get("date")
    if not date_str:
        return jsonify({
            "error": "Date must be provided in YYYY-MM-DD format."
        }), 400
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({
            "error": "Invalid date format. Use YYYY-MM-DD."
        }), 400

    if not os.path.exists(MODEL_LOCAL_PATH):
        return jsonify({
            "error": "Model not found. Execute /train first."
        }), 500
    if not os.path.exists(SCALER_LOCAL_PATH):
        return jsonify({
            "error": "Scaler not found. Execute /train first."
        }), 500

    model = LSTMModel(
        input_size=1, hidden_size=hidden_size, num_layers=num_layers,
        output_size=1
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH))
    model.eval()

    with open(SCALER_LOCAL_PATH, "rb") as f:
        scaler = pickle.load(f)

    last_date = get_last_update()

    if target_date > last_date:
        start_date_seq = (
            last_date - timedelta(days=sequence_length * 3)
        ).strftime("%Y-%m-%d")
        end_date_seq = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        df_seq = yf.download(ticker, start=start_date_seq,
                             end=end_date_seq, progress=False)

        if df_seq.empty or len(df_seq) < sequence_length:
            return jsonify({
                "error": "Insufficient history to form an input sequence."
            }), 500

        last_sequence = df_seq["Close"].values[-sequence_length:]
        last_sequence = last_sequence.reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        current_sequence = torch.tensor(
            last_sequence_scaled, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        num_days = (target_date - last_date).days
        predictions_scaled = []
        for _ in range(num_days):
            with torch.no_grad():
                pred = model(current_sequence)
            predictions_scaled.append(pred.item())
            pred_tensor = pred.view(1, 1, 1)
            current_sequence = torch.cat(
                (current_sequence[:, 1:, :], pred_tensor), dim=1
            )

        final_pred_scaled = np.array([[predictions_scaled[-1]]])
        predicted_value = scaler.inverse_transform(final_pred_scaled)[0, 0]

        return jsonify({
            "date": date_str,
            "predicted_close_price": predicted_value,
            "note": "Prediction for a future date (untrained)"
        }), 200

    else:
        start_date_seq = (
            target_date - timedelta(days=sequence_length * 3)
        ).strftime("%Y-%m-%d")
        end_date_seq = target_date.strftime("%Y-%m-%d")
        df_seq = yf.download(ticker, start=start_date_seq,
                             end=end_date_seq, progress=False)

        if df_seq.empty or len(df_seq) < sequence_length:
            return jsonify({
                "error": "Insufficient history to form an input sequence "
                         "for the given date."
            }), 500

        df_seq = df_seq[df_seq.index < pd.Timestamp(target_date)]
        if len(df_seq) < sequence_length:
            return jsonify({
                "error": "Insufficient history to form an input sequence "
                         "for the given date."
            }), 500

        last_sequence = df_seq["Close"].values[-sequence_length:]
        last_sequence = last_sequence.reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)
        current_sequence = torch.tensor(
            last_sequence_scaled, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(current_sequence)
        predicted_scaled = pred.item()
        predicted_value = scaler.inverse_transform(
            np.array([[predicted_scaled]])
        )[0, 0]

        df_actual = yf.download(
            ticker,
            start=target_date.strftime("%Y-%m-%d"),
            end=(target_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False
        )
        if df_actual.empty:
            actual_value = None
        else:
            actual_value = float(df_actual["Close"].iloc[0])

        return jsonify({
            "date": date_str,
            "predicted_close_price": predicted_value,
            "actual_close_price": actual_value,
            "note": "Prediction for a trained date (for accuracy comparison)"
        }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
