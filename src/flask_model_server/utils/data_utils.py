import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from config import LAST_UPDATE_FILE

def download_data_from_yfinance(start_date, ticker):
    end_date = datetime.today().date()
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df if not df.empty else None

def preprocess_data(df, sequence_length, device):
    df = df[["Close"]].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(df_scaled)):
        X.append(df_scaled[i - sequence_length:i])
        y.append(df_scaled[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler
