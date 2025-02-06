import requests
import pandas as pd

def get_btc_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=500"
    data = requests.get(url).json()

    df = pd.DataFrame(data, columns=["Time", "Open", "High", "Low", "Close", "Volume",
                                     "CloseTime", "_", "_", "_", "_", "_"])
    
    df["Time"] = pd.to_datetime(df["Time"], unit='ms')
    df["Close"] = df["Close"].astype(float)

    return df
