import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def prepare_data(df, future_hours):
    df["Target"] = df["Close"].shift(-future_hours)  
    df = df.dropna()

    X = df[["Close", "High", "Low", "Volume"]]
    y = df["Target"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(df, future_hours):
    X_train, X_test, y_train, y_test = prepare_data(df, future_hours)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    last_data = df[["Close", "High", "Low", "Volume"]].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(last_data)[0]

    return prediction
