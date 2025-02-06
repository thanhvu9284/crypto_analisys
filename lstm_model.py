import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import get_btc_data

# Chuẩn bị dữ liệu
df = get_btc_data('1h', 100)
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['close']])

X, y = [], []
for i in range(10, len(df_scaled)-1):
    X.append(df_scaled[i-10:i])
    y.append(df_scaled[i])

X, y = np.array(X), np.array(y)

# Xây dựng mô hình LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=16)

# Dự đoán giá BTC tiếp theo
future_pred = model.predict(X[-1].reshape(1, X.shape[1], 1))
future_pred = scaler.inverse_transform(future_pred)
print(f"🔥 Dự đoán giá BTC tiếp theo: {future_pred[0][0]:,.2f} USD")
