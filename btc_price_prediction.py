import streamlit as st
from binance.client import Client
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# Tiêu đề ứng dụng
st.title('Dự đoán giá Bitcoin (BTC) từ Binance với ARIMA')

# Khởi tạo client Binance (không cần API key nếu chỉ lấy dữ liệu public)
client = Client()

# Lấy dữ liệu giá BTC từ Binance
@st.cache_data
def load_data():
    klines = client.get_historical_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_str="2020-01-01",
        end_str=datetime.now().strftime('%Y-%m-%d')
    )
    
    data = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    
    data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
    data['Close'] = data['Close'].astype(float)
    
    return data[['Open time', 'Close']].rename(columns={"Open time": "Date", "Close": "Close"})

data = load_data()

# Hiển thị dữ liệu
st.subheader('Dữ liệu giá BTC từ Binance')
st.write(data.tail())

# Kiểm tra dữ liệu có đủ không
if len(data) < 50:
    st.error("Dữ liệu quá ít để huấn luyện mô hình ARIMA. Hãy thử tải lại sau!")
    st.stop()

# Huấn luyện mô hình ARIMA
st.subheader("Huấn luyện mô hình ARIMA")
try:
    model = ARIMA(data['Close'], order=(2, 1, 2))
    model_fit = model.fit()
    st.success("Huấn luyện mô hình ARIMA thành công!")
except Exception as e:
    st.error(f"Lỗi khi huấn luyện mô hình ARIMA: {e}")
    st.stop()

# Dự đoán giá BTC
future_periods = [1, 24, 24*7]  # 1 giờ, 1 ngày, 1 tuần
forecast_results = {}

for period in future_periods:
    try:
        forecast = model_fit.forecast(steps=period)
        
        # Kiểm tra nếu dự đoán rỗng
        if len(forecast) > 0:
            forecast_results[period] = forecast.iloc[-1]
        else:
            forecast_results[period] = np.nan
    except Exception as e:
        forecast_results[period] = np.nan
        st.warning(f"Lỗi khi dự đoán cho {period} bước: {e}")

# Hiển thị kết quả dự đoán
st.subheader('Dự đoán giá BTC')
st.write(f"📌 **1 giờ tới**: {forecast_results[1]:,.2f} USDT")
st.write(f"📌 **1 ngày tới**: {forecast_results[24]:,.2f} USDT")
st.write(f"📌 **1 tuần tới**: {forecast_results[24*7]:,.2f} USDT")

# Vẽ biểu đồ giá thực tế và dự đoán
st.subheader('Biểu đồ dự đoán giá BTC')
fig = go.Figure()

fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Giá thực tế'))
fig.add_trace(go.Scatter(x=[data['Date'].iloc[-1] + pd.Timedelta(hours=h) for h in future_periods], 
                         y=[forecast_results[h] for h in future_periods], 
                         mode='markers+lines', name='Dự đoán', marker=dict(size=8, color='red')))

st.plotly_chart(fig)
