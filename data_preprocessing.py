import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

# Tải dữ liệu Bitcoin từ Yahoo Finance
def load_btc_data(start='2021-01-01', end='2025-01-01', interval='1h'):
    try:
        # Tải dữ liệu với Yahoo Finance
        btc_data = yf.download('BTC-USD', start=start, end=end, interval=interval)
        
        # Kiểm tra nếu dữ liệu trả về trống
        if btc_data.empty:
            raise ValueError("Dữ liệu không tải về, vui lòng kiểm tra lại kết nối hoặc phạm vi thời gian.")
        
        # Chỉ lấy cột 'Close' (giá đóng cửa)
        btc_data = btc_data[['Close']]  
        
        # Reset chỉ mục để có thể xử lý dữ liệu chính xác hơn
        btc_data.reset_index(inplace=True)  
        
        # Đảm bảo chỉ có giá trị số được đưa vào
        btc_data['Date'] = pd.to_datetime(btc_data['Date'])  # Đảm bảo rằng cột 'Date' là datetime

        return btc_data

    except Exception as e:
        print(f"Đã có lỗi xảy ra khi tải dữ liệu: {e}")
        return None

# Tiền xử lý dữ liệu: chuẩn hóa và tạo dữ liệu đầu vào cho LSTM
def preprocess_data(df, time_step=60):
    if df is None or df.empty:
        raise ValueError("Dữ liệu không hợp lệ!")

    # Chỉ chuẩn hóa cột giá đóng cửa 'Close'
    data = df[['Close']].values
    
    # Kiểm tra nếu cột giá đóng cửa không phải số
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Cột giá đóng cửa không phải là số!")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])  # Các giá trị trước thời gian i
        y.append(data_scaled[i, 0])  # Giá trị tại thời điểm i
    
    X, y = np.array(X), np.array(y)
    
    # Kiểm tra dữ liệu sau khi tách
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Không đủ dữ liệu để huấn luyện mô hình!")

    # Chia dữ liệu thành train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape dữ liệu cho LSTM (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, scaler
