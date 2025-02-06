from data_preprocessing import load_btc_data, preprocess_data
from lstm_model import train_lstm_model
from predict import predict_btc, plot_predictions

def main():
    # Lấy dữ liệu và tiền xử lý
    df = load_btc_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Huấn luyện mô hình LSTM
    model = train_lstm_model(X_train, y_train, X_test, y_test)

    # Dự đoán giá Bitcoin
    predicted_price = predict_btc(model, X_test, scaler)

    # Vẽ đồ thị so sánh giữa giá thực tế và giá dự đoán
    plot_predictions(y_test, predicted_price)

if __name__ == "__main__":
    main()
