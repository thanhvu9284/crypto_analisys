import matplotlib.pyplot as plt

# Dự đoán giá
def predict_btc(model, X_test, scaler):
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)  # Chuyển giá trị về dạng ban đầu
    return predicted_price

# Vẽ đồ thị so sánh giữa giá thực tế và giá dự đoán
def plot_predictions(y_test, predicted_price):
    plt.plot(y_test, color='blue', label='Giá thực tế Bitcoin')
    plt.plot(predicted_price, color='red', label='Giá dự đoán Bitcoin')
    plt.title('Dự đoán giá Bitcoin')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá Bitcoin')
    plt.legend()
    plt.show()
