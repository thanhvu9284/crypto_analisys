import streamlit as st
from data_fetcher import get_btc_data
from model import train_and_predict
from visualization import plot_btc_chart

st.title("📈 Dự báo giá Bitcoin 🚀")

df = get_btc_data()
st.plotly_chart(plot_btc_chart(df))

future_hours = {
    "1 Giờ": 1,
    "1 Ngày": 24,
    "1 Tuần": 24 * 7,
    "1 Tháng": 24 * 30
}

st.subheader("🔮 Dự báo giá Bitcoin:")
for key, hours in future_hours.items():
    prediction = train_and_predict(df, hours)
    st.write(f"📊 **{key} tới**: ${prediction:,.2f}")

st.write("📌 Dữ liệu từ Binance API và dự đoán bằng Machine Learning")
