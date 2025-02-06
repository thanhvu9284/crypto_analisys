import streamlit as st
from data_fetcher import get_btc_data
from plot_chart import plot_candlestick
from lstm_model import future_pred

st.title("📈 Ứng Dụng Phân Tích & Dự Đoán Giá Bitcoin")
st.write("Biểu đồ giá BTC theo giờ, ngày, tháng")

#df_hourly = get_btc_data('1h', 48)
#df_daily = get_btc_data('1d', 30)
#df_monthly = get_btc_data('1M', 12)

#st.plotly_chart(plot_candlestick(df_hourly, "Giá BTC theo giờ"))
#st.plotly_chart(plot_candlestick(df_daily, "Giá BTC theo ngày"))
#st.plotly_chart(plot_candlestick(df_monthly, "Giá BTC theo tháng"))

st.write(f"🔥 Dự đoán giá BTC tiếp theo: {future_pred[0][0]:,.2f} USD")
