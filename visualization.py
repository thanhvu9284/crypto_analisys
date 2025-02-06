import plotly.graph_objects as go

def plot_btc_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=df["Close"], mode="lines", name="Giá BTC"))
    fig.update_layout(title="Biểu đồ giá Bitcoin", xaxis_title="Thời gian", yaxis_title="Giá (USD)")
    return fig
