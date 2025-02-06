import plotly.graph_objects as go

def plot_candlestick(df, title="Biểu đồ giá BTC"):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'])])
    fig.update_layout(title=title, xaxis_title="Thời gian", yaxis_title="Giá BTC (USDT)", template="plotly_dark")
    return fig
