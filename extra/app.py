import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

# Load model
model = load_model("Stock Predictions Model.keras")

# App configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    layout="wide",
)

# Custom CSS for dark blue theme
st.markdown("""
    <style>
    .reportview-container {
        background-color: #0A1A2F;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #0A1A2F;
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #1E3A5F;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Market Predictor")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2020-01-01"))

if st.sidebar.button("Predict"):

    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)

    st.subheader("Stock Data")
    st.write(data.tail())

    # Moving averages
    ma50 = data["Close"].rolling(window=50).mean()
    ma100 = data["Close"].rolling(window=100).mean()
    ma200 = data["Close"].rolling(window=200).mean()

    # Candlestick chart
    fig_candle = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick"
        ),
        go.Scatter(
            x=ma50.index,
            y=ma50,
            mode="lines",
            line=dict(color="cyan", width=1),
            name="MA50"
        ),
        go.Scatter(
            x=ma100.index,
            y=ma100,
            mode="lines",
            line=dict(color="orange", width=1),
            name="MA100"
        ),
        go.Scatter(
            x=ma200.index,
            y=ma200,
            mode="lines",
            line=dict(color="magenta", width=1),
            name="MA200"
        ),
    ])
    fig_candle.update_layout(
        title=f"{ticker} Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        autosize=True,
        height=600,
    )

    st.plotly_chart(fig_candle, use_container_width=True)

    # Prepare data for prediction
    data_train = pd.DataFrame(data["Close"][0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data["Close"][int(len(data)*0.80):])

    scaler = MinMaxScaler(feature_range=(0,1))
    past_100_days = data_train.tail(100)
    final_test = pd.concat([past_100_days, data_test], ignore_index=True)
    final_test_scaled = scaler.fit_transform(final_test)

    X_test = []
    y_test = []

    for i in range(100, final_test_scaled.shape[0]):
        X_test.append(final_test_scaled[i-100:i])
        y_test.append(final_test_scaled[i,0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], final_test_scaled.shape[1]-1))], axis=1))[:,0]
    y_test = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], final_test_scaled.shape[1]-1))], axis=1))[:,0]


    # Predicted vs Actual
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        y=y_test.flatten(),
        mode="lines",
        name="Actual Price",
        line=dict(color="lime")
    ))
    fig_pred.add_trace(go.Scatter(
        y=predictions.flatten(),
        mode="lines",
        name="Predicted Price",
        line=dict(color="red")
    ))
    fig_pred.update_layout(
        title="Predicted vs Actual Closing Price",
        xaxis_title="Time Step",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Moving averages chart (scrollable section)
    st.subheader("Additional Moving Averages Comparison")
    st.line_chart(data[["Close"]])

    st.write("Prediction completed successfully!")

