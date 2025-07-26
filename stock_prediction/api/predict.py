import json
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("Stock_Predictions_Model.keras")

def handler(request, response):
    try:
        body = json.loads(request.body)
        ticker = body.get("ticker", "AAPL")
        start_date = body.get("start", "2015-01-01")
        end_date = body.get("end", "2020-01-01")

        data = yf.download(ticker, start=start_date, end=end_date)
        ma50 = data["Close"].rolling(window=50).mean().tolist()
        ma100 = data["Close"].rolling(window=100).mean().tolist()
        ma200 = data["Close"].rolling(window=200).mean().tolist()

        scaler = MinMaxScaler(feature_range=(0,1))
        data_train = pd.DataFrame(data["Close"][0:int(len(data)*0.80)])
        data_test = pd.DataFrame(data["Close"][int(len(data)*0.80):])

        past_100_days = data_train.tail(100)
        final_test = pd.concat([past_100_days, data_test], ignore_index=True)
        final_test_scaled = scaler.fit_transform(final_test)

        X_test = []
        for i in range(100, final_test_scaled.shape[0]):
            X_test.append(final_test_scaled[i-100:i])
        X_test = np.array(X_test)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(
            np.concatenate(
                [predictions, np.zeros((predictions.shape[0], 1))],
                axis=1
            )
        )[:,0]

        return response.json({
            "dates": data.index.strftime('%Y-%m-%d').tolist(),
            "close": data["Close"].tolist(),
            "ma50": ma50,
            "ma100": ma100,
            "ma200": ma200,
            "predictions": predictions.tolist()
        })
    except Exception as e:
        return response.status(500).json({"error": str(e)})
