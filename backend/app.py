# import joblib
# import numpy as np
from fastapi import FastAPI

# from pydantic import BaseModel

# from tensorflow.keras.models import load_model

app = FastAPI()

# model = load_model("models/gold_lstm.h5")
# scaler = joblib.load("models/scaler.gz")

# class PriceHistory(BaseModel):
#     prices: list[float]

# @app.post("/predict")
# def predict(data: PriceHistory):
#     input_data = np.array(data.prices).reshape(-1, 1)
#     scaled = scaler.transform(input_data)
#     look_back = 60
#     X = np.expand_dims(scaled[-look_back:], axis=(0, -1))
#     pred_scaled = model.predict(X)
#     pred = scaler.inverse_transform(pred_scaled)
#     return {"predicted_price": float(pred.flatten()[0])}


@app.get("/")
def root():
    return {"message": "Hello, World!"}
