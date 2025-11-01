import pandas as pd
import joblib
from models.lstm.model import train_lstm_model

df = pd.read_csv('data/gold_prices.csv')
model, scaler = train_lstm_model(df, epochs=20)

model.save('models/gold_lstm.h5')
joblib.dump(scaler, 'models/scaler.gz')

print("Training complete and artifacts saved.")
