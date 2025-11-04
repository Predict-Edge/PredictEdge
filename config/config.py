import os

from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file


class Config:
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/gold_lstm.h5")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.gz")

    # Broker API keys (example)
    BROKER_API_KEY = os.getenv("BROKER_API_KEY", "")
    BROKER_API_SECRET = os.getenv("BROKER_API_SECRET", "")

    # Other settings
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")


config = Config()
