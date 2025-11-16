"""
Prediction utilities for gold price forecasting.

This module provides functions for:
- Making predictions on test data
- Generating future forecasts
- Saving predictions to CSV
"""

import os

# Set environment variables BEFORE importing TensorFlow to prevent mutex lock errors on macOS
# This fixes the "mutex lock failed: Invalid argument" error
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Additional TensorFlow threading configuration as backup
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass  # Configuration may fail on some TensorFlow versions


def predict_future(
    model,
    scaler: MinMaxScaler,
    last_sequence: np.ndarray,
    n_days: int = 7,
    target_col_idx: int = 0,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate future predictions using the last sequence.

    Parameters
    ----------
    model
        Trained Keras model
    scaler : MinMaxScaler
        Fitted scaler for inverse transformation
    last_sequence : np.ndarray
        Last sequence of shape (seq_len, n_features)
    n_days : int
        Number of days to predict (default: 7)
    target_col_idx : int
        Index of target column in features (default: 0)

    Returns
    -------
    Tuple[np.ndarray, pd.DatetimeIndex]
        Predictions (n_days,) and future dates
    """
    seq_len = last_sequence.shape[0]
    current_window = last_sequence.copy()
    predictions_scaled = []

    for i in range(n_days):
        # Reshape for model input
        x_input = current_window.reshape(1, seq_len, -1)

        # Predict next value
        next_scaled = model.predict(x_input, verbose=0).squeeze()
        predictions_scaled.append(next_scaled)

        # Update window: shift and add prediction
        next_row = current_window[-1].copy()
        next_row[target_col_idx] = next_scaled
        current_window = np.vstack([current_window[1:], next_row])

    # Inverse transform predictions
    # Need to create full feature vectors with target at correct position
    n_features = last_sequence.shape[1]
    n_predictions = len(predictions_scaled)

    # Create dummy arrays with target at correct position
    dummy_features = np.zeros((n_predictions, n_features))
    dummy_features[:, target_col_idx] = np.array(predictions_scaled)

    # Inverse transform
    predictions = scaler.inverse_transform(dummy_features)[:, target_col_idx]

    # Generate future dates
    last_date = pd.Timestamp.now().normalize()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_days,
        freq="B",  # Business days
    )

    return predictions, future_dates


def make_predictions(
    model, X: np.ndarray, scaler: MinMaxScaler, target_col_idx: int = 0
) -> np.ndarray:
    """Make predictions on input sequences.

    Parameters
    ----------
    model
        Trained Keras model
    X : np.ndarray
        Input sequences (n_samples, seq_len, n_features)
    scaler : MinMaxScaler
        Fitted scaler for inverse transformation
    target_col_idx : int
        Index of target column (default: 0)

    Returns
    -------
    np.ndarray
        Inverse-transformed predictions
    """
    # Predict
    predictions_scaled = model.predict(X, verbose=0).squeeze()

    # Inverse transform
    if predictions_scaled.ndim == 1:
        predictions_scaled = predictions_scaled.reshape(-1, 1)

    # For inverse transform, we need full feature vector
    # Create a dummy array with target at the right position
    n_samples = predictions_scaled.shape[0]
    n_features = X.shape[2]

    # Create full feature arrays for inverse transform
    dummy_features = np.zeros((n_samples, n_features))
    dummy_features[:, target_col_idx] = predictions_scaled.squeeze()

    predictions = scaler.inverse_transform(dummy_features)[:, target_col_idx]

    return predictions


def save_predictions(
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    output_path: Path,
    column_name: str = "Predicted_Gold_Price",
) -> None:
    """Save predictions to CSV.

    Parameters
    ----------
    predictions : np.ndarray
        Prediction values
    dates : pd.DatetimeIndex
        Prediction dates
    output_path : Path
        Output CSV file path
    column_name : str
        Column name for predictions (default: "Predicted_Gold_Price")
    """
    df = pd.DataFrame({"Date": dates, column_name: predictions})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def load_model_and_scaler(
    model_path: Path, scaler_path: Path
) -> Tuple[any, MinMaxScaler]:
    """Load model and scaler from files.

    Parameters
    ----------
    model_path : Path
        Path to model file
    scaler_path : Path
        Path to scaler file

    Returns
    -------
    Tuple[model, MinMaxScaler]
        Loaded model and scaler
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


if __name__ == "__main__":
    # Example usage
    import sys
    from datetime import datetime
    from pathlib import Path

    # Use default paths if not provided via command line
    if len(sys.argv) >= 3:
        model_path = Path(sys.argv[1])
        scaler_path = Path(sys.argv[2])
    else:
        # Try to find the most recent model and scaler files
        models_dir = Path("models")
        notebooks_dir = Path("notebooks")

        # Look for model files in both directories
        model_candidates = []
        if models_dir.exists():
            model_candidates.extend(list(models_dir.glob("gold_lstm_*.h5")))
            model_candidates.extend(list(models_dir.glob("*.h5")))
        if notebooks_dir.exists():
            model_candidates.extend(list(notebooks_dir.glob("*.h5")))

        if model_candidates:
            # Get the most recent model file
            model_path = max(model_candidates, key=lambda p: p.stat().st_mtime)
            # Try to find corresponding scaler
            timestamp = (
                model_path.stem.split("_")[-1]
                if "_" in model_path.stem
                else datetime.now().strftime("%Y%m%d")
            )
            scaler_candidates = []
            if models_dir.exists():
                scaler_candidates.extend(
                    list(models_dir.glob(f"scaler_{timestamp}.pkl"))
                )
                scaler_candidates.extend(list(models_dir.glob("scaler_*.pkl")))
            if scaler_candidates:
                scaler_path = max(scaler_candidates, key=lambda p: p.stat().st_mtime)
            else:
                scaler_path = models_dir / "scaler.pkl"
            print("No paths provided, using defaults:")
            print(f"  Model: {model_path}")
            print(f"  Scaler: {scaler_path}")
        else:
            # Fallback to config defaults
            from config.config import config

            model_path = Path(config.MODEL_PATH)
            scaler_path = Path(config.SCALER_PATH)
            print("No paths provided, using config defaults:")
            print(f"  Model: {model_path}")
            print(f"  Scaler: {scaler_path}")

    # Load model and scaler
    if model_path.exists() and scaler_path.exists():
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        print("Prediction script ready!")
    else:
        print("Warning: Model or scaler file not found.")
        print(f"  Model exists: {model_path.exists()}")
        print(f"  Scaler exists: {scaler_path.exists()}")
        print("Prediction script ready (but model/scaler not loaded)!")
