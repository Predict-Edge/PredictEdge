"""
Prediction utilities for gold price forecasting.

This module provides functions for:
- Making predictions on test data
- Generating future forecasts
- Saving predictions to CSV
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


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
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: python run_prediction.py <model_path> <scaler_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    scaler_path = Path(sys.argv[2])

    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    print("Prediction script ready!")
