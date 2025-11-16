"""
LSTM model training utilities for gold price prediction.

This module provides functions for:
- Building LSTM models
- Training with callbacks
- Saving models with timestamps
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed (default: 42)
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # Not all TensorFlow versions support this


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
) -> Sequential:
    """Build an LSTM model for time series prediction.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Input shape (sequence_length, n_features)
    lstm_units : int
        Number of LSTM units per layer (default: 50)
    dropout_rate : float
        Dropout rate (default: 0.2)
    dense_units : int
        Number of dense layer units (default: 25)

    Returns
    -------
    Sequential
        Compiled LSTM model
    """
    model = Sequential(
        [
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
    verbose: int = 1,
) -> Tuple[Sequential, Dict[str, Any]]:
    """Train an LSTM model with early stopping and model checkpointing.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences (n_samples, seq_len, n_features)
    y_train : np.ndarray
        Training targets (n_samples,)
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation targets
    epochs : int
        Maximum number of epochs (default: 100)
    batch_size : int
        Batch size (default: 32)
    patience : int
        Early stopping patience (default: 10)
    lstm_units : int
        Number of LSTM units (default: 50)
    dropout_rate : float
        Dropout rate (default: 0.2)
    dense_units : int
        Dense layer units (default: 25)
    verbose : int
        Verbosity level (default: 1)

    Returns
    -------
    Tuple[Sequential, Dict[str, Any]]
        Trained model and training history
    """
    # Set random seed
    set_random_seed(42)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        )
    ]

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    return model, history.history


def save_model_with_timestamp(
    model: Sequential, scaler: MinMaxScaler, models_dir: Path, prefix: str = "gold_lstm"
) -> Tuple[Path, Path]:
    """Save model and scaler with timestamp.

    Parameters
    ----------
    model : Sequential
        Trained Keras model
    scaler : MinMaxScaler
        Fitted scaler
    models_dir : Path
        Models directory
    prefix : str
        File prefix (default: "gold_lstm")

    Returns
    -------
    Tuple[Path, Path]
        Paths to saved model and scaler files
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d")

    # Save model
    model_path = models_dir / f"{prefix}_{timestamp}.h5"
    model.save(model_path)

    # Save scaler
    scaler_path = models_dir / f"scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    return model_path, scaler_path


def load_trained_model(
    model_path: Path, scaler_path: Path
) -> Tuple[Sequential, MinMaxScaler]:
    """Load trained model and scaler.

    Parameters
    ----------
    model_path : Path
        Path to model file
    scaler_path : Path
        Path to scaler file

    Returns
    -------
    Tuple[Sequential, MinMaxScaler]
        Loaded model and scaler
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python train_lstm.py <data_dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    models_dir = Path("models")

    # Load sequences (this would come from preprocessing)
    # X_train, y_train, X_val, y_val = load_sequences(data_dir)

    # Train model
    # model, history = train_lstm_model(X_train, y_train, X_val, y_val)

    # Save model
    # save_model_with_timestamp(model, scaler, models_dir)

    print("Training script ready!")
