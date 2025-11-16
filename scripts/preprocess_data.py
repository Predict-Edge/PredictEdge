"""
Data preprocessing utilities for gold price prediction.

This module provides functions for:
- Feature engineering (technical indicators, lag features)
- Data scaling and normalization
- Chronological train/val/test splitting
- Sequence creation for LSTM models
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Price series
    period : int
        RSI period (default: 14)

    Returns
    -------
    pd.Series
        RSI values
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)


def add_lag_features(
    df: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """Create lag features for given columns and lags.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Column names to create lags for
    lags : List[int]
        Lag periods to create

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features added
    """
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def preprocess_gold_data(
    raw_data: pd.DataFrame, target_col: str = "Gold", lags: List[int] = [1, 3, 5, 7, 10]
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Preprocess gold price data with feature engineering and scaling.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw gold price data with date index
    target_col : str
        Name of target column (default: "Gold")
    lags : List[int]
        Lag periods to create (default: [1, 3, 5, 7, 10])

    Returns
    -------
    Tuple[pd.DataFrame, MinMaxScaler]
        Preprocessed dataframe and fitted scaler
    """
    df = raw_data.copy()

    # Fix duplicated column names (e.g., 'Gold_Gold' -> 'Gold')
    df.columns = [c.split("_")[0] for c in df.columns]

    # Ensure target column exists
    if target_col not in df.columns:
        # Try to find a column containing 'Gold'
        gold_cols = [c for c in df.columns if "Gold" in c or "gold" in c.lower()]
        if gold_cols:
            target_col = gold_cols[0]
            print(f"Using '{target_col}' as target column")
        else:
            raise ValueError(
                f"Target column '{target_col}' not found. Available: {df.columns.tolist()}"
            )

    # Technical indicators for gold
    df[f"{target_col}_MA_7"] = df[target_col].rolling(window=7, min_periods=7).mean()
    df[f"{target_col}_MA_30"] = df[target_col].rolling(window=30, min_periods=30).mean()
    df[f"{target_col}_MA_90"] = df[target_col].rolling(window=90, min_periods=90).mean()

    df[f"{target_col}_Returns"] = df[target_col].pct_change()
    df[f"{target_col}_Volatility_30"] = (
        df[f"{target_col}_Returns"].rolling(window=30, min_periods=30).std()
    )
    df[f"{target_col}_RSI_14"] = compute_rsi(df[target_col], period=14)

    # Replace NaN values from rolling windows
    df = df.ffill().bfill()

    # Create lagged features for all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_lagged = add_lag_features(df, numeric_cols, lags)

    # Drop initial rows with NaN due to lagging
    df_lagged = df_lagged.dropna()

    # Scale all features to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(df_lagged.values)
    df_scaled = pd.DataFrame(
        scaled_values, index=df_lagged.index, columns=df_lagged.columns
    )

    return df_scaled, scaler


def create_sequences(
    features: np.ndarray, targets: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create LSTM sequences of length seq_len from aligned feature and target arrays.

    Parameters
    ----------
    features : np.ndarray
        Feature array (n_samples, n_features)
    targets : np.ndarray
        Target array (n_samples,)
    seq_len : int
        Sequence length for LSTM

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X sequences (n_samples - seq_len, seq_len, n_features) and y targets
    """
    X_list, y_list = [], []
    for i in range(seq_len, len(features)):
        X_list.append(features[i - seq_len : i])
        y_list.append(targets[i])
    return np.array(X_list), np.array(y_list)


def split_and_create_sequences(
    preprocessed_data: pd.DataFrame,
    target_col: str = "Gold",
    seq_len: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Split data chronologically and create LSTM sequences.

    Parameters
    ----------
    preprocessed_data : pd.DataFrame
        Preprocessed data with date index
    target_col : str
        Target column name
    seq_len : int
        Sequence length for LSTM
    train_ratio : float
        Training set ratio (default: 0.70)
    val_ratio : float
        Validation set ratio (default: 0.15)

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]
        Dictionary with 'train', 'val', 'test' keys containing (X, y, dates) tuples
    """
    df = preprocessed_data.copy()

    # Determine split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split dataframes
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Features and targets
    feature_cols = df.columns.tolist()
    X_train_all = train_df[feature_cols].values
    y_train_all = train_df[[target_col]].values.squeeze()

    X_val_all = val_df[feature_cols].values
    y_val_all = val_df[[target_col]].values.squeeze()

    X_test_all = test_df[feature_cols].values
    y_test_all = test_df[[target_col]].values.squeeze()

    # Create sequences
    X_train, y_train = create_sequences(X_train_all, y_train_all, seq_len)
    X_val, y_val = create_sequences(X_val_all, y_val_all, seq_len)
    X_test, y_test = create_sequences(X_test_all, y_test_all, seq_len)

    # Get dates for sequences (skip first seq_len rows)
    train_dates = train_df.index[seq_len:]
    val_dates = val_df.index[seq_len:]
    test_dates = test_df.index[seq_len:]

    print(f"Total samples: {n}")
    print(f"Train dates: {train_df.index[0].date()} -> {train_df.index[-1].date()}")
    print(f"Val dates: {val_df.index[0].date()} -> {val_df.index[-1].date()}")
    print(f"Test dates: {test_df.index[0].date()} -> {test_df.index[-1].date()}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return {
        "train": (X_train, y_train, train_dates),
        "val": (X_val, y_val, val_dates),
        "test": (X_test, y_test, test_dates),
    }


def save_splits(
    preprocessed_data: pd.DataFrame,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]],
    output_dir: Path,
    target_col: str = "Gold",
) -> None:
    """Save train/val/test splits to CSV files.

    Parameters
    ----------
    preprocessed_data : pd.DataFrame
        Full preprocessed dataframe
    splits : Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]
        Dictionary with splits
    output_dir : Path
        Base output directory (data/processed)
    target_col : str
        Target column name
    """
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save full preprocessed data
    full_path = output_dir / "gold_preprocessed_data.csv"
    preprocessed_data.to_csv(full_path)
    print(f"Saved preprocessed data to {full_path}")

    # Save splits
    for split_name, (X, y, dates) in splits.items():
        # Create dataframes for X and y
        # X is 3D (samples, timesteps, features), we'll save it flattened
        # For simplicity, save the last timestep of each sequence
        n_samples, seq_len, n_features = X.shape

        # Save last timestep features
        X_last = X[:, -1, :]  # Take last timestep
        X_df = pd.DataFrame(X_last, index=dates, columns=preprocessed_data.columns)
        X_df.to_csv(
            output_dir / split_name / "X_train.csv"
            if split_name == "train"
            else output_dir / split_name / f"X_{split_name}.csv"
        )

        # Save targets
        y_df = pd.DataFrame({target_col: y}, index=dates)
        y_df.to_csv(
            output_dir / split_name / "y_train.csv"
            if split_name == "train"
            else output_dir / split_name / f"y_{split_name}.csv"
        )

        print(f"Saved {split_name} split: X shape {X.shape}, y shape {y.shape}")


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Use default raw data path if not provided via command line
    if len(sys.argv) >= 2:
        raw_path = Path(sys.argv[1])
    else:
        raw_path = Path("data/raw/gold_raw_data.csv")
        print(f"No raw data path provided, using default: {raw_path}")

    output_dir = Path("data/processed")

    # Load raw data
    raw_data = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # Preprocess
    preprocessed, scaler = preprocess_gold_data(raw_data)

    # Split and create sequences
    splits = split_and_create_sequences(preprocessed, seq_len=60)

    # Save splits
    save_splits(preprocessed, splits, output_dir)

    print("Preprocessing complete!")
