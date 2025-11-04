"""
Model evaluation utilities for gold price prediction.

This module provides functions for:
- Computing evaluation metrics (RMSE, MAE, MAPE, R²)
- Saving evaluation results to CSV
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (MAPE).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAPE percentage
    """
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    nonzero = np.where(y_true != 0)[0]
    if len(nonzero) == 0:
        return np.nan
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary with RMSE, MAE, MAPE, R2 metrics
    """
    return {
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def evaluate_model(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray
) -> pd.DataFrame:
    """Evaluate model on train, validation, and test sets.
    
    Parameters
    ----------
    y_train_true : np.ndarray
        Training true values
    y_train_pred : np.ndarray
        Training predictions
    y_val_true : np.ndarray
        Validation true values
    y_val_pred : np.ndarray
        Validation predictions
    y_test_true : np.ndarray
        Test true values
    y_test_pred : np.ndarray
        Test predictions
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each split
    """
    results = {
        "Train": compute_metrics(y_train_true, y_train_pred),
        "Validation": compute_metrics(y_val_true, y_val_pred),
        "Test": compute_metrics(y_test_true, y_test_pred)
    }
    
    df = pd.DataFrame(results).T
    return df


def save_evaluation_results(
    results_df: pd.DataFrame,
    output_path: Path,
    append: bool = True
) -> None:
    """Save evaluation results to CSV.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    output_path : Path
        Output CSV file path
    append : bool
        Whether to append to existing file (default: True)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if append and output_path.exists():
        # Append with timestamp
        existing_df = pd.read_csv(output_path, index_col=0)
        # Add timestamp column
        results_df["Timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        combined_df = pd.concat([existing_df, results_df])
        combined_df.to_csv(output_path)
    else:
        results_df.to_csv(output_path)
    
    print(f"Evaluation results saved to: {output_path}")


def print_metrics_table(results_df: pd.DataFrame) -> None:
    """Print formatted metrics table.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    """
    print("\nEvaluation Metrics:")
    print(results_df.to_string(float_format=lambda x: f"{x:,.4f}"))

