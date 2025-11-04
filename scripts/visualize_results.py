"""
Visualization utilities for gold price prediction results.

This module provides functions for creating publication-ready visualizations:
- Training/validation loss curves
- Actual vs Predicted plots
- Error distributions
- Feature correlation heatmaps
- Future predictions with confidence bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.dates as mdates

# Set style
plt.style.use("seaborn-v0_8")
sns.set_theme(style="whitegrid")

# Color scheme
COLORS = {
    "train": "#2E86AB",  # Blue
    "val": "#A23B72",  # Purple
    "test": "#F18F01",  # Orange
    "pred": "#C73E1D",  # Red
    "actual": "#06A77D",  # Green
}


def plot_training_history(
    history: Dict[str, list],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot training and validation loss over epochs.

    Parameters
    ----------
    history : Dict[str, list]
        Training history dictionary
    output_path : Optional[Path]
        Path to save figure (optional)
    figsize : Tuple[int, int]
        Figure size (default: (12, 6))
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(
        epochs, history["loss"], label="Train Loss", color=COLORS["train"], linewidth=2
    )
    ax.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        color=COLORS["val"],
        linewidth=2,
    )

    ax.set_title("Training History: Loss over Epochs", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history plot to: {output_path}")

    plt.show()


def plot_actual_vs_predicted(
    train_dates: pd.DatetimeIndex,
    train_true: np.ndarray,
    train_pred: np.ndarray,
    val_dates: pd.DatetimeIndex,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_dates: pd.DatetimeIndex,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """Plot actual vs predicted prices for train/val/test sets.

    Parameters
    ----------
    train_dates : pd.DatetimeIndex
        Training dates
    train_true : np.ndarray
        Training actual values
    train_pred : np.ndarray
        Training predictions
    val_dates : pd.DatetimeIndex
        Validation dates
    val_true : np.ndarray
        Validation actual values
    val_pred : np.ndarray
        Validation predictions
    test_dates : pd.DatetimeIndex
        Test dates
    test_true : np.ndarray
        Test actual values
    test_pred : np.ndarray
        Test predictions
    output_path : Optional[Path]
        Path to save figure (optional)
    figsize : Tuple[int, int]
        Figure size (default: (16, 8))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot train
    ax.plot(
        train_dates,
        train_true,
        label="Train Actual",
        color=COLORS["train"],
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        train_dates,
        train_pred,
        label="Train Predicted",
        color=COLORS["train"],
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
    )

    # Plot validation
    ax.plot(
        val_dates,
        val_true,
        label="Validation Actual",
        color=COLORS["val"],
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        val_dates,
        val_pred,
        label="Validation Predicted",
        color=COLORS["val"],
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
    )

    # Plot test
    ax.plot(
        test_dates,
        test_true,
        label="Test Actual",
        color=COLORS["test"],
        alpha=0.7,
        linewidth=1.5,
    )
    ax.plot(
        test_dates,
        test_pred,
        label="Test Predicted",
        color=COLORS["test"],
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
    )

    ax.set_title("Actual vs Predicted Gold Prices", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved actual vs predicted plot to: {output_path}")

    plt.show()


def plot_error_distribution(
    errors: np.ndarray,
    split_name: str = "Test",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Plot prediction error distribution with KDE.

    Parameters
    ----------
    errors : np.ndarray
        Prediction errors (actual - predicted)
    split_name : str
        Name of data split (default: "Test")
    output_path : Optional[Path]
        Path to save figure (optional)
    figsize : Tuple[int, int]
        Figure size (default: (10, 6))
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(errors, bins=50, kde=True, ax=ax, color=COLORS["test"], alpha=0.7)

    # Add vertical line at zero
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add mean line
    mean_error = np.mean(errors)
    ax.axvline(
        x=mean_error,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_error:.2f}",
        alpha=0.7,
    )

    ax.set_title(
        f"Prediction Error Distribution ({split_name})", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Error (Actual - Predicted)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved error distribution plot to: {output_path}")

    plt.show()


def plot_feature_correlation(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Plot feature correlation heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Feature dataframe
    output_path : Optional[Path]
        Path to save figure (optional)
    figsize : Tuple[int, int]
        Figure size (default: (14, 10))
    """
    # Compute correlation
    corr = data.corr()

    # Select top features if too many (for readability)
    if len(corr) > 50:
        # Select features with highest variance in correlations
        corr_var = corr.abs().sum().sort_values(ascending=False)
        top_features = corr_var.head(50).index
        corr = corr.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        cbar_kws={"shrink": 0.6},
        square=True,
        ax=ax,
        vmin=-1,
        vmax=1,
    )

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation heatmap to: {output_path}")

    plt.show()


def plot_last_n_days_with_confidence(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    n_days: int = 100,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """Plot last N days with confidence bands.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Date index
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
    n_days : int
        Number of days to show (default: 100)
    output_path : Optional[Path]
        Path to save figure (optional)
    figsize : Tuple[int, int]
        Figure size (default: (14, 8))
    """
    last_n = min(n_days, len(dates))
    last_dates = dates[-last_n:]
    last_actual = actual[-last_n:]
    last_pred = predicted[-last_n:]

    # Compute confidence bands (95%)
    errors = last_actual - last_pred
    resid_std = np.std(errors)
    upper = last_pred + 1.96 * resid_std
    lower = last_pred - 1.96 * resid_std

    fig, ax = plt.subplots(figsize=figsize)

    # Confidence band
    ax.fill_between(
        last_dates,
        lower,
        upper,
        alpha=0.3,
        color=COLORS["test"],
        label="95% Confidence Band",
    )

    # Actual and predicted
    ax.plot(
        last_dates,
        last_actual,
        label="Actual",
        color=COLORS["actual"],
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.plot(
        last_dates,
        last_pred,
        label="Predicted",
        color=COLORS["pred"],
        linewidth=2,
        marker="s",
        markersize=3,
    )

    ax.set_title(
        f"Last {last_n} Days: Prediction with Confidence Bands",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved last {last_n} days plot to: {output_path}")

    plt.show()


def plot_future_predictions(
    historical_dates: pd.DatetimeIndex,
    historical_actual: np.ndarray,
    future_dates: pd.DatetimeIndex,
    future_pred: np.ndarray,
    n_history: int = 200,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """Plot future predictions with historical context.

    Parameters
    ----------
    historical_dates : pd.DatetimeIndex
        Historical dates
    historical_actual : np.ndarray
        Historical actual values
    future_dates : pd.DatetimeIndex
        Future dates
    future_pred : np.ndarray
        Future predictions
    n_history : int
        Number of historical days to show (default: 200)
    output_path : Optional[Path]
        Path to save figure (optional)
    figsize : Tuple[int, int]
        Figure size (default: (16, 8))
    """
    n_history = min(n_history, len(historical_dates))
    hist_dates = historical_dates[-n_history:]
    hist_actual = historical_actual[-n_history:]

    fig, ax = plt.subplots(figsize=figsize)

    # Historical
    ax.plot(
        hist_dates,
        hist_actual,
        label="Historical Actual",
        color=COLORS["actual"],
        linewidth=2,
    )

    # Future
    ax.plot(
        future_dates,
        future_pred,
        label="Future Predicted (7 days)",
        color=COLORS["pred"],
        linewidth=2,
        marker="o",
        markersize=8,
    )

    # Add vertical line separator
    ax.axvline(
        x=hist_dates[-1],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Forecast Start",
    )

    ax.set_title(
        "Future 7-Day Forecast with Historical Context", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved future predictions plot to: {output_path}")

    plt.show()
