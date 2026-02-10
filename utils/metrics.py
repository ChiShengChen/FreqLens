"""
Evaluation metrics for time series forecasting.
"""

import numpy as np
import torch


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        threshold: Minimum absolute value of y_true to include in MAPE calculation
                   (to avoid division by very small numbers)
    """
    # Filter out near-zero values to avoid MAPE explosion
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        # If all values are near zero, return a large value or NaN
        return np.nan
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    epsilon = 1e-8
    mape_values = np.abs((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon)) * 100
    
    # Clip extreme values (e.g., > 1000%) to avoid outliers dominating
    mape_values = np.clip(mape_values, 0, 1000)
    
    return np.mean(mape_values)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
