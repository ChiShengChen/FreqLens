"""
Data preprocessing utilities.
"""

import numpy as np
from typing import Optional


class StandardScaler:
    """Standard scaler for time series data."""
    
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
    
    def fit(self, data: np.ndarray):
        """Fit scaler to data."""
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0  # Avoid division by zero
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse transform")
        return data * self.std + self.mean


class MinMaxScaler:
    """Min-max scaler for time series data."""
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None
        self.feature_range = feature_range
    
    def fit(self, data: np.ndarray):
        """Fit scaler to data."""
        self.min = np.min(data, axis=0, keepdims=True)
        self.max = np.max(data, axis=0, keepdims=True)
        self.max[self.max == self.min] = self.min[self.max == self.min] + 1.0
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before transform")
        
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min)
        return scale * (data - self.min) + self.feature_range[0]
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min)
        return (data - self.feature_range[0]) / scale + self.min
