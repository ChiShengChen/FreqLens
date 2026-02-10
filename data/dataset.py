"""
PyTorch Dataset class for time series data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 96,
        label_len: int = 48,
        target_col: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data: Time series data array (T, features)
            seq_len: Input sequence length
            pred_len: Prediction length
            label_len: Label length for decoder
            target_col: Target column index (None for multivariate)
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.target_col = target_col
        
        # Calculate valid samples
        self.total_len = seq_len + pred_len
        self.valid_indices = list(range(len(data) - self.total_len + 1))
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            x_enc: Encoder input (seq_len, features)
            x_mark_enc: Encoder time marks (optional, can be None)
            y_dec: Decoder input (label_len + pred_len, features)
            y_mark_dec: Decoder time marks (optional, can be None)
        """
        s_begin = self.valid_indices[idx]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Encoder input
        x_enc = self.data[s_begin:s_end]
        
        # Decoder input (label + prediction)
        y_dec = self.data[r_begin:r_end]
        
        # Select target if specified
        if self.target_col is not None:
            x_enc = x_enc[:, self.target_col:self.target_col+1]
            y_dec = y_dec[:, self.target_col:self.target_col+1]
        
        # Convert to tensors
        x_enc = torch.FloatTensor(x_enc)
        y_dec = torch.FloatTensor(y_dec)
        
        # Split decoder into label and prediction
        y_label = y_dec[:self.label_len]
        y_pred = y_dec[self.label_len:]
        
        # Placeholder for time marks (can be extended later)
        # Return empty tensors instead of None to avoid collate issues
        x_mark_enc = torch.zeros(self.seq_len, 0)  # Empty tensor
        y_mark_dec = torch.zeros(self.label_len + self.pred_len, 0)  # Empty tensor
        
        return x_enc, x_mark_enc, y_label, y_pred
