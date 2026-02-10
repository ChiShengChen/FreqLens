"""
Unified data loader for time series datasets.

支持的數據集：
- ETTh1, ETTh2: 小時粒度，60/20/20 分割
- ETTm1, ETTm2: 15分鐘粒度，60/20/20 分割
- Electricity: 小時粒度，70/10/20 分割
- Weather: 10分鐘粒度，70/10/20 分割
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader as TorchDataLoader
from .dataset import TimeSeriesDataset
from .preprocessing import StandardScaler


# Data root: set FREQLENS_DATA_PATH to your dataset directory (see DATA_SETUP.md).
_env_path = os.environ.get("FREQLENS_DATA_PATH", "").strip()
if _env_path and os.path.exists(_env_path):
    BASE_PATH = _env_path
else:
    raise FileNotFoundError(
        "FREQLENS_DATA_PATH is not set or path does not exist. "
        "Set it to your dataset root, e.g.: export FREQLENS_DATA_PATH=/path/to/all_six_datasets"
    )

# 數據集配置
DATASET_CONFIGS = {
    'ETTh1': {
        'path': f"{BASE_PATH}/ETT-small/ETTh1.csv",
        'target': 'OT',
        'split_ratio': (0.6, 0.2, 0.2),  # 60/20/20
    },
    'ETTh2': {
        'path': f"{BASE_PATH}/ETT-small/ETTh2.csv",
        'target': 'OT',
        'split_ratio': (0.6, 0.2, 0.2),  # 60/20/20
    },
    'ETTm1': {
        'path': f"{BASE_PATH}/ETT-small/ETTm1.csv",
        'target': 'OT',
        'split_ratio': (0.6, 0.2, 0.2),  # 60/20/20
    },
    'ETTm2': {
        'path': f"{BASE_PATH}/ETT-small/ETTm2.csv",
        'target': 'OT',
        'split_ratio': (0.6, 0.2, 0.2),  # 60/20/20
    },
    'electricity': {
        'path': f"{BASE_PATH}/electricity/electricity.csv",
        'target': 'OT',
        'split_ratio': (0.7, 0.1, 0.2),  # 70/10/20
    },
    'weather': {
        'path': f"{BASE_PATH}/weather/weather.csv",
        'target': 'OT',
        'split_ratio': (0.7, 0.1, 0.2),  # 70/10/20
    },
    'traffic': {
        'path': f"{BASE_PATH}/traffic/traffic.csv",
        'target': 'OT',
        'split_ratio': (0.7, 0.1, 0.2),  # 70/10/20
    },
    'exchange_rate': {
        'path': f"{BASE_PATH}/exchange_rate/exchange_rate.csv",
        'target': 'OT',
        'split_ratio': (0.7, 0.1, 0.2),  # 70/10/20
    },
    'ili': {
        'path': f"{BASE_PATH}/illness/national_illness.csv",
        'target': 'OT',
        'split_ratio': (0.7, 0.1, 0.2),  # 70/10/20
    },
}


class TimeSeriesDataLoader:
    """
    統一時間序列數據加載器
    
    遵循標準 benchmark 設定：
    - ETT 數據集：12 months train / 4 months val / 4 months test (60/20/20)
    - Electricity/Weather：70/10/20
    """
    
    def __init__(
        self,
        data_path: str,
        seq_len: int = 96,
        pred_len: int = 96,
        label_len: int = 48,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        use_paper_splits: bool = False,
        dataset_name: Optional[str] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV file
            seq_len: Input sequence length (96, 192, 336, 512)
            pred_len: Prediction length (96, 192, 336, 720)
            label_len: Label length for decoder
            features: Feature mode ('M': multivariate, 'S': univariate, 'MS': multi-input single-output)
            target: Target column name (default: 'OT' for ETT)
            scale: Whether to standardize (default: True)
            train_ratio: Training set ratio (auto if None)
            val_ratio: Validation set ratio (auto if None)
            test_ratio: Test set ratio (auto if None)
        """
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features
        self.target = target
        self.scale = scale
        
        # Load and preprocess data
        self.df_data = self._load_data()
        
        # Determine split ratios
        if train_ratio is None or val_ratio is None or test_ratio is None:
            # Use default ratios based on dataset
            train_ratio, val_ratio, test_ratio = self._get_default_split_ratios()
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        
        # Initialize scaler (will be fitted on training data)
        self.scaler = StandardScaler() if scale else None
        
        # Split data
        self.train_data, self.val_data, self.test_data = self._split_data(
            train_ratio, val_ratio, test_ratio,
            use_paper_splits=use_paper_splits,
            dataset_name=dataset_name,
        )
    
    def _get_default_split_ratios(self) -> Tuple[float, float, float]:
        """Get default split ratios based on dataset name."""
        # Check if this is an ETT dataset
        if 'ETT' in self.data_path:
            return (0.6, 0.2, 0.2)  # 60/20/20
        else:
            return (0.7, 0.1, 0.2)  # 70/10/20
    
    def _get_paper_exact_splits(self, dataset_name: str, seq_len: int) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Get paper exact split borders for ETT datasets (matching official LTSF-Linear code).
        
        Official LTSF-Linear logic:
        - ETTh1/ETTh2: 
          border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
          border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        - ETTm1/ETTm2:
          border1s = [0, 12*30*24*4 - seq_len, 12*30*24*4 + 4*30*24*4 - seq_len]
          border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
        
        Returns:
            Dict with 'train', 'val', 'test' as (border1, border2) tuples, or None
        """
        # Extract dataset name from path
        if 'ETTh1' in dataset_name or 'ETTh1' in self.data_path:
            # ETTh1/ETTh2: hourly data
            # 12 months = 12 * 30 * 24 = 8640
            # 4 months = 4 * 30 * 24 = 2880
            # 8 months = 8 * 30 * 24 = 5760
            train_end = 12 * 30 * 24  # 8640
            val_end = train_end + 4 * 30 * 24  # 11520
            test_end = train_end + 8 * 30 * 24  # 14400
            
            return {
                'train': (0, train_end - seq_len),  # [0, 8640 - seq_len]
                'val': (train_end - seq_len, val_end - seq_len),  # [8640 - seq_len, 11520 - seq_len]
                'test': (val_end - seq_len, test_end),  # [11520 - seq_len, 14400]
            }
        elif 'ETTh2' in dataset_name or 'ETTh2' in self.data_path:
            # Same as ETTh1
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = train_end + 8 * 30 * 24
            
            return {
                'train': (0, train_end - seq_len),
                'val': (train_end - seq_len, val_end - seq_len),
                'test': (val_end - seq_len, test_end),
            }
        elif 'ETTm1' in dataset_name or 'ETTm1' in self.data_path:
            # ETTm1/ETTm2: 15-minute data
            # 12 months = 12 * 30 * 24 * 4 = 34560
            # 4 months = 4 * 30 * 24 * 4 = 11520
            # 8 months = 8 * 30 * 24 * 4 = 23040
            train_end = 12 * 30 * 24 * 4  # 34560
            val_end = train_end + 4 * 30 * 24 * 4  # 46080
            test_end = train_end + 8 * 30 * 24 * 4  # 57600
            
            return {
                'train': (0, train_end - seq_len),
                'val': (train_end - seq_len, val_end - seq_len),
                'test': (val_end - seq_len, test_end),
            }
        elif 'ETTm2' in dataset_name or 'ETTm2' in self.data_path:
            # Same as ETTm1
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = train_end + 8 * 30 * 24 * 4
            
            return {
                'train': (0, train_end - seq_len),
                'val': (train_end - seq_len, val_end - seq_len),
                'test': (val_end - seq_len, test_end),
            }
        
        return None
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Handle date column (usually first column)
        if df.columns[0].lower() in ['date', 'time', 'timestamp']:
            df = df.drop(df.columns[0], axis=1)
        
        # Select features based on feature mode
        if self.features == 'M':
            # Multivariate: use all features
            cols_data = df.columns.tolist()
        elif self.features == 'S':
            # Univariate: only target
            if self.target not in df.columns:
                # If target not found, use first column
                self.target = df.columns[0]
            cols_data = [self.target]
        elif self.features == 'MS':
            # Multi-input single-output: all features as input, target as output
            cols_data = df.columns.tolist()
        else:
            cols_data = df.columns.tolist()
        
        df_data = df[cols_data]
        
        return df_data
    
    def _split_data(
        self, 
        train_ratio: float, 
        val_ratio: float, 
        test_ratio: float,
        use_paper_splits: bool = False,
        dataset_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.
        
        標準化處理：
        - 使用訓練集的 mean/std 進行標準化
        - 保存 scaler 用於反標準化預測結果
        
        Args:
            train_ratio: Training set ratio (ignored if use_paper_splits=True)
            val_ratio: Validation set ratio (ignored if use_paper_splits=True)
            test_ratio: Test set ratio (ignored if use_paper_splits=True)
            use_paper_splits: If True, use paper exact split counts for ETT datasets
            dataset_name: Dataset name for paper splits lookup
        """
        data = self.df_data.values.astype(np.float32)
        n = len(data)
        
        # Try to use paper exact splits if requested (matching official LTSF-Linear code)
        if use_paper_splits:
            paper_borders = self._get_paper_exact_splits(dataset_name or self.data_path, self.seq_len)
            if paper_borders is not None:
                train_border1, train_border2 = paper_borders['train']
                val_border1, val_border2 = paper_borders['val']
                test_border1, test_border2 = paper_borders['test']
                
                # Check if data is long enough
                max_needed = max(train_border2, val_border2, test_border2)
                if n >= max_needed:
                    # Use exact borders (matching official code)
                    train_data_raw = data[train_border1:train_border2]
                    val_data_raw = data[val_border1:val_border2]
                    test_data_raw = data[test_border1:test_border2]
                    
                    if n > max_needed:
                        print(f"ℹ️ Info: Data length ({n}) > expected paper split ({max_needed}), using borders matching official code")
                else:
                    # Data is shorter than expected, use ratio but warn
                    print(f"⚠️ Warning: Data length ({n}) < expected paper split ({max_needed}), using ratio split")
                    train_end = int(n * train_ratio)
                    val_end = int(n * (train_ratio + val_ratio))
                    train_data_raw = data[:train_end]
                    val_data_raw = data[train_end:val_end]
                    test_data_raw = data[val_end:]
            else:
                # No paper splits available, use ratio
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                train_data_raw = data[:train_end]
                val_data_raw = data[train_end:val_end]
                test_data_raw = data[val_end:]
        else:
            # Calculate split indices using ratios
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            # Split raw data
            train_data_raw = data[:train_end]
            val_data_raw = data[train_end:val_end]
            test_data_raw = data[val_end:]
        
        # Scale data using training set statistics
        if self.scaler is not None:
            # Fit scaler on training data only
            self.scaler.fit(train_data_raw)
            
            # Transform all splits
            train_data = self.scaler.transform(train_data_raw)
            val_data = self.scaler.transform(val_data_raw)
            test_data = self.scaler.transform(test_data_raw)
        else:
            train_data = train_data_raw
            val_data = val_data_raw
            test_data = test_data_raw
        
        return train_data, val_data, test_data
    
    def get_datasets(
        self
    ) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """
        Get train/val/test datasets.
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Determine target column for univariate mode
        target_col = None
        if self.features == 'S':
            # Find target column index
            if self.target in self.df_data.columns:
                target_col = list(self.df_data.columns).index(self.target)
            else:
                target_col = 0  # Default to first column
        
        train_dataset = TimeSeriesDataset(
            self.train_data,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            target_col=target_col,
        )
        
        val_dataset = TimeSeriesDataset(
            self.val_data,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            target_col=target_col,
        )
        
        test_dataset = TimeSeriesDataset(
            self.test_data,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            label_len=self.label_len,
            target_col=target_col,
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data
    
    def get_scaler(self) -> Optional[StandardScaler]:
        """Get the scaler for inverse transformation."""
        return self.scaler


def get_dataloader(
    dataset_name: str,
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    split: str = 'train',
    num_workers: int = 4,
    label_len: int = 48,
    features: str = 'M',
    scale: bool = True,
    shuffle: Optional[bool] = None,
    use_paper_splits: bool = False,
    **kwargs
) -> TorchDataLoader:
    """
    工廠函數：根據數據集名稱返回對應的 DataLoader
    
    Args:
        dataset_name: 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'weather'
        seq_len: Input sequence length (96, 192, 336, 512)
        pred_len: Prediction length (96, 192, 336, 720)
        batch_size: Batch size
        split: Data split ('train', 'val', 'test')
        num_workers: Number of data loading workers
        label_len: Label length for decoder
        features: Feature mode ('M', 'S', 'MS')
        scale: Whether to standardize
        shuffle: Whether to shuffle (auto: True for train, False for val/test)
        **kwargs: Additional arguments for TimeSeriesDataLoader
        
    Returns:
        DataLoader for the specified split
    """
    # Get dataset configuration
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )
    
    config = DATASET_CONFIGS[dataset_name]
    data_path = config['path']
    target = config.get('target', 'OT')
    split_ratio = config.get('split_ratio', (0.7, 0.1, 0.2))
    
    # Initialize data loader
    data_loader = TimeSeriesDataLoader(
        data_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        features=features,
        target=target,
        scale=scale,
        train_ratio=split_ratio[0],
        val_ratio=split_ratio[1],
        test_ratio=split_ratio[2],
        use_paper_splits=use_paper_splits,
        dataset_name=dataset_name,
        **kwargs
    )
    
    # Get datasets
    train_dataset, val_dataset, test_dataset = data_loader.get_datasets()
    
    # Select dataset based on split
    if split == 'train':
        dataset = train_dataset
        if shuffle is None:
            shuffle = True
    elif split == 'val':
        dataset = val_dataset
        if shuffle is None:
            shuffle = False
    elif split == 'test':
        dataset = test_dataset
        if shuffle is None:
            shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
    
    # Create DataLoader
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),  # Drop last batch only for training
    )
    
    # Attach data_loader and scaler for inverse transform
    dataloader.data_loader = data_loader
    dataloader.scaler = data_loader.get_scaler()
    
    return dataloader


# Alias for backward compatibility
DataLoader = TimeSeriesDataLoader