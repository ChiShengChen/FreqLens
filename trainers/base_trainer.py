"""
Base trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base trainer class for all training tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        gradient_clip: float = 1.0,
    ):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to use
            gradient_clip: Gradient clipping value
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        
        self.train_losses = []
        self.val_losses = []
    
    @abstractmethod
    def train_step(self, batch: tuple) -> Dict[str, float]:
        """Perform one training step."""
        pass
    
    @abstractmethod
    def val_step(self, batch: tuple) -> Dict[str, float]:
        """Perform one validation step."""
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            metrics = self.train_step(batch)
            total_loss += metrics.get("loss", 0.0)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        
        return {"loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.val_step(batch)
                total_loss += metrics.get("loss", 0.0)
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        
        return {"loss": avg_loss}
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            **kwargs,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        return checkpoint.get("epoch", 0)
