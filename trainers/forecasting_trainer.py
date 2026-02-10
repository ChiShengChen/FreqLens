"""
Forecasting trainer for time series prediction.

損失函數：
L_total = L_pred + λ₁*L_sparse + λ₂*L_recon + λ₃*L_ortho

其中：
- L_pred = MSE(ŷ, y)  # 主損失
- L_sparse = L1_norm(selection_mask)  # 稀疏性懲罰
- L_recon = MSE(reconstructed, x)  # 重構損失
- L_ortho = orthogonality_loss(freq_features)  # 頻率正交性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
import os
from pathlib import Path
import time

# TensorBoard 是可選的
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    # 創建一個假的 SummaryWriter 類
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

from .base_trainer import BaseTrainer
from utils.metrics import mse, mae, evaluate_metrics


class ForecastingTrainer(BaseTrainer):
    """
    預測任務訓練器
    
    特點：
    1. 支持多 GPU 訓練
    2. 學習率調度（CosineAnnealing）
    3. Early Stopping
    4. TensorBoard 日誌
    5. 自動保存最佳模型
    6. Gumbel 溫度退火
    
    Args:
        model: FreqLens 模型
        train_loader: 訓練 DataLoader
        val_loader: 驗證 DataLoader
        config: 訓練配置字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: str = "cuda",
        use_multi_gpu: bool = False,
    ):
        """
        Initialize forecasting trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            config: Training configuration dictionary
            device: Device to use
            use_multi_gpu: Whether to use multiple GPUs
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, device, 
            gradient_clip=config.get("gradient_clip", 1.0)
        )
        
        self.config = config
        self.use_multi_gpu = use_multi_gpu
        
        # Multi-GPU support
        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using {torch.cuda.device_count()} GPUs")
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Regularization weights
        self.lambda_sparse = config.get("lambda_sparse", 0.01)
        self.lambda_recon = config.get("lambda_recon", 0.1)
        self.lambda_ortho = config.get("lambda_ortho", 0.01)
        self.lambda_variance = config.get("lambda_variance", 0.1)  # Encourage prediction variance
        
        # Learning rate scheduler
        scheduler_type = config.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get("num_epochs", 100),
                eta_min=config.get("min_lr", 1e-6),
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 30),
                gamma=config.get("gamma", 0.1),
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Verbose flag
        self.verbose = config.get("verbose", True)
        
        # TensorBoard logging (optional, can be disabled)
        use_tensorboard = config.get("use_tensorboard", False)  # 默認禁用
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.log_dir = config.get("log_dir", "experiments/logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None
            # 不顯示警告，因為 TensorBoard 是可選的
        
        # Checkpoint directory
        self.checkpoint_dir = config.get("checkpoint_dir", "experiments/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training history
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # Gumbel temperature annealing
        self.tau_init = config.get("gumbel_tau_init", 1.0)
        self.tau_min = config.get("gumbel_tau_min", 0.1)
        self.current_tau = self.tau_init
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        y_true: torch.Tensor,
        output: Dict[str, torch.Tensor],
        x_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算總損失（主損失 + 正則化）。
        
        For FreqLensV4:
          L_total = L_pred + λ_div*L_diversity + λ_recon*L_recon
                    + λ_sparse*L_sparse + λ_var*L_variance
        
        For legacy FreqLens:
          L_total = L_pred + λ₁*L_sparse + λ₂*L_recon + λ₃*L_ortho
        """
        # 主損失: L_pred = MSE(ŷ, y)
        loss_pred = self.criterion(predictions, y_true)
        
        loss_dict = {"pred": loss_pred.item()}
        total_loss = loss_pred
        
        # --- FreqLensV4: frequency diversity loss ---
        lambda_diversity = self.config.get("lambda_diversity", 0.01)
        if lambda_diversity > 0 and hasattr(self.model, 'diversity_loss'):
            model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            loss_div = lambda_diversity * model_ref.diversity_loss()
            total_loss = total_loss + loss_div
            loss_dict["diversity"] = loss_div.item()
        
        # --- FreqLensV4: reconstruction loss (on hidden features) ---
        if "reconstruction" in output and self.lambda_recon > 0:
            reconstruction = output["reconstruction"]  # (B, L, d_model)
            # Compare with projected input (output of input_proj)
            model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            if hasattr(model_ref, 'input_proj'):
                with torch.no_grad():
                    h_target = model_ref.input_proj(x_enc)
                loss_recon = self.lambda_recon * self.criterion(reconstruction, h_target.detach())
                total_loss = total_loss + loss_recon
                loss_dict["recon"] = loss_recon.item()
        
        # --- Legacy: reconstruction loss (on raw input) ---
        elif "reconstructed" in output and self.lambda_recon > 0:
            reconstructed = output["reconstructed"]
            if reconstructed.shape[1] == x_enc.shape[1]:
                loss_recon = self.lambda_recon * self.criterion(reconstructed, x_enc)
                total_loss = total_loss + loss_recon
                loss_dict["recon"] = loss_recon.item()
        
        # 稀疏性懲罰: L_sparse = λ₁ * L1_norm(selection_mask)
        if "selection_mask" in output and self.lambda_sparse > 0:
            selection_mask = output["selection_mask"]
            loss_sparse = self.lambda_sparse * torch.mean(torch.abs(selection_mask))
            total_loss = total_loss + loss_sparse
            loss_dict["sparse"] = loss_sparse.item()
        
        # 頻率正交性損失: L_ortho = λ₃ * orthogonality_loss(freq_features)
        if "frequency_spectrum" in output and self.lambda_ortho > 0:
            freq_spectrum = output["frequency_spectrum"]  # (B, N_freq, L)
            freq_repr = freq_spectrum.mean(dim=2)  # (B, N_freq)
            freq_norm = F.normalize(freq_repr, p=2, dim=1)  # (B, N_freq)
            similarity = torch.matmul(freq_norm, freq_norm.transpose(0, 1))  # (B, B)
            identity = torch.eye(similarity.shape[0], device=similarity.device)
            loss_ortho = self.lambda_ortho * torch.mean((similarity - identity) ** 2)
            total_loss = total_loss + loss_ortho
            loss_dict["ortho"] = loss_ortho.item()
        
        # 變異數匹配損失: 鼓勵預測的變異數接近真實值的變異數
        if self.lambda_variance > 0:
            pred_var = predictions.var(dim=1).mean()
            true_var = y_true.var(dim=1).mean()
            loss_variance = self.lambda_variance * F.mse_loss(pred_var, true_var)
            total_loss = total_loss + loss_variance
            loss_dict["variance"] = loss_variance.item()
        
        loss_dict["total"] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_step(self, batch: tuple, epoch: int) -> Dict[str, float]:
        """Perform one training step."""
        x_enc, x_mark_enc, y_label, y_true = batch
        
        # Move to device
        x_enc = x_enc.to(self.device)
        y_label = y_label.to(self.device) if y_label is not None else None
        y_true = y_true.to(self.device)
        
        # Compute Gumbel temperature (annealing)
        num_epochs = self.config.get("num_epochs", 100)
        self.current_tau = max(
            self.tau_min,
            self.tau_init - (self.tau_init - self.tau_min) * (epoch / num_epochs)
        )
        
        # Forward pass
        self.optimizer.zero_grad()
        # Check if model accepts gumbel_temperature parameter
        import inspect
        sig = inspect.signature(self.model.forward)
        if 'gumbel_temperature' in sig.parameters:
            output = self.model(x_enc, x_mark_enc, y_label, None, gumbel_temperature=self.current_tau)
        else:
            output = self.model(x_enc, x_mark_enc, y_label, None)
        predictions = output["predictions"]
        
        # Compute total loss
        total_loss, loss_dict = self.compute_loss(predictions, y_true, output, x_enc)
        
        # NaN guard — skip this step if loss is NaN/Inf
        if not torch.isfinite(total_loss):
            self.optimizer.zero_grad()
            loss_dict["total"] = float("nan")
            return loss_dict

        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        return loss_dict
    
    def val_step(self, batch: tuple, scaler=None) -> Dict[str, float]:
        """Perform one validation step."""
        x_enc, x_mark_enc, y_label, y_true = batch
        
        # Move to device
        x_enc = x_enc.to(self.device)
        y_label = y_label.to(self.device) if y_label is not None else None
        y_true = y_true.to(self.device)
        
        # Forward pass
        output = self.model(x_enc, x_mark_enc, y_label, None)
        predictions = output["predictions"]
        
        # Compute loss (on normalized data for training stability)
        total_loss, loss_dict = self.compute_loss(predictions, y_true, output, x_enc)
        
        # Compute metrics (on normalized data for consistency with loss)
        predictions_np = predictions.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        
        # Note: Metrics are computed on normalized data for consistency
        # Denormalization happens only in predict() for final evaluation
        metrics = {
            **loss_dict,
            "mse": mse(y_true_np, predictions_np),
            "mae": mae(y_true_np, predictions_np),
        }
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_metrics = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch, epoch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        self.train_metrics_history.append(avg_metrics)
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model and return metrics."""
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.val_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        self.val_metrics_history.append(avg_metrics)
        
        return avg_metrics
    
    def train(self) -> Dict[str, list]:
        """
        完整訓練流程
        
        Returns:
            Dictionary with training history
        """
        num_epochs = self.config.get("num_epochs", 100)
        log_interval = self.config.get("log_interval", 10)
        save_interval = self.config.get("save_interval", 10)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Regularization: sparse={self.lambda_sparse}, recon={self.lambda_recon}, ortho={self.lambda_ortho}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard (if available)
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train", train_metrics["total"], epoch)
                self.writer.add_scalar("Loss/Val", val_metrics["total"], epoch)
                self.writer.add_scalar("Metrics/Val_MSE", val_metrics["mse"], epoch)
                self.writer.add_scalar("Metrics/Val_MAE", val_metrics["mae"], epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                self.writer.add_scalar("Gumbel_Temperature", self.current_tau, epoch)
                
                # Log component losses
                for key in ["pred", "sparse", "recon", "ortho"]:
                    if key in train_metrics:
                        self.writer.add_scalar(f"Loss_Components/Train_{key}", train_metrics[key], epoch)
                    if key in val_metrics:
                        self.writer.add_scalar(f"Loss_Components/Val_{key}", val_metrics[key], epoch)
            
            # Print progress
            if epoch % log_interval == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['total']:.4f} | "
                    f"Val Loss: {val_metrics['total']:.4f} | "
                    f"Val MSE: {val_metrics['mse']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"τ: {self.current_tau:.3f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Save checkpoint
            if epoch % save_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pt"
                )
                self.save_checkpoint(checkpoint_path, epoch, is_best=False)
            
            # Early stopping
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self.save_checkpoint(best_model_path, epoch, is_best=True)
                if epoch % log_interval == 0:
                    print(f"  ✓ New best model saved (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        if self.writer is not None:
            self.writer.close()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            "train_metrics": self.train_metrics_history,
            "val_metrics": self.val_metrics_history,
        }
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """
        保存檢查點
        
        Args:
            path: Checkpoint path
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "train_metrics": self.train_metrics_history,
            "val_metrics": self.val_metrics_history,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "is_best": is_best,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        加載檢查點
        
        Args:
            path: Checkpoint path
            
        Returns:
            Epoch number
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state dict
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.train_metrics_history = checkpoint.get("train_metrics", [])
        self.val_metrics_history = checkpoint.get("val_metrics", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        return checkpoint.get("epoch", 0)
    
    def predict(self, test_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Make predictions on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with predictions, ground truth, and metrics
        """
        self.model.eval()
        all_predictions = []
        all_ground_truth = []
        all_attributions = []
        
        with torch.no_grad():
            for batch in test_loader:
                x_enc, x_mark_enc, y_label, y_true = batch
                
                x_enc = x_enc.to(self.device)
                y_label = y_label.to(self.device) if y_label is not None else None
                y_true = y_true.to(self.device)
                
                output = self.model(x_enc, x_mark_enc, y_label, None)
                predictions = output["predictions"]
                attributions = output.get("attributions", None)
                
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truth.append(y_true.cpu().numpy())
                if attributions is not None:
                    all_attributions.append(attributions.cpu().numpy())
        
        predictions_norm = np.concatenate(all_predictions, axis=0)
        ground_truth_norm = np.concatenate(all_ground_truth, axis=0)
        
        # Denormalize predictions and ground truth if scaler is available (for reference)
        predictions_denorm = None
        ground_truth_denorm = None
        if hasattr(test_loader, 'scaler') and test_loader.scaler is not None:
            # Reshape for inverse transform: (batch, pred_len, features) -> (batch * pred_len, features)
            batch_size, pred_len, n_features = predictions_norm.shape
            predictions_flat = predictions_norm.reshape(-1, n_features)
            ground_truth_flat = ground_truth_norm.reshape(-1, n_features)
            
            # Inverse transform
            predictions_denorm = test_loader.scaler.inverse_transform(predictions_flat)
            ground_truth_denorm = test_loader.scaler.inverse_transform(ground_truth_flat)
            
            # Reshape back: (batch * pred_len, features) -> (batch, pred_len, features)
            predictions_denorm = predictions_denorm.reshape(batch_size, pred_len, n_features)
            ground_truth_denorm = ground_truth_denorm.reshape(batch_size, pred_len, n_features)
        
        # Use normalized data for predictions/ground_truth in result dict
        predictions = predictions_norm
        ground_truth = ground_truth_norm
        
        # For standard benchmark evaluation, evaluate only the target channel (OT)
        # Standard practice: For multivariate forecasting (M mode), evaluate only target channel
        target_channel_idx = None
        try:
            from data.data_loader import DATASET_CONFIGS
            import pandas as pd
            
            # Try to identify dataset by matching data path
            # Check if test_loader has scaler (which contains data path info)
            data_path = None
            if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'data'):
                # Try to get path from scaler if available
                if hasattr(test_loader, 'scaler') and hasattr(test_loader.scaler, '__dict__'):
                    # Scaler might have path info, but usually we need to match by dataset name
                    pass
            
            # Match dataset by trying all configs and checking which one matches
            # For ETT datasets, we can infer from the fact that OT is always the last column
            for name, config in DATASET_CONFIGS.items():
                try:
                    df = pd.read_csv(config['path'])
                    if 'date' in df.columns:
                        df = df.drop('date', axis=1)
                    
                    # Check if this dataset matches (by checking column structure)
                    # For ETT datasets: 7 columns with OT as last
                    # For other datasets: check target column exists
                    if config['target'] in df.columns:
                        target_channel_idx = df.columns.get_loc(config['target'])
                        # For ETT datasets, OT is always index 6 (last column)
                        # We'll use this as the default
                        if 'ETT' in name:
                            # Verify it's index 6
                            if target_channel_idx == 6:
                                break
                        else:
                            # For other datasets, use the found index
                            break
                except Exception:
                    continue
            
            # If still not found, use default for ETT datasets (OT is always last column, index 6)
            if target_channel_idx is None and n_features == 7:
                # Likely an ETT dataset with 7 features, OT is at index 6
                target_channel_idx = 6
        except Exception as e:
            # If we can't determine target channel, evaluate all channels
            pass
        
        # Evaluate metrics
        if target_channel_idx is not None and target_channel_idx < n_features:
            # Evaluate only target channel (standard benchmark practice)
            predictions_eval = predictions[:, :, target_channel_idx].reshape(-1)
            ground_truth_eval = ground_truth[:, :, target_channel_idx].reshape(-1)
        else:
            # Evaluate all channels (flatten for metrics)
            predictions_eval = predictions.reshape(-1)
            ground_truth_eval = ground_truth.reshape(-1)
        
        # For standard benchmark, compute metrics on NORMALIZED data
        # (Standard practice: all papers report normalized MSE/MAE for fair comparison)
        # The predictions and ground_truth here are already normalized
        metrics = evaluate_metrics(ground_truth_eval, predictions_eval)
        
        # Also compute denormalized metrics for reference (if available)
        if predictions_denorm is not None and ground_truth_denorm is not None:
            if target_channel_idx is not None and target_channel_idx < n_features:
                predictions_denorm_eval = predictions_denorm[:, :, target_channel_idx].reshape(-1)
                ground_truth_denorm_eval = ground_truth_denorm[:, :, target_channel_idx].reshape(-1)
            else:
                predictions_denorm_eval = predictions_denorm.reshape(-1)
                ground_truth_denorm_eval = ground_truth_denorm.reshape(-1)
            
            metrics_denorm = evaluate_metrics(ground_truth_denorm_eval, predictions_denorm_eval)
            # Add denormalized metrics with suffix for reference
            metrics['MSE_denorm'] = metrics_denorm['MSE']
            metrics['MAE_denorm'] = metrics_denorm['MAE']
            metrics['RMSE_denorm'] = metrics_denorm['RMSE']
            metrics['MAPE_denorm'] = metrics_denorm['MAPE']
        
        result = {
            "predictions": predictions_denorm if predictions_denorm is not None else predictions,
            "ground_truth": ground_truth_denorm if ground_truth_denorm is not None else ground_truth,
            "predictions_norm": predictions,
            "ground_truth_norm": ground_truth,
            "metrics": metrics,
        }
        
        if all_attributions:
            result["attributions"] = np.concatenate(all_attributions, axis=0)
        
        return result
