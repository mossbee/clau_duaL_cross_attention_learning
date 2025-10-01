"""
Training Script for Dual Cross-Attention Learning

Main training script that handles both FGVC and Re-ID tasks with the dual
cross-attention architecture. Supports the complete training pipeline
including multi-task loss weighting, pair-wise attention regularization,
and comprehensive evaluation.

Usage:
    # FGVC training
    python train.py --task fgvc --dataset cub --config configs/cub_config.yaml
    
    # Re-ID training  
    python train.py --task reid --dataset market1501 --config configs/market1501_config.yaml
    
    # Ablation study
    python train.py --task fgvc --dataset cub --ablation sa_only
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import json
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dual_cross_attention.models import DualCrossAttentionViT
from dual_cross_attention.datasets import FGVCDataLoader, ReIDDataLoader
from dual_cross_attention.utils import (
    DualCrossAttentionLoss, FGVCMetrics, ReIDMetrics, 
    AttentionVisualizer, MetricsTracker
)
from dual_cross_attention.configs import get_fgvc_config, get_reid_config


class DualCrossAttentionTrainer:
    """
    Main trainer class for dual cross-attention learning.
    
    Handles the complete training pipeline for both FGVC and Re-ID tasks:
    - Model initialization and configuration
    - Multi-task loss computation with uncertainty weighting
    - Pair-wise cross-attention regularization during training
    - Periodic evaluation and checkpoint saving
    - Attention visualization and logging
    
    Args:
        config: Task-specific configuration object
        device: Training device ("cuda" or "cpu")
        wandb_project: Weights & Biases project name for logging
    """
    
    def __init__(self, config, device: str = "cuda", wandb_project: Optional[str] = None):
        self.config = config
        self.device = device
        self.wandb_project = wandb_project
        
        # Initialize wandb if specified
        if wandb_project and not config.debug:
            import wandb
            wandb.init(
                project=wandb_project,
                config=vars(config),
                name=getattr(config, 'experiment_name', None)
            )
        
        # Setup directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Best metric tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(config.task_type)
    
    def setup_model(self) -> nn.Module:
        """
        Initialize dual cross-attention model.
        
        Creates the model with specified architecture:
        - L=12 SA blocks, M=1 GLCA block, T=12 PWCA blocks  
        - Loads pretrained ViT weights if specified
        - Configures for task-specific requirements
        
        Returns:
            model: Initialized dual cross-attention model
        """
        model = DualCrossAttentionViT(**self.config.get_model_config())
        
        # Load pretrained weights if specified or discover common defaults
        pretrained_path = getattr(self.config, 'pretrained_model', None)
        if not pretrained_path:
            # Try common default locations for ViT-B_16 ImageNet-21k .npz
            candidate_paths = [
                os.path.join(os.getcwd(), 'pretrained', 'ViT-B_16.npz'),
                os.path.join(os.getcwd(), 'ViT-B_16.npz'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained', 'ViT-B_16.npz'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ViT-B_16.npz'),
                # If user kept a copy in the reference folder, allow picking it up (will be removed later)
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ViT-pytorch', 'checkpoint', 'ViT-B_16.npz')
            ]
            for cpath in candidate_paths:
                if os.path.isfile(cpath):
                    pretrained_path = cpath
                    print(f"Auto-detected pretrained ViT weights at: {pretrained_path}")
                    break
            if not pretrained_path:
                msg = "No pretrained ViT checkpoint provided. Pass --pretrained or set config.pretrained_model to use ImageNet-21k weights."
                if getattr(self.config, 'require_pretrained', False):
                    raise FileNotFoundError(msg)
                else:
                    print(msg)
        
        if pretrained_path:
            try:
                model.load_pretrained_vit(pretrained_path)
            except Exception as e:
                print(f"Warning: failed to load pretrained weights from {pretrained_path}: {e}")
        
        # Move to device
        model = model.to(self.device)
        
        # Enable mixed precision if specified
        if hasattr(self.config, 'mixed_precision') and self.config.mixed_precision:
            try:
                # Use new API if available (PyTorch 2.1+)
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
            except ImportError:
                # Fallback to old API
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        return model
    
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup data loaders for training and evaluation.
        
        Creates appropriate data loaders based on task type:
        - FGVC: Standard train/test splits with pair sampling
        - Re-ID: Train/query/gallery splits with identity sampling
        
        Returns:
            train_loader: Training data loader
            val_loader: Validation/query data loader  
            test_loader: Test/gallery data loader
        """
        if self.config.task_type == "fgvc":
            from dual_cross_attention.datasets import FGVCDataLoader
            
            data_loader = FGVCDataLoader(
                dataset_name=self.config.dataset_name,
                root_dirs={self.config.dataset_name: self.config.data_root},
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                use_extra_augmentations=getattr(self.config, 'use_extra_augmentations', False)
            )
            
            train_loader, test_loader, self.num_classes, self.class_names = data_loader.create_data_loaders()
            val_loader = test_loader  # Use test set for validation in FGVC
            
        else:  # Re-ID
            from dual_cross_attention.datasets import ReIDDataLoader
            
            data_loader = ReIDDataLoader(
                dataset_name=self.config.dataset_name,
                root_dirs={self.config.dataset_name: self.config.data_root},
                batch_size=self.config.batch_size,
                images_per_id=self.config.images_per_id,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            train_loader, val_loader, test_loader, self.num_identities, self.num_cameras = data_loader.create_data_loaders()
        
        return train_loader, val_loader, test_loader
    
    def setup_criterion(self) -> nn.Module:
        """
        Setup loss function with uncertainty weighting.
        
        Creates the combined loss function:
        - Cross-entropy for classification
        - Triplet loss for Re-ID tasks
        - Uncertainty-based multi-task weighting
        
        Returns:
            criterion: Combined loss function
        """
        from dual_cross_attention.utils import DualCrossAttentionLoss
        
        criterion = DualCrossAttentionLoss(
            task_type=self.config.task_type,
            num_classes=getattr(self.config, 'num_classes', self.num_classes),
            triplet_margin=getattr(self.config, 'triplet_margin', 0.3),
            triplet_weight=getattr(self.config, 'triplet_weight', 1.0),
            label_smoothing=getattr(self.config, 'label_smoothing', 0.0)
        )
        
        return criterion.to(self.device)
    
    def setup_optimizer(self, model: nn.Module, criterion: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Setup optimizer and learning rate scheduler.
        
        Configures task-specific optimization:
        - FGVC: Adam with cosine scheduling
        - Re-ID: SGD with momentum and cosine scheduling
        
        Includes both model and criterion parameters (w1, w2, w3 from uncertainty weighting).
        
        Args:
            model: Model to optimize
            criterion: Loss function with learnable parameters
            
        Returns:
            optimizer: Configured optimizer
            scheduler: Learning rate scheduler
        """
        # Combine model and criterion parameters for joint optimization
        params = list(model.parameters()) + list(criterion.parameters())
        
        if self.config.task_type == "fgvc":
            optimizer = optim.Adam(
                params,
                lr=self.config.scaled_lr,
                weight_decay=self.config.weight_decay
            )
        else:  # Re-ID
            optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """
        Train one epoch with dual cross-attention and gradient accumulation.
        
        Implements the complete training loop with gradient accumulation:
        1. Forward pass through SA, GLCA, and PWCA branches
        2. Compute individual losses for each branch
        3. Apply uncertainty-based loss weighting
        4. Accumulate gradients over multiple mini-batches
        5. Update model parameters after accumulation steps
        6. Log training metrics
        
        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            metrics: Training metrics for the epoch
        """
        model.train()
        self.metrics_tracker.reset()
        
        # Gradient accumulation settings
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract batch data
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Create pairs for PWCA (shuffle batch for pairing, ensure no self-pairing)
            B = images.size(0)
            perm_indices = torch.randperm(B, device=self.device)
            
            # Ensure no image is paired with itself (as per paper implementation)
            self_paired = (perm_indices == torch.arange(B, device=self.device))
            if self_paired.any():
                # Swap self-paired indices with their neighbors
                swap_mask = self_paired.nonzero().flatten()
                for i in swap_mask:
                    swap_with = (i + 1) % B
                    perm_indices[i], perm_indices[swap_with] = perm_indices[swap_with].clone(), perm_indices[i].clone()
            
            paired_images = images[perm_indices]
            
            # Only zero gradients at the start of accumulation
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                try:
                    # Use new API if available (PyTorch 2.1+)
                    from torch.amp import autocast
                    context = autocast('cuda')
                except ImportError:
                    # Fallback to old API
                    from torch.cuda.amp import autocast
                    context = autocast()
            else:
                # Use nullcontext to enable gradients when AMP is disabled
                from contextlib import nullcontext
                context = nullcontext()
            
            with context:
                outputs = model(images, paired_images)
                
                # Compute loss
                targets = {"labels": labels}
                total_loss, loss_dict, metrics_dict = criterion(outputs, targets)
                
                # Scale loss by accumulation steps to get the average
                total_loss = total_loss / accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # Update metrics (loss_dict already contains correct per-batch values)
            # Note: We scaled total_loss for gradient accumulation, but loss_dict values
            # are kept at their original scale for accurate logging
            batch_size = images.size(0)
            self.metrics_tracker.update(
                {**loss_dict, **metrics_dict}, 
                batch_size
            )
            
            # Optimizer step after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
            
            # Update progress bar
            if batch_idx % self.config.log_frequency == 0:
                current_metrics = self.metrics_tracker.get_current_metrics()
                effective_batch_size = images.size(0) * accumulation_steps
                pbar.set_postfix({
                    'loss': f"{current_metrics['total_loss']:.4f}",
                    'sa_acc': f"{current_metrics.get('sa_acc', 0):.3f}",
                    'glca_acc': f"{current_metrics.get('glca_acc', 0):.3f}",
                    'eff_bs': effective_batch_size
                })
            # Add uncertainty weights to wandb every log_frequency steps
            if batch_idx % self.config.log_frequency == 0 and self.wandb_project:
                import wandb
                current_metrics = self.metrics_tracker.get_current_metrics()
                wandb.log({
                    "step": epoch * len(train_loader) + batch_idx,
                    "uncertainty/w1": current_metrics.get('w1', 0),
                    "uncertainty/w2": current_metrics.get('w2', 0),
                    "uncertainty/w3": current_metrics.get('w3', 0),
                    "uncertainty/weight_sa": current_metrics.get('weight_sa', 0),
                    "uncertainty/weight_glca": current_metrics.get('weight_glca', 0),
                    "uncertainty/weight_pwca": current_metrics.get('weight_pwca', 0),
                    "losses/sa_loss": current_metrics.get('sa_loss', 0),
                    "losses/glca_loss": current_metrics.get('glca_loss', 0),
                    "losses/pwca_loss": current_metrics.get('pwca_loss', 0)
                })
        
        return self.metrics_tracker.get_current_metrics()
    
    def evaluate(self, model: nn.Module, data_loader: DataLoader,
                criterion: nn.Module, split: str = "val") -> Dict[str, float]:
        """
        Evaluate model on validation or test set.
        
        Performs evaluation with appropriate metrics:
        - FGVC: Top-1 and Top-5 accuracy
        - Re-ID: mAP, Rank-1, Rank-5, Rank-10
        
        Args:
            model: Model to evaluate
            data_loader: Evaluation data loader
            criterion: Loss function
            split: Data split being evaluated ("val" or "test")
            
        Returns:
            metrics: Evaluation metrics
        """
        model.eval()
        
        if self.config.task_type == "fgvc":
            return self._evaluate_fgvc(model, data_loader, criterion, split)
        else:
            return self._evaluate_reid(model, data_loader, criterion, split)
    
    def _evaluate_fgvc(self, model: nn.Module, data_loader: DataLoader,
                      criterion: nn.Module, split: str) -> Dict[str, float]:
        """Evaluate FGVC model"""
        from dual_cross_attention.utils import FGVCMetrics
        
        fgvc_metrics = FGVCMetrics(self.num_classes, self.class_names)
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass (no PWCA during evaluation, use inference mode for SA+GLCA combination)
                outputs = model(images, inference_mode=True)
                
                # Compute loss
                targets = {"labels": labels}
                loss, _, _ = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Combine SA and GLCA predictions as per paper for FGVC inference
                if 'sa_logits' in outputs and 'glca_logits' in outputs:
                    # Convert logits to probabilities and combine (as per paper)
                    sa_probs = F.softmax(outputs['sa_logits'], dim=-1)
                    glca_probs = F.softmax(outputs['glca_logits'], dim=-1)
                    combined_probs = sa_probs + glca_probs
                    # Convert back to logits for metrics computation
                    combined_logits = torch.log(combined_probs + 1e-8)
                    fgvc_metrics.update(combined_logits, labels)
                elif 'sa_logits' in outputs:
                    # Fallback to SA only if GLCA is not available
                    fgvc_metrics.update(outputs['sa_logits'], labels)
        
        # Compute final metrics
        final_metrics = fgvc_metrics.compute()
        final_metrics['loss'] = total_loss / num_batches
        
        return final_metrics
    
    def _evaluate_reid(self, model: nn.Module, data_loader: DataLoader,
                      criterion: nn.Module, split: str) -> Dict[str, float]:
        """Evaluate Re-ID model with concatenated SA and GLCA features"""
        model.eval()
        all_features = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting Re-ID features ({split})"):
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass (no PWCA during evaluation, use inference mode for SA+GLCA combination)
                outputs = model(images, inference_mode=True)
                
                # Concatenate SA and GLCA features as per paper for Re-ID
                if 'sa_features' in outputs and 'glca_features' in outputs:
                    # Concatenate final class tokens of SA and GLCA for Re-ID
                    combined_features = torch.cat([
                        outputs['sa_features'], 
                        outputs['glca_features']
                    ], dim=-1)
                elif 'sa_features' in outputs:
                    # Fallback to SA only if GLCA is not available
                    combined_features = outputs['sa_features']
                else:
                    print("Warning: No features available for Re-ID evaluation")
                    continue
                
                all_features.append(combined_features.cpu())
                all_labels.append(labels.cpu())
                
                # Compute loss if available
                if 'sa_logits' in outputs:
                    targets = {"labels": labels}
                    loss, _, _ = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
        
        # Basic feature statistics (placeholder for full Re-ID evaluation)
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            print(f"Extracted {all_features.shape[0]} features with dimension {all_features.shape[1]}")
            
            # Placeholder metrics - proper Re-ID evaluation would need query/gallery protocol
            metrics = {
                "mAP": 0.0,  # TODO: Implement proper mAP computation
                "rank1": 0.0,  # TODO: Implement proper rank-k accuracy
                "rank5": 0.0,
                "rank10": 0.0,
                "feature_dim": all_features.shape[1],
                "num_features": all_features.shape[0]
            }
            
            if num_batches > 0:
                metrics["loss"] = total_loss / num_batches
            
            return metrics
        else:
            return {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0}
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler._LRScheduler, epoch: int,
                       metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with metric: {metrics.get('top1_acc', metrics.get('mAP', 0)):.4f}")
        
        # Save periodic checkpoints
        if epoch % self.config.save_frequency == 0:
            periodic_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)
    
    def visualize_attention(self, model: nn.Module, data_loader: DataLoader, epoch: int):
        """
        Generate attention visualizations.
        
        Creates attention rollout visualizations for SA and GLCA branches
        to understand what the model is focusing on.
        
        Args:
            model: Trained model
            data_loader: Data loader for visualization samples
            epoch: Current epoch for naming
        """
        # TODO: Implement attention visualization
        # This will be implemented in the attention_visualization task
        pass
    
    def train(self):
        """
        Main training loop.
        
        Executes the complete training procedure:
        1. Setup model, data, loss, optimizer
        2. Training loop with periodic evaluation
        3. Save checkpoints and generate visualizations
        4. Log results to wandb
        """
        print("Setting up training components...")
        
        # Setup model and training components
        model = self.setup_model()
        train_loader, val_loader, test_loader = self.setup_data_loaders()
        criterion = self.setup_criterion()
        optimizer, scheduler = self.setup_optimizer(model, criterion)
        
        # Print training configuration
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        effective_batch_size = self.config.batch_size * accumulation_steps
        
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Physical batch size: {self.config.batch_size}")
        print(f"Gradient accumulation steps: {accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training phase
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Update learning rate
            scheduler.step()
            print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Evaluation phase
            if epoch % self.config.eval_frequency == 0 or epoch == self.config.num_epochs:
                val_metrics = self.evaluate(model, val_loader, criterion, "val")
                
                # Determine if this is the best model
                current_metric = val_metrics.get('top1_acc', val_metrics.get('mAP', 0))
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                
                # Log metrics
                print(f"Epoch {epoch}/{self.config.num_epochs}")
                print(f"Train Loss: {train_metrics['total_loss']:.4f}")
                print(f"Val Metric: {current_metric:.4f} (Best: {self.best_metric:.4f} @ epoch {self.best_epoch})")
                
                # Log to wandb
                if self.wandb_project:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "learning_rate": scheduler.get_last_lr()[0],
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        **{f"val/{k}": v for k, v in val_metrics.items()},
                        # Add uncertainty weights monitoring
                        "uncertainty/w1": train_metrics.get('w1', 0),
                        "uncertainty/w2": train_metrics.get('w2', 0), 
                        "uncertainty/w3": train_metrics.get('w3', 0),
                        "uncertainty/weight_sa": train_metrics.get('weight_sa', 0),
                        "uncertainty/weight_glca": train_metrics.get('weight_glca', 0),
                        "uncertainty/weight_pwca": train_metrics.get('weight_pwca', 0),
                        # Add individual loss components
                        "losses/sa_loss": train_metrics.get('sa_loss', 0),
                        "losses/glca_loss": train_metrics.get('glca_loss', 0),
                        "losses/pwca_loss": train_metrics.get('pwca_loss', 0)
                    })
                
                # Save checkpoint
                self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, is_best)
                
                # Generate visualizations periodically
                if epoch % (self.config.eval_frequency * 2) == 0:
                    self.visualize_attention(model, val_loader, epoch)
        
        print(f"Training completed! Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dual Cross-Attention Training")
    
    # Task and dataset
    parser.add_argument("--task", type=str, choices=["fgvc", "reid"], required=True,
                       help="Task type: fgvc or reid")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (cub/cars/aircraft for fgvc, market1501/duke/msmt17/veri776 for reid)")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, help="Override config batch size")
    parser.add_argument("--epochs", type=int, help="Override config epochs")
    parser.add_argument("--lr", type=float, help="Override config learning rate")
    
    # Model settings  
    parser.add_argument("--pretrained", type=str, help="Path to pretrained ViT checkpoint")
    parser.add_argument("--ablation", type=str, choices=["sa_only", "sa_glca", "sa_pwca", "full"],
                       help="Ablation study configuration")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name for logging")
    parser.add_argument("--no_logging", action="store_true", help="Disable wandb logging")
    
    # Paths
    parser.add_argument("--data_root", type=str, help="Override dataset root path")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    
    # Data augmentation
    parser.add_argument("--use_extra_augmentations", action="store_true", 
                       help="Enable ColorJitter and RandomRotation (not in paper, FGVC only)")
    
    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (small dataset)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    return parser.parse_args()


def setup_logging(log_dir: str, experiment_name: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{experiment_name}.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.task}_{args.dataset}"
        if args.ablation:
            args.experiment_name += f"_{args.ablation}"
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # Load configuration
    if args.task == "fgvc":
        config = get_fgvc_config(args.dataset)
    else:
        config = get_reid_config(args.dataset)
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs  
    if args.lr:
        config.learning_rate = args.lr
    if args.pretrained:
        config.pretrained_model = args.pretrained
    if args.data_root:
        config.data_root = args.data_root
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Set augmentation flag (default False to match paper)
    config.use_extra_augmentations = getattr(args, 'use_extra_augmentations', False)
    
    # Apply ablation settings
    if args.ablation:
        if args.ablation == "sa_only":
            config.num_glca_layers = 0
            config.num_pwca_layers = 0
        elif args.ablation == "sa_glca":
            config.num_pwca_layers = 0
        elif args.ablation == "sa_pwca":
            config.num_glca_layers = 0
        # "full" uses default config
    
    logger.info(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = DualCrossAttentionTrainer(
        config=config,
        device=args.device,
        wandb_project=args.wandb_project if not args.no_logging else None
    )
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

