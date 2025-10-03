"""
Loss Functions for Dual Cross-Attention Learning

This module implements all loss functions used in the paper:
1. Uncertainty-weighted multi-task loss for balancing SA, GLCA, PWCA
2. Cross-entropy loss for FGVC classification
3. Triplet loss for Re-ID tasks  
4. Combined loss strategies

Key equation from paper:
L_total = 1/2 * (1/e^w1 * L_SA + 1/e^w2 * L_GLCA + 1/e^w3 * L_PWCA + w1 + w2 + w3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-based loss weighting for multi-task learning.
    
    Implements the automatic loss balancing from Kendall et al. (2018) as used
    in the paper to balance SA, GLCA, and PWCA losses without manual tuning.
    
    Loss formulation from the paper:
    L_total = 1/2 * (1/exp(w1) * L_SA + 1/exp(w2) * L_GLCA + 1/exp(w3) * L_PWCA + w1 + w2 + w3)
    
    CRITICAL: Although SA and PWCA share NETWORK weights (Q/K/V projections), they have
    SEPARATE LOSS WEIGHTS (w1, w2, w3) in the uncertainty weighting formula. This allows
    the model to automatically balance the contribution of each attention mechanism during training.
    
    Args:
        num_tasks: Number of tasks to balance (3 for SA, GLCA, PWCA)
        init_values: Initial values for uncertainty parameters
        
    Returns:
        total_loss: Weighted combination of all task losses
        log_vars: Current uncertainty parameter values for monitoring
        individual_weights: Current task weights for analysis
    """
    
    def __init__(self, num_tasks: int = 3, init_values: Optional[torch.Tensor] = None):
        super().__init__()
        
        if init_values is None:
            # Initialize to 0, which gives equal weight (1/e^0 = 1) to all tasks initially
            init_values = torch.zeros(num_tasks)
        
        self.log_vars = nn.Parameter(init_values)
        self.num_tasks = num_tasks

        # Allow a broader range; Kendall et al. (2018) does not clamp log vars and
        # the paper relies on the model to rebalance tasks automatically. We keep
        # gentle bounds only to avoid numerical explosions.
        self.min_log_var = -5.0
        self.max_log_var = 5.0
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        Compute uncertainty-weighted total loss.
        
        Per the paper's formula, each attention mechanism (SA, GLCA, PWCA) has its own
        uncertainty weight (w1, w2, w3) that is learned during training to automatically
        balance their contributions.
        
        Args:
            losses: Dictionary with task losses
                   Expected keys: "sa_loss", "glca_loss", "pwca_loss" (optional)
                   
        Returns:
            total_loss: Weighted combination of losses
            log_vars: Current log variance values (w1, w2, w3)
            weights: Current task weights (1/exp(w1), 1/exp(w2), 1/exp(w3))
        """
        total_loss = 0
        
        # Clip log_vars to prevent unbounded growth and numerical instability
        with torch.no_grad():
            self.log_vars.clamp_(self.min_log_var, self.max_log_var)
        
        # SA loss with w1
        if "sa_loss" in losses:
            precision_sa = torch.exp(-self.log_vars[0])
            total_loss += precision_sa * losses["sa_loss"] + self.log_vars[0]
        
        # GLCA loss with w2
        if "glca_loss" in losses and self.num_tasks >= 2:
            precision_glca = torch.exp(-self.log_vars[1])
            total_loss += precision_glca * losses["glca_loss"] + self.log_vars[1]
        
        # PWCA loss with w3
        if "pwca_loss" in losses and self.num_tasks >= 3:
            precision_pwca = torch.exp(-self.log_vars[2])
            total_loss += precision_pwca * losses["pwca_loss"] + self.log_vars[2]
        
        # Apply the 1/2 factor from the paper's formula
        total_loss = 0.5 * total_loss
        
        # Get current values for monitoring
        log_vars_dict = {
            "w1": self.log_vars[0].item(),
            "w2": self.log_vars[1].item() if self.num_tasks >= 2 else 0.0,
            "w3": self.log_vars[2].item() if self.num_tasks >= 3 else 0.0
        }
        
        weights_dict = {
            "weight_sa": torch.exp(-self.log_vars[0]).item(),
            "weight_glca": torch.exp(-self.log_vars[1]).item() if self.num_tasks >= 2 else 1.0,
            "weight_pwca": torch.exp(-self.log_vars[2]).item() if self.num_tasks >= 3 else 1.0
        }
        
        return total_loss, log_vars_dict, weights_dict
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current task weights for monitoring"""
        return {f"weight_{i}": torch.exp(-self.log_vars[i]).item() for i in range(self.num_tasks)}


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for classification tasks.
    
    Standard cross-entropy loss used for FGVC tasks and Re-ID classification.
    Supports label smoothing for better regularization.
    
    Args:
        num_classes: Number of classes
        label_smoothing: Label smoothing factor (default: 0.0)
        ignore_index: Index to ignore in loss computation
        
    Returns:
        loss: Cross-entropy loss value
    """
    
    def __init__(self, num_classes: int, label_smoothing: float = 0.0, 
                 ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            # Apply label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.label_smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = -torch.sum(smooth_targets * log_probs) / logits.size(0)
        else:
            loss = self.criterion(logits, targets)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for Re-ID tasks.
    
    Implements hard triplet mining strategy commonly used in Re-ID.
    For each anchor, selects the hardest positive (same identity, different camera)
    and hardest negative (different identity) within the batch.
    
    Loss formulation:
    L_triplet = max(0, d(a,p) - d(a,n) + margin)
    
    where a=anchor, p=positive, n=negative
    
    Args:
        margin: Margin value for triplet loss (default: 0.3)
        distance_metric: Distance metric ("euclidean" or "cosine")
        hard_mining: Whether to use hard negative mining
        
    Returns:
        loss: Triplet loss value
        num_positive_triplets: Number of active triplets for monitoring
    """
    
    def __init__(self, margin: float = 0.3, distance_metric: str = "euclidean",
                 hard_mining: bool = True):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.hard_mining = hard_mining
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Compute triplet loss with hard mining.
        
        Args:
            features: Feature embeddings [batch_size, feature_dim]
            labels: Identity labels [batch_size]
            
        Returns:
            loss: Triplet loss value
            num_active: Number of active (non-zero) triplets
        """
        # Normalize features if using cosine distance
        if self.distance_metric == "cosine":
            features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distance matrix
        distance_matrix = self._pairwise_distance(features)
        
        if self.hard_mining:
            return self._hard_mining_loss(distance_matrix, labels)
        else:
            return self._batch_all_loss(distance_matrix, labels)
    
    def _pairwise_distance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between all features"""
        if self.distance_metric == "euclidean":
            # Euclidean distance
            n = features.size(0)
            dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(features, features.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()
        elif self.distance_metric == "cosine":
            # Cosine distance (features should be normalized)
            dist = 1 - torch.mm(features, features.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return dist
    
    def _hard_mining_loss(self, distance_matrix: torch.Tensor, 
                         labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Perform hard triplet mining.
        
        Args:
            distance_matrix: Pairwise distance matrix [batch, batch]
            labels: Identity labels [batch]
            
        Returns:
            loss: Hard mining triplet loss
            num_active: Number of active triplets
        """
        batch_size = distance_matrix.size(0)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # For each anchor, find hardest positive and hardest negative
        losses = []
        num_active = 0
        
        for i in range(batch_size):
            # Positive pairs (same identity, excluding self)
            positive_mask = labels_equal[i].clone()
            positive_mask[i] = False  # Exclude self
            
            if positive_mask.any():
                # Hardest positive (maximum distance among positives)
                hardest_positive_dist = distance_matrix[i][positive_mask].max()
                
                # Hardest negative (minimum distance among negatives)
                negative_mask = labels_not_equal[i]
                if negative_mask.any():
                    hardest_negative_dist = distance_matrix[i][negative_mask].min()
                    
                    # Compute triplet loss
                    loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
                    losses.append(loss)
                    
                    if loss > 0:
                        num_active += 1
        
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=distance_matrix.device, requires_grad=True)
        
        return total_loss, num_active
    
    def _batch_all_loss(self, distance_matrix: torch.Tensor,
                       labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Compute batch-all triplet loss (all valid triplets)"""
        batch_size = distance_matrix.size(0)
        
        # Create masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal (self-pairs)
        mask = torch.eye(batch_size, dtype=torch.bool, device=distance_matrix.device)
        labels_equal = labels_equal & ~mask
        
        # Compute all valid triplet losses
        losses = []
        num_active = 0
        
        for i in range(batch_size):
            for j in range(batch_size):
                if labels_equal[i, j]:  # Positive pair
                    for k in range(batch_size):
                        if labels_not_equal[i, k]:  # Negative pair
                            loss = F.relu(distance_matrix[i, j] - distance_matrix[i, k] + self.margin)
                            losses.append(loss)
                            if loss > 0:
                                num_active += 1
        
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=distance_matrix.device, requires_grad=True)
        
        return total_loss, num_active


class DualCrossAttentionLoss(nn.Module):
    """
    Combined loss function for dual cross-attention training.
    
    Integrates all loss components used in the paper:
    - SA branch: Cross-entropy (+ triplet for Re-ID)
    - GLCA branch: Cross-entropy (+ triplet for Re-ID)  
    - PWCA branch: Cross-entropy (+ triplet for Re-ID)
    - Uncertainty weighting for automatic balancing
    
    Args:
        task_type: "fgvc" or "reid" for task-specific loss configuration
        num_classes: Number of classes/identities
        triplet_margin: Margin for triplet loss (Re-ID only)
        triplet_weight: Weight for triplet loss (Re-ID only)
        label_smoothing: Label smoothing factor
        
    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary with individual loss components
        metrics_dict: Dictionary with training metrics
    """
    
    def __init__(self, task_type: str = "fgvc", num_classes: int = 1000,
                 triplet_margin: float = 0.3, triplet_weight: float = 1.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        self.triplet_weight = triplet_weight
        
        # Cross-entropy loss for all branches
        self.ce_loss = CrossEntropyLoss(num_classes, label_smoothing)
        
        # Triplet loss for Re-ID tasks
        if task_type == "reid":
            self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        # Uncertainty-based loss weighting
        # Use 3 learnable weights: w1 for SA, w2 for GLCA, w3 for PWCA
        # Per paper equation: L_total = 1/2 * (1/e^w1 * L_SA + 1/e^w2 * L_GLCA + 1/e^w3 * L_PWCA + w1 + w2 + w3)
        self.uncertainty_weighting = UncertaintyWeightedLoss(num_tasks=3)
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        Compute combined loss for all attention branches.
        
        Args:
            outputs: Model outputs dictionary
                    Keys: "sa_logits", "glca_logits", "pwca_logits", "sa_features", "glca_features"
            targets: Target dictionary  
                    Keys: "labels", "identities" (Re-ID only)
                    
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
            metrics_dict: Training metrics (accuracy, etc.)
        """
        device = next(iter(outputs.values())).device
        labels = targets["labels"]
        
        # Initialize loss dictionary
        individual_losses = {}
        metrics_dict = {}
        
        # Compute classification losses for each branch
        if "sa_logits" in outputs:
            individual_losses["sa_loss"] = self.ce_loss(outputs["sa_logits"], labels)
            metrics_dict["sa_acc"] = self._compute_accuracy(outputs["sa_logits"], labels)
        
        if "glca_logits" in outputs:
            individual_losses["glca_loss"] = self.ce_loss(outputs["glca_logits"], labels)
            metrics_dict["glca_acc"] = self._compute_accuracy(outputs["glca_logits"], labels)
        
        if "pwca_logits" in outputs:
            individual_losses["pwca_loss"] = self.ce_loss(outputs["pwca_logits"], labels)
            metrics_dict["pwca_acc"] = self._compute_accuracy(outputs["pwca_logits"], labels)
        
        # Add triplet losses for Re-ID tasks
        if self.task_type == "reid":
            if "sa_features" in outputs:
                triplet_loss, num_active = self.triplet_loss(outputs["sa_features"], labels)
                individual_losses["sa_loss"] += self.triplet_weight * triplet_loss
                metrics_dict["sa_triplet_active"] = num_active
            
            if "glca_features" in outputs:
                triplet_loss, num_active = self.triplet_loss(outputs["glca_features"], labels)
                individual_losses["glca_loss"] += self.triplet_weight * triplet_loss
                metrics_dict["glca_triplet_active"] = num_active
        
        # Apply uncertainty-based weighting
        total_loss, log_vars, weights = self.uncertainty_weighting(individual_losses)
        
        # Convert tensor losses to floats for logging (prevent computation graph issues)
        loss_dict_floats = {k: v.item() if isinstance(v, torch.Tensor) else v 
                           for k, v in individual_losses.items()}
        
        # Combine loss and metrics dictionaries
        loss_dict = {**loss_dict_floats, **log_vars, **weights}
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict, metrics_dict
    
    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute top-1 accuracy"""
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == targets).float().sum()
            acc = correct / targets.size(0)
            return acc.item()


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                    topk: Tuple[int] = (1,)) -> List[float]:
    """
    Compute top-k accuracy for given predictions and targets.
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size] 
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        accuracies: List of top-k accuracy values
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = predictions.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res
