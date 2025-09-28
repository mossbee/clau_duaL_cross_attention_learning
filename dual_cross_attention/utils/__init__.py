"""
Utility functions for dual cross-attention learning.

This module provides essential utilities for attention rollout computation,
loss functions, evaluation metrics, and visualization capabilities.
"""

from .attention_rollout import AttentionRollout, rollout_attention_maps
from .loss_functions import UncertaintyWeightedLoss, TripletLoss, CrossEntropyLoss, DualCrossAttentionLoss
from .metrics import FGVCMetrics, ReIDMetrics, MetricsTracker, compute_accuracy, compute_mAP
from .visualization import AttentionVisualizer, plot_attention_maps, save_attention_heatmaps

__all__ = [
    'AttentionRollout', 'rollout_attention_maps',
    'UncertaintyWeightedLoss', 'TripletLoss', 'CrossEntropyLoss', 'DualCrossAttentionLoss',
    'FGVCMetrics', 'ReIDMetrics', 'MetricsTracker', 'compute_accuracy', 'compute_mAP',
    'AttentionVisualizer', 'plot_attention_maps', 'save_attention_heatmaps'
]

