"""
Cross-Attention Mechanisms for Dual Cross-Attention Learning

This module implements the core innovations of the paper:
1. Global-Local Cross-Attention (GLCA) - Equation 3 in the paper
2. Pair-Wise Cross-Attention (PWCA) - Equation 4 in the paper

Both mechanisms extend standard self-attention to better capture fine-grained features
for visual categorization and object re-identification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

from ..utils.attention_rollout import AttentionRollout


class SelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention.
    
    Implements Equation 1 from the paper:
    f_SA(Q,K,V) = softmax(QK^T/√d)V = SV
    
    This serves as the baseline attention mechanism that will be combined
    with GLCA and PWCA in the dual cross-attention architecture.
    
    Args:
        embed_dim: Embedding dimension (d in the paper)
        num_heads: Number of attention heads for multi-head attention
        dropout: Dropout probability for attention weights
        qkv_bias: Whether to add bias to Q,K,V projections
    
    Returns:
        output: Self-attended features
        attention_weights: Raw attention matrices S for rollout computation
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 dropout: float = 0.1, qkv_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # Generate Q, K, V matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = attn.softmax(dim=-1)
        
        # Store clean attention weights for rollout (before dropout)
        attn_weights_clean = attn_weights
        
        # Apply dropout for training
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        # Return clean weights for attention rollout computation
        return x, attn_weights_clean


class GlobalLocalCrossAttention(nn.Module):
    """
    Global-Local Cross-Attention mechanism.
    
    Implements the GLCA mechanism from Equation 3 in the paper:
    f_GLCA(Q^l,K^g,V^g) = softmax(Q^l(K^g)^T/√d)V^g
    
    Key innovation: Selects top R query vectors with highest attention rollout scores
    to focus on discriminative local regions while maintaining global context through
    the full key-value pairs.
    
    The attention rollout computation follows Equation 2:
    Ŝ_i = S̄_i ⊗ S̄_{i-1} ⊗ ... ⊗ S̄_1
    where S̄_l = 0.5S_l + 0.5E
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads  
        dropout: Dropout probability
        top_k_ratio: Ratio of top queries to select (R in paper, 10% for FGVC, 30% for Re-ID)
        qkv_bias: Whether to add bias to projections
    
    Returns:
        output: Cross-attended features focusing on discriminative regions
        local_attention_weights: Attention weights for selected local queries
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 dropout: float = 0.1, top_k_ratio: float = 0.1, qkv_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k_ratio = top_k_ratio
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Separate Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias) 
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Attention rollout for local region selection
        self.attention_rollout = AttentionRollout()
    
    def compute_attention_rollout(self, attention_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout scores to identify high-response regions.
        
        Based on the attention rollout method described in the paper and
        implemented in vit_rollout.py. This function accumulates attention 
        weights across all previous layers to identify important patches.
        
        Args:
            attention_weights: List of attention matrices from all previous SA layers
            
        Returns:
            rollout_scores: Accumulated attention scores for CLS token to all patches
        """
        return self.attention_rollout.rollout(attention_weights)
    
    def select_local_queries(self, queries: torch.Tensor, 
                           rollout_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k queries based on attention rollout scores.
        
        IMPORTANT: Per the paper, we select "top R query vectors from Q_i that correspond to 
        the top R highest responses". The paper's description focuses on patch selection,
        suggesting we select ONLY the high-response patches, not necessarily including CLS.
        
        However, for classification, the CLS token must be able to attend to these regions.
        Looking at the paper's implementation details and Figure 1, GLCA computes cross-attention
        where local queries attend to global key-values. For the final classification to use
        GLCA features, we need the CLS token to be updated through GLCA.
        
        Args:
            queries: All query vectors Q [B, N, embed_dim]
            rollout_scores: Attention rollout scores for patch selection [B, N-1]
            
        Returns:
            local_queries: Selected query vectors Q^l with CLS + highest response patches
            selected_indices: Indices of selected patches (CLS=0 is always first)
        """
        B, N, C = queries.shape
        num_patches = N - 1  # Exclude CLS token from patch selection
        num_selected = max(1, int(num_patches * self.top_k_ratio))
        
        # Get top-k patch indices based on rollout scores
        _, patch_indices = torch.topk(rollout_scores, num_selected, dim=-1)  # [B, num_selected]
        
        # Add 1 to indices to account for CLS token at position 0
        patch_indices = patch_indices + 1  # [B, num_selected]
        
        # Include CLS token (position 0) as first query
        # This allows GLCA to update CLS token based on local discriminative regions
        cls_index = torch.zeros(B, 1, dtype=torch.long, device=queries.device)
        selected_indices = torch.cat([cls_index, patch_indices], dim=1)
        
        # Select corresponding queries: CLS + top-R patches
        batch_indices = torch.arange(B, device=queries.device).unsqueeze(1).expand(-1, num_selected + 1)
        local_queries = queries[batch_indices, selected_indices]  # [B, 1+num_selected, C]
        
        return local_queries, selected_indices
    
    def forward(self, x: torch.Tensor, attention_history: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # Compute attention rollout from previous SA layers
        rollout_scores = self.compute_attention_rollout(attention_history)
        
        if rollout_scores is None:
            # Fallback: use all patches if no attention history
            rollout_scores = torch.ones(B, N-1, device=x.device)
        
        # CRITICAL: Add numerical stability checks
        # Replace any NaN/Inf values with uniform scores to prevent crashes
        if torch.isnan(rollout_scores).any() or torch.isinf(rollout_scores).any():
            print("Warning: NaN/Inf detected in attention rollout, using uniform scores")
            rollout_scores = torch.ones(B, N-1, device=x.device)
        
        # Generate global K, V from full input
        k_global = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_global = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Select local queries based on rollout scores
        local_queries, selected_indices = self.select_local_queries(x, rollout_scores)
        num_selected = local_queries.size(1)
        
        # Generate local Q from selected patches  
        q_local = self.q_proj(local_queries).reshape(B, num_selected, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Cross-attention: Q^l × K^g → V^g
        attn = (q_local @ k_global.transpose(-2, -1)) * self.scale  # [B, num_heads, num_selected, N]
        attn_weights = attn.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to global values
        out = (attn_weights @ v_global).transpose(1, 2).reshape(B, num_selected, C)  # [B, num_selected, C]
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        # CRITICAL FIX: Return only the RESIDUAL, not full output
        # The residual connection in dual_vit.py does: glca_out = x + glca_out
        # So we need to return the CHANGE (delta), not x + delta
        # For non-selected patches, the change should be zero (identity mapping)
        # For selected patches, return the attention output as the residual
        
        # Create output as zeros (identity residual for non-selected patches)
        # IMPORTANT: Use same dtype as out to avoid mixed precision issues
        output = torch.zeros_like(x, dtype=out.dtype)
        
        # Place local attention results back to their positions
        # CRITICAL: Use proper 3D indexing to assign [B, num_selected, C] to specific positions
        # We need to scatter the results back to their original positions in the sequence
        for b in range(B):
            output[b, selected_indices[b]] = out[b]
        
        return output, attn_weights


class PairWiseCrossAttention(nn.Module):
    """
    Pair-Wise Cross-Attention for regularization during training.
    
    Implements the PWCA mechanism from Equation 4 in the paper:
    f_PWCA(Q_1,K_c,V_c) = softmax(Q_1 K_c^T/√d)V_c
    where K_c = [K_1; K_2] and V_c = [V_1; V_2]
    
    Key innovation: Uses another image as a distractor by concatenating key-value
    pairs from both images, creating "contaminated" attention scores that reduce
    overfitting to sample-specific features.
    
    This mechanism is only used during training (T=12 blocks) and removed during
    inference to avoid extra computational cost.
    
    IMPORTANT: As stated in the paper, "PWCA branch shares weights with the SA branch"
    This means PWCA reuses the same Q/K/V and projection weights from Self-Attention.
    
    Args:
        sa_attention: The SelfAttention module to share weights with
        dropout: Dropout probability for attention weights
    
    Returns:
        output: Cross-attended features with pair-wise regularization
        attention_weights: Combined attention weights (2N total scores, where N includes CLS token)
    """
    
    def __init__(self, sa_attention: SelfAttention, dropout: float = 0.1):
        super().__init__()
        self.sa_attention = sa_attention  # Share weights with SA
        self.embed_dim = sa_attention.embed_dim
        self.num_heads = sa_attention.num_heads
        self.head_dim = sa_attention.head_dim
        self.scale = sa_attention.scale
        
        # Use SA's dropout settings, but allow override for PWCA-specific dropout
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with image pair using shared SA weights.
        
        Args:
            x1: Target image features (the image being trained)
            x2: Distractor image features (randomly sampled pair)
            
        Returns:
            output: Cross-attended features for x1 with x2 as distractor
            attention_weights: Combined attention matrix
        """
        B, N, C = x1.shape
        
        # Use shared SA weights for Q, K, V generation
        # Generate Q from target image x1 using SA's qkv projection
        qkv1 = self.sa_attention.qkv(x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1.unbind(0)  # [B, num_heads, N, head_dim]
        
        # Generate K, V from distractor image x2 using SA's qkv projection
        qkv2 = self.sa_attention.qkv(x2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)  # [B, num_heads, N, head_dim]
        
        # Concatenate key and value matrices: K_c = [K_1; K_2], V_c = [V_1; V_2]
        k_combined = torch.cat([k1, k2], dim=2)  # [B, num_heads, 2N, head_dim]
        v_combined = torch.cat([v1, v2], dim=2)  # [B, num_heads, 2N, head_dim]
        
        # Cross-attention: Q1 attends to combined K,V from both images
        attn = (q1 @ k_combined.transpose(-2, -1)) * self.scale  # [B, num_heads, N, 2N]
        attn_weights = attn.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to combined values
        x = (attn_weights @ v_combined).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Use shared SA projection weights
        x = self.sa_attention.proj(x)
        x = self.sa_attention.proj_dropout(x)
        
        return x, attn_weights
    
    def create_image_pairs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create random image pairs for training.
        
        Randomly samples pairs from the same training batch to create
        distractor relationships as described in the paper.
        
        Args:
            x: Batch of image features [B, N, C]
            
        Returns:
            x1: Target images [B, N, C]
            x2: Corresponding distractor images [B, N, C]
        """
        B = x.size(0)
        
        # Create random permutation for pairing
        perm_indices = torch.randperm(B, device=x.device)
        
        # Ensure no image is paired with itself
        self_paired = (perm_indices == torch.arange(B, device=x.device))
        if self_paired.any():
            # Swap self-paired indices with their neighbors
            swap_mask = self_paired.nonzero().flatten()
            for i in swap_mask:
                swap_with = (i + 1) % B
                perm_indices[i], perm_indices[swap_with] = perm_indices[swap_with], perm_indices[i]
        
        x1 = x  # Target images
        x2 = x[perm_indices]  # Distractor images
        
        return x1, x2



