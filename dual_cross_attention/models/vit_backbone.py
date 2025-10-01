"""
Vision Transformer Backbone Implementation

This module provides the core Vision Transformer architecture that serves as the foundation
for the dual cross-attention learning method. Based on the ViT implementation in the 
reference ViT-pytorch folder but adapted for fine-grained recognition tasks.

Key Components:
- Patch embedding layer
- Position embeddings  
- Multiple transformer encoder blocks
- Classification head
- Support for different input resolutions (FGVC vs Re-ID)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings.
    
    This layer divides the input image into patches and linearly projects them to embeddings.
    Similar to the patch embedding in ViT-pytorch/models/modeling.py but with flexibility
    for different input sizes required by FGVC (448x448) and Re-ID (256x128, 256x256).
    
    Args:
        img_size: Input image size (height, width) tuple
        patch_size: Size of each patch (default: 16)
        in_channels: Number of input channels (default: 3)
        embed_dim: Embedding dimension (default: 768)
    
    Returns:
        patch_embeddings: Tensor of shape (batch_size, num_patches, embed_dim)
        num_patches: Total number of patches
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Convolution layer to extract patches and project to embeddings
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, C, H, W = x.shape
        
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}, {W}) doesn't match expected size {self.img_size}"
        
        # Extract patches and project: [B, C, H, W] -> [B, embed_dim, num_patches_h, num_patches_w]
        x = self.proj(x)  # [B, embed_dim, num_patches_h, num_patches_w]
        
        # Flatten spatial dimensions: [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # Transpose to get [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x, self.num_patches


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Standard transformer self-attention as described in "Attention Is All You Need".
    This serves as the baseline SA block in the dual cross-attention architecture.
    Implementation should follow the attention mechanism in ViT-pytorch/models/modeling.py.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        qkv_bias: Whether to add bias to QKV projections
    
    Returns:
        attention_output: Self-attended features
        attention_weights: Attention weight matrices for rollout computation
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


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) used in transformer blocks.
    
    Two-layer MLP with GELU activation as described in the paper.
    Should match the FFN implementation in ViT-pytorch/models/modeling.py.
    
    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension (typically 4 * embed_dim)
        dropout: Dropout probability
    
    Returns:
        output: Transformed features
    """
    
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer Encoder Block.
    
    Combines Multi-Head Self-Attention and Feed-Forward Network with residual connections
    and layer normalization. This serves as the baseline transformer block before
    integrating cross-attention mechanisms.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: FFN hidden dimension
        dropout: Dropout probability
        qkv_bias: Whether to add bias to QKV projections
    
    Returns:
        output: Transformer block output
        attention_weights: Attention weights for rollout
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, 
                 hidden_dim: int = 3072, dropout: float = 0.1, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with pre-norm and residual connection
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # Feed-forward with pre-norm and residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer backbone for dual cross-attention learning.
    
    Core ViT architecture that will be extended with GLCA and PWCA mechanisms.
    Based on the ViT implementation in ViT-pytorch/models/modeling.py but adapted
    for fine-grained recognition tasks with different input resolutions.
    
    Architecture:
    - Patch embedding layer
    - Position embeddings (learnable)
    - CLS token
    - L=12 transformer blocks (configurable)
    - Classification head
    
    Args:
        img_size: Input image size tuple (height, width)
        patch_size: Patch size for patch embedding
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: FFN hidden dimension
        dropout: Dropout probability
        in_channels: Input channels (default: 3)
    
    Returns:
        logits: Classification logits
        features: Feature representations from all layers (for cross-attention)
        attention_weights: All attention weights (for rollout computation)
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, hidden_dim: int = 3072, dropout: float = 0.1,
                 in_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Store attention weights for rollout
        self.attention_weights = []
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding
        x, num_patches = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Store for attention rollout
        self.attention_weights = []
        layer_features = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, attn_weights = block(x)
            self.attention_weights.append(attn_weights)
            layer_features.append(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Get features and logits from CLS token
        features = x[:, 0]  # [B, embed_dim]
        logits = self.head(features)  # [B, num_classes]
        
        # Stack layer features for cross-attention use
        all_features = torch.stack(layer_features, dim=1)  # [B, num_layers, seq_len, embed_dim]
        
        return logits, features, all_features
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Returns:
            attention_maps: List of attention maps from all layers
        """
        return self.attention_weights

