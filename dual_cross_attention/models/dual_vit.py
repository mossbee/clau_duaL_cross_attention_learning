"""
Dual Cross-Attention Vision Transformer

This module implements the complete dual cross-attention architecture that combines:
1. L=12 Self-Attention (SA) blocks - baseline transformer
2. M=1 Global-Local Cross-Attention (GLCA) block - local discriminative features  
3. T=12 Pair-Wise Cross-Attention (PWCA) blocks - regularization during training

The architecture follows the multi-task learning approach described in the paper
with uncertainty-based loss weighting to balance the three attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Optional, List
import math

from .vit_backbone import VisionTransformer, TransformerBlock, PatchEmbedding
from ..utils.stochastic_depth import DropPath
from .attention_modules import SelfAttention, GlobalLocalCrossAttention, PairWiseCrossAttention


class TransformerBlockWithPWCA(nn.Module):
    """
    Transformer block that supports both SA and PWCA with shared weights.
    
    This block implements the weight sharing between SA and PWCA branches as 
    described in the paper: "The PWCA branch shares weights with the SA branch"
    
    During training:
    - SA branch: processes x through self-attention
    - PWCA branch: processes (x, x_pair) through pair-wise cross-attention using SAME weights
    
    During inference:
    - Only SA branch is used (PWCA is disabled)
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        hidden_dim: FFN hidden dimension
        dropout: Dropout probability
    
    Returns:
        sa_output: Self-attention output
        pwca_output: Pair-wise cross-attention output (training only)
        sa_weights: Self-attention weights for rollout
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 hidden_dim: int = 3072, dropout: float = 0.1,
                 drop_path_prob: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Self-Attention components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.sa = SelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth for residual branches
        self.drop_path1 = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

        # PWCA uses the SAME SA module (weight sharing)
        self.pwca = PairWiseCrossAttention(self.sa, dropout)
    
    def forward(self, x: torch.Tensor, x_pair: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional PWCA computation.
        
        Args:
            x: Input features [B, N, C]
            x_pair: Paired features for PWCA (training only) [B, N, C]
            
        Returns:
            outputs: Dictionary with 'sa_output', 'sa_weights', and optionally 'pwca_output'
        """
        outputs = {}
        
        # Self-Attention branch
        sa_out, sa_weights = self.sa(self.norm1(x))
        sa_out = x + self.drop_path1(sa_out)
        sa_out = sa_out + self.drop_path2(self.ffn(self.norm2(sa_out)))
        
        outputs['sa_output'] = sa_out
        outputs['sa_weights'] = sa_weights
        
        # Pair-Wise Cross-Attention branch (training only, shares weights with SA)
        if x_pair is not None and self.training:
            pwca_out, pwca_weights = self.pwca(self.norm1(x), self.norm1(x_pair))
            pwca_out = x + self.drop_path1(pwca_out)
            pwca_out = pwca_out + self.drop_path2(self.ffn(self.norm2(pwca_out)))
            
            outputs['pwca_output'] = pwca_out
            outputs['pwca_weights'] = pwca_weights
        
        return outputs


class DualCrossAttentionBlock(nn.Module):
    """
    Global-Local Cross-Attention block operating on SA-refined features.

    The paper routes SA features into the GLCA module without running another
    self-attention stage. This block therefore performs only the global-local
    cross-attention followed by its dedicated feed-forward network.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 hidden_dim: int = 3072, dropout: float = 0.1,
                 top_k_ratio: float = 0.1, drop_path_prob: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.norm_glca = nn.LayerNorm(embed_dim)
        self.glca = GlobalLocalCrossAttention(embed_dim, num_heads, dropout, top_k_ratio)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor,
                attention_history: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        if attention_history is None:
            outputs['glca_output'] = x
            return outputs

        glca_input = self.norm_glca(x)
        glca_out, glca_weights = self.glca(glca_input, attention_history)
        x = x + self.drop_path_attn(glca_out)
        x = x + self.drop_path_ffn(self.ffn(self.norm_ffn(x)))

        outputs['glca_output'] = x
        outputs['glca_weights'] = glca_weights
        return outputs


class UncertaintyLossWeighting(nn.Module):
    """
    Uncertainty-based loss weighting for multi-task learning.
    
    Implements the automatic loss balancing strategy from Kendall et al. (2018)
    as used in the paper:
    
    L_total = 1/2 * (1/e^w1 * L_SA + 1/e^w2 * L_GLCA + 1/e^w3 * L_PWCA + w1 + w2 + w3)
    
    The learnable parameters w1, w2, w3 automatically balance the contribution
    of each attention mechanism during training, avoiding manual hyperparameter tuning.
    
    Args:
        num_tasks: Number of tasks (3 for SA, GLCA, PWCA)
    
    Returns:
        weighted_loss: Combined loss with automatic weighting
        log_vars: Current uncertainty parameters for monitoring
    """
    
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_keys = ["sa_loss", "glca_loss", "pwca_loss"]
        total_loss = 0
        
        for i, key in enumerate(loss_keys):
            if key in losses and i < len(self.log_vars):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * losses[key] + self.log_vars[i]
        
        total_loss = 0.5 * total_loss
        
        log_vars_dict = {f"w{i+1}": self.log_vars[i].item() for i in range(self.num_tasks)}
        
        return total_loss, log_vars_dict


class DualCrossAttentionViT(nn.Module):
    """
    Complete Dual Cross-Attention Vision Transformer model.
    
    This is the main model that integrates all components:
    - Vision Transformer backbone with patch embeddings
    - L=12 Self-Attention blocks for baseline features
    - M=1 Global-Local Cross-Attention block for local discrimination
    - T=12 Pair-Wise Cross-Attention blocks for training regularization
    - Uncertainty-based loss weighting for multi-task learning
    - Dual classification heads for SA and GLCA branches
    
    Architecture supports both FGVC and Re-ID tasks with different configurations:
    - FGVC: 448x448 input, 10% local queries, cross-entropy loss
    - Re-ID: 256x128/256x256 input, 30% local queries, cross-entropy + triplet loss
    
    Args:
        img_size: Input image size (height, width)
        patch_size: Patch size for embedding
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        num_sa_layers: Number of self-attention layers (L=12)
        num_glca_layers: Number of GLCA layers (M=1)  
        num_pwca_layers: Number of PWCA layers (T=12)
        num_heads: Number of attention heads
        hidden_dim: FFN hidden dimension
        dropout: Dropout probability
        top_k_ratio: Local query selection ratio (0.1 for FGVC, 0.3 for Re-ID)
        task_type: Task type ("fgvc" or "reid") for configuration
    
    Returns:
        During training:
            logits_sa: Classification logits from SA branch
            logits_glca: Classification logits from GLCA branch  
            logits_pwca: Classification logits from PWCA branch
            features: Feature representations for visualization
            
        During inference:
            combined_logits: Combined SA + GLCA predictions
            features: Feature representations
            attention_maps: Attention visualizations
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 num_sa_layers: int = 12, num_glca_layers: int = 1, num_pwca_layers: int = 12,
                 num_heads: int = 12, hidden_dim: int = 3072, dropout: float = 0.1,
                 top_k_ratio: float = 0.1, task_type: str = "fgvc", use_gradient_checkpointing: bool = False,
                 use_stochastic_depth: bool = False, stochastic_depth_prob: float = 0.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.task_type = task_type
        self.top_k_ratio = top_k_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Patch embedding (shared by all branches)
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # ============================================================================
        # ARCHITECTURE SUMMARY (from paper Section 3):
        # ============================================================================
        # 1. L=12 Self-Attention (SA) blocks + T=12 Pair-Wise Cross-Attention (PWCA) blocks
        #    → IMPLEMENTED AS: 12 TransformerBlockWithPWCA blocks
        #    → WEIGHT SHARING: PWCA shares weights with SA (same Q/K/V projections)
        #    → During training: Each block processes both SA(x) and PWCA(x, x_pair)
        #    → During inference: Only SA(x) is used, PWCA is disabled
        #
        # 2. M=1 Global-Local Cross-Attention (GLCA) block  
        #    → IMPLEMENTED AS: 1 DualCrossAttentionBlock (uses SA-refined features)
        #    → INDEPENDENT WEIGHTS: GLCA has separate weights from SA/PWCA
        #    → Takes SA-refined embeddings and attention history as input
        # ============================================================================
        
        # Compute per-block stochastic depth rates (linearly scaled)
        total_blocks = num_sa_layers + max(num_glca_layers, 0)
        if use_stochastic_depth and stochastic_depth_prob > 0.0 and total_blocks > 0:
            dpr = [x.item() for x in torch.linspace(0, stochastic_depth_prob, total_blocks)]
        else:
            dpr = [0.0] * total_blocks
        sa_dpr = dpr[:num_sa_layers]
        glca_dpr = dpr[num_sa_layers: num_sa_layers + num_glca_layers]

        # SA/PWCA blocks (L=12, T=12) - PWCA shares weights with SA
        self.sa_pwca_blocks = nn.ModuleList([
            TransformerBlockWithPWCA(embed_dim, num_heads, hidden_dim, dropout, drop_path_prob=sa_dpr[i] if i < len(sa_dpr) else 0.0)
            for i in range(num_sa_layers)
        ])
        
        # GLCA blocks (M=1) - Independent weights from SA
        self.glca_blocks = nn.ModuleList([
            DualCrossAttentionBlock(
                embed_dim, num_heads, hidden_dim, dropout,
                top_k_ratio, drop_path_prob=(glca_dpr[i] if i < len(glca_dpr) else 0.0)
            )
            for i in range(num_glca_layers)
        ])
        
        # Final normalization and classification heads
        self.norm = nn.LayerNorm(embed_dim)
        
        # Separate classification heads for each branch
        self.sa_head = nn.Linear(embed_dim, num_classes)
        self.glca_head = nn.Linear(embed_dim, num_classes) if num_glca_layers > 0 else None
        
        # Uncertainty-based loss weighting
        self.loss_weighting = UncertaintyLossWeighting(3)
        
        # Initialize weights
        self._init_weights()
        
        # Storage for attention maps
        self.attention_maps = {}
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Initialize uncertainty weights to small random values for better learning dynamics
        if hasattr(self, 'loss_weighting') and hasattr(self.loss_weighting, 'log_vars'):
            with torch.no_grad():
                # Initialize with small random values instead of zeros
                self.loss_weighting.log_vars.data = torch.randn_like(self.loss_weighting.log_vars.data) * 0.1
    
    def forward(self, x: torch.Tensor, x_pair: Optional[torch.Tensor] = None, 
               inference_mode: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual cross-attention architecture.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            x_pair: Paired images for PWCA regularization (training only)
            
        Returns:
            output: Dictionary containing logits, features, and attention maps
        """
        B = x.shape[0]
        outputs = {}
        
        # Patch embedding
        x, _ = self.patch_embed(x)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process paired image similarly (for PWCA)
        if x_pair is not None and self.training:
            x_pair, _ = self.patch_embed(x_pair)
            cls_token_pair = self.cls_token.expand(B, -1, -1)
            x_pair = torch.cat([cls_token_pair, x_pair], dim=1)
            x_pair = x_pair + self.pos_embed
            x_pair = self.pos_drop(x_pair)
        
        # SA/PWCA branch - L=12, T=12 blocks with SHARED weights
        # Per paper: "The PWCA branch shares weights with the SA branch"
        # 
        # Paper: "query, key and value vectors are separately computed for both images"
        # Both images evolve through SA layers in parallel
        sa_attention_history = []
        sa_x = x.clone()
        pwca_x = x.clone() if (x_pair is not None and self.training) else None
        
        for block in self.sa_pwca_blocks:
            if x_pair is not None and self.training:
                # Use the block's forward method which handles both SA and PWCA properly
                block_outputs = block(sa_x, x_pair)
                sa_x = block_outputs['sa_output']
                pwca_x = block_outputs['pwca_output']
                # Detach attention weights to prevent memory leak during training
                sa_attention_history.append(block_outputs['sa_weights'].detach())
                
                # Update x_pair for next layer (evolve the paired image through SA)
                sa_out_pair, _ = block.sa(block.norm1(x_pair))
                x_pair = x_pair + block.drop_path1(sa_out_pair)
                x_pair = x_pair + block.drop_path2(block.ffn(block.norm2(x_pair)))
            else:
                # No PWCA training, just SA
                block_outputs = block(sa_x, x_pair=None)
                sa_x = block_outputs['sa_output']
                # Detach attention weights to prevent memory leak
                sa_attention_history.append(block_outputs['sa_weights'].detach() if self.training else block_outputs['sa_weights'])
        
        # SA branch output
        sa_features = self.norm(sa_x)[:, 0]  # CLS token
        sa_logits = self.sa_head(sa_features)
        
        outputs['sa_logits'] = sa_logits
        outputs['sa_features'] = sa_features
        
        # PWCA branch output (training only, shares classification head with SA)
        if pwca_x is not None:
            pwca_features = self.norm(pwca_x)[:, 0]  # CLS token
            pwca_logits = self.sa_head(pwca_features)  # Share head with SA
            
            outputs['pwca_logits'] = pwca_logits
            outputs['pwca_features'] = pwca_features
        
        # GLCA branch - M=1 block (uses SA-refined embeddings, INDEPENDENT weights)
        # Per paper: "GLCA does not share weights with SA"
        if len(self.glca_blocks) > 0:
            # Use SA-refined embeddings as input to GLCA for coordinated learning
            glca_x = sa_x.clone()
            
            for glca_block in self.glca_blocks:
                block_outputs = glca_block(glca_x, attention_history=sa_attention_history)
                if 'glca_output' in block_outputs:
                    glca_x = block_outputs['glca_output']
            
            glca_features = self.norm(glca_x)[:, 0]  # CLS token
            glca_logits = self.glca_head(glca_features)
            
            outputs['glca_logits'] = glca_logits
            outputs['glca_features'] = glca_features
        
        # Store attention maps for visualization (only during evaluation to save memory)
        if not self.training:
            self.attention_maps = {
                'sa_attention': sa_attention_history,
            }
        else:
            # Clear attention maps during training to save memory
            self.attention_maps = {}
        
        # Add inference-time combinations as per paper
        if inference_mode and not self.training:
            if self.task_type == "fgvc" and 'sa_logits' in outputs and 'glca_logits' in outputs:
                # FGVC: Add class probabilities from SA and GLCA
                sa_probs = torch.nn.functional.softmax(outputs['sa_logits'], dim=-1)
                glca_probs = torch.nn.functional.softmax(outputs['glca_logits'], dim=-1)
                combined_probs = sa_probs + glca_probs
                outputs['combined_logits'] = torch.log(combined_probs + 1e-8)
                outputs['combined_probs'] = combined_probs
                
            elif self.task_type == "reid" and 'sa_features' in outputs and 'glca_features' in outputs:
                # Re-ID: Concatenate final class tokens from SA and GLCA
                combined_features = torch.cat([
                    outputs['sa_features'], 
                    outputs['glca_features']
                ], dim=-1)
                outputs['combined_features'] = combined_features
        
        return outputs
    
    def configure_for_task(self, task_type: str):
        """
        Configure model parameters for specific tasks.
        
        Adjusts model settings based on task requirements:
        - FGVC: 10% local queries, cross-entropy loss only
        - Re-ID: 30% local queries, cross-entropy + triplet loss
        
        Args:
            task_type: "fgvc" or "reid"
        """
        self.task_type = task_type
        
        if task_type == "fgvc":
            self.top_k_ratio = 0.1  # 10% for FGVC
        elif task_type == "reid":
            self.top_k_ratio = 0.3  # 30% for Re-ID
        
        # Update GLCA blocks with new top_k_ratio
        for block in self.glca_blocks:
            if hasattr(block, 'glca'):
                block.glca.top_k_ratio = self.top_k_ratio
    
    def get_attention_maps(self) -> Dict[str, List[torch.Tensor]]:
        """
        Extract attention maps for visualization.
        
        Returns attention rollout maps from SA and GLCA branches
        for visualizing which regions the model focuses on.
        
        Returns:
            attention_maps: Dictionary with SA and GLCA attention visualizations
        """
        return self.attention_maps
    
    def freeze_pretrained_weights(self):
        """
        Freeze pretrained backbone weights and only train attention modules.
        
        Useful for transfer learning scenarios where we want to preserve
        pretrained ImageNet features and only adapt the attention mechanisms.
        """
        # Freeze patch embedding and position embeddings
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        
        # Freeze SA/PWCA blocks 
        for block in self.sa_pwca_blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # Keep GLCA and classification heads trainable
        print("Froze pretrained weights. GLCA and classification heads remain trainable.")
    
    def load_pretrained_vit(self, checkpoint_path: str):
        """
        Load pretrained Vision Transformer weights.
        
        Loads weights from standard ViT checkpoints (ImageNet-21k or ImageNet-1k)
        and initializes the SA branch. GLCA and PWCA branches are randomly initialized.
        
        Args:
            checkpoint_path: Path to pretrained ViT checkpoint (.npz or .pth)
        """
        import numpy as np
        from scipy import ndimage
        
        def numpy_to_torch(weights_np, conv: bool = False):
            if conv:
                # HWIO -> OIHW
                weights_np = weights_np.transpose([3, 2, 0, 1])
            return torch.from_numpy(weights_np)
        
        if checkpoint_path.endswith('.npz'):
            # Load from numpy format (original Google checkpoints)
            weights = np.load(checkpoint_path, allow_pickle=True)
            print(f"Loading pretrained weights from {checkpoint_path}")
            with torch.no_grad():
                # Patch embedding proj
                if 'embedding/kernel' in weights and 'embedding/bias' in weights:
                    w = numpy_to_torch(weights['embedding/kernel'], conv=True)
                    b = numpy_to_torch(weights['embedding/bias'])
                    self.patch_embed.proj.weight.copy_(w)
                    self.patch_embed.proj.bias.copy_(b)
                
                # CLS token
                if 'cls' in weights:
                    self.cls_token.copy_(numpy_to_torch(weights['cls']))
                
                # Encoder norm -> our final norm
                if 'Transformer/encoder_norm/scale' in weights and 'Transformer/encoder_norm/bias' in weights:
                    self.norm.weight.copy_(numpy_to_torch(weights['Transformer/encoder_norm/scale']))
                    self.norm.bias.copy_(numpy_to_torch(weights['Transformer/encoder_norm/bias']))
                
                # Position embeddings (resize if needed)
                if 'Transformer/posembed_input/pos_embedding' in weights:
                    posemb = numpy_to_torch(weights['Transformer/posembed_input/pos_embedding'])
                    posemb_new = self.pos_embed
                    if posemb.size() == posemb_new.size():
                        self.pos_embed.copy_(posemb)
                    else:
                        ntok_new = posemb_new.size(1)
                        # split between class and the rest
                        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                        ntok_new -= 1
                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))
                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                        posemb = torch.cat([posemb_tok, numpy_to_torch(posemb_grid)], dim=1)
                        self.pos_embed.copy_(posemb)
                
                # Encoder blocks -> our SA/PWCA blocks
                # ViT-pytorch naming:
                # ROOT = f"Transformer/encoderblock_{i}"
                # query/key/value/out, LayerNorm_0 and LayerNorm_2, MlpBlock_3/Dense_0, Dense_1
                hidden_size = self.embed_dim
                for i, block in enumerate(self.sa_pwca_blocks):
                    root = f"Transformer/encoderblock_{i}"
                    # Collect attention linear weights (separate in npz, combined in our module)
                    try:
                        q_w = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/query/kernel"]).view(hidden_size, hidden_size).t()
                        k_w = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/key/kernel"]).view(hidden_size, hidden_size).t()
                        v_w = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/value/kernel"]).view(hidden_size, hidden_size).t()
                        q_b = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/query/bias"]).view(-1)
                        k_b = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/key/bias"]).view(-1)
                        v_b = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/value/bias"]).view(-1)
                        # Combine into qkv (for SA, which PWCA shares)
                        block.sa.qkv.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
                        block.sa.qkv.bias.copy_(torch.cat([q_b, k_b, v_b], dim=0))
                        # Attention out/proj
                        out_w = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/out/kernel"]).view(hidden_size, hidden_size).t()
                        out_b = numpy_to_torch(weights[f"{root}/MultiHeadDotProductAttention_1/out/bias"]).view(-1)
                        block.sa.proj.weight.copy_(out_w)
                        block.sa.proj.bias.copy_(out_b)
                        
                        # MLP (FFN)
                        fc1_w = numpy_to_torch(weights[f"{root}/MlpBlock_3/Dense_0/kernel"]).t()
                        fc1_b = numpy_to_torch(weights[f"{root}/MlpBlock_3/Dense_0/bias"]).t()
                        fc2_w = numpy_to_torch(weights[f"{root}/MlpBlock_3/Dense_1/kernel"]).t()
                        fc2_b = numpy_to_torch(weights[f"{root}/MlpBlock_3/Dense_1/bias"]).t()
                        block.ffn[0].weight.copy_(fc1_w)
                        block.ffn[0].bias.copy_(fc1_b)
                        block.ffn[3].weight.copy_(fc2_w)
                        block.ffn[3].bias.copy_(fc2_b)
                        
                        # LayerNorms
                        attn_norm_w = numpy_to_torch(weights[f"{root}/LayerNorm_0/scale"])  # attention_norm
                        attn_norm_b = numpy_to_torch(weights[f"{root}/LayerNorm_0/bias"]) 
                        ffn_norm_w = numpy_to_torch(weights[f"{root}/LayerNorm_2/scale"])   # ffn_norm
                        ffn_norm_b = numpy_to_torch(weights[f"{root}/LayerNorm_2/bias"]) 
                        block.norm1.weight.copy_(attn_norm_w)
                        block.norm1.bias.copy_(attn_norm_b)
                        block.norm2.weight.copy_(ffn_norm_w)
                        block.norm2.bias.copy_(ffn_norm_b)
                    except KeyError:
                        # Stop if fewer blocks in checkpoint
                        break
        elif checkpoint_path.endswith('.pth'):
            # Load from PyTorch format
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Load compatible weights
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} pretrained parameters from {checkpoint_path}")
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

