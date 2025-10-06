"""
Attention Rollout Implementation for Dual Cross-Attention Learning

This module implements attention rollout computation as described in the paper
and based on the reference implementation in vit_rollout.py. Attention rollout
tracks how information flows from input tokens through transformer layers by
recursively computing attention across all previous layers.

Key equation from paper:
Ŝ_i = S̄_i ⊗ S̄_{i-1} ⊗ ... ⊗ S̄_1
where S̄_l = 0.5S_l + 0.5E (accounts for residual connections)

This is used by GLCA to identify high-response local regions for cross-attention.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from PIL import Image


class AttentionRollout:
    """
    Attention rollout computation for identifying important image patches.
    
    Based on the attention rollout method and vit_rollout.py implementation.
    Computes accumulated attention scores across all transformer layers to identify
    which input patches are most important for the model's predictions.
    
    This class is used by GlobalLocalCrossAttention to select high-response
    local regions (top R% patches) for discriminative feature learning.
    
    Args:
        discard_ratio: Ratio of lowest attention paths to discard (default: 0.9)
        head_fusion: Method to fuse multi-head attention ("mean", "max", "min")
        
    Reference: vit_rollout.py rollout() function
    """
    
    def __init__(self, discard_ratio: float = 0.9, head_fusion: str = "mean"):
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
    
    def rollout(self, attentions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention rollout across all transformer layers.
        
        Implements the core rollout algorithm from vit_rollout.py:
        1. Fuse attention heads using specified method
        2. Add identity matrix (0.5 * attention + 0.5 * identity) for residual connections
        3. Normalize attention weights
        4. Multiply attention matrices across all layers
        5. Extract CLS token attention to all patches
        
        Args:
            attentions: List of attention matrices from all SA layers
                       Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
                       
        Returns:
            rollout_scores: CLS token attention scores for all patches
                          Shape: [batch_size, num_patches]
        """
        if not attentions:
            return None
            
        device = attentions[0].device
        B = attentions[0].size(0)
        seq_len = attentions[0].size(-1)
        
        # Initialize result matrix as identity
        result = torch.eye(seq_len, device=device).unsqueeze(0).expand(B, -1, -1)
        
        with torch.no_grad():
            for attention in attentions:
                # Fuse attention heads using specified method
                attention_heads_fused = self.fuse_attention_heads(attention)  # [B, seq_len, seq_len]
                
                # Add identity matrix for residual connections (S̄ = 0.5*S + 0.5*I)
                # This accounts for residual connections in transformer: out = attn(x) + x
                I = torch.eye(seq_len, device=device).unsqueeze(0).expand(B, -1, -1)
                attention_heads_fused = (attention_heads_fused + I) / 2
                
                # Re-normalize after adding identity
                attention_heads_fused = attention_heads_fused / (attention_heads_fused.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Recursively multiply: Ŝ_i = S̄_i ⊗ S̄_{i-1} ⊗ ... ⊗ S̄_1
                result = torch.matmul(attention_heads_fused, result)
        
        # Extract CLS token attention to all patches (first row, excluding CLS token itself)
        cls_attention = result[:, 0, 1:]  # [B, num_patches]
        
        return cls_attention
    
    def fuse_attention_heads(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-head attention using specified method.
        
        Args:
            attention: Multi-head attention tensor [batch, heads, seq, seq]
            
        Returns:
            fused_attention: Single attention matrix [batch, seq, seq]
        """
        if self.head_fusion == "mean":
            return attention.mean(dim=1)
        elif self.head_fusion == "max":
            return attention.max(dim=1)[0]
        elif self.head_fusion == "min":
            return attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion method: {self.head_fusion}")
    
    def get_cls_attention_map(self, rollout_result: torch.Tensor, 
                             img_size: Tuple[int, int], patch_size: int = 16) -> np.ndarray:
        """
        Convert CLS attention scores to spatial attention map.
        
        Reshapes the 1D attention scores back to 2D spatial map matching
        the original image patch layout for visualization.
        
        Args:
            rollout_result: CLS attention scores [batch_size, num_patches]
            img_size: Original image size (height, width)  
            patch_size: Size of each patch (default: 16)
            
        Returns:
            attention_map: 2D spatial attention map [height/patch_size, width/patch_size]
        """
        if rollout_result is None:
            return None
            
        # Get the first image in batch
        cls_attention = rollout_result[0].cpu().numpy()
        
        # Calculate grid size
        grid_h = img_size[0] // patch_size
        grid_w = img_size[1] // patch_size
        
        # Reshape to 2D spatial map
        attention_map = cls_attention.reshape(grid_h, grid_w)
        
        # Normalize to [0, 1]
        attention_map = attention_map / np.max(attention_map)
        
        return attention_map


def rollout_attention_maps(model: nn.Module, input_tensor: torch.Tensor,
                          discard_ratio: float = 0.9, head_fusion: str = "mean") -> Dict[str, torch.Tensor]:
    """
    Compute attention rollout maps for a given model and input.
    
    Extracts attention weights from all transformer layers in the model
    and computes rollout maps for both SA and GLCA branches if available.
    
    Args:
        model: Dual cross-attention model
        input_tensor: Input image tensor [batch_size, channels, height, width]
        discard_ratio: Ratio of attention paths to discard
        head_fusion: Multi-head fusion method
        
    Returns:
        rollout_maps: Dictionary containing rollout maps for different branches
                     Keys: "sa_rollout", "glca_rollout"
    """
    
    # Hook function to collect attention weights
    attention_weights = {"sa": [], "glca": []}
    hooks = []
    
    def get_attention_hook(name):
        def hook(module, input, output):
            # Extract attention weights from module output
            if hasattr(output, 'attention_weights'):
                attention_weights[name].append(output.attention_weights)
            elif len(output) > 1 and isinstance(output[1], torch.Tensor):
                # Assume second output is attention weights
                attention_weights[name].append(output[1])
        return hook
    
    # Register hooks for SA and GLCA layers
    for name, module in model.named_modules():
        if 'self_attention' in name or 'sa_block' in name:
            hook = module.register_forward_hook(get_attention_hook("sa"))
            hooks.append(hook)
        elif 'global_local_cross_attention' in name or 'glca_block' in name:
            hook = module.register_forward_hook(get_attention_hook("glca"))
            hooks.append(hook)
    
    # Forward pass to collect attention weights
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute rollout for each branch
    rollout_computer = AttentionRollout(discard_ratio, head_fusion)
    rollout_maps = {}
    
    if attention_weights["sa"]:
        rollout_maps["sa_rollout"] = rollout_computer.rollout(attention_weights["sa"])
    
    if attention_weights["glca"]:
        rollout_maps["glca_rollout"] = rollout_computer.rollout(attention_weights["glca"])
    
    return rollout_maps


def visualize_attention_rollout(image: np.ndarray, attention_map: np.ndarray,
                               save_path: Optional[str] = None, alpha: float = 0.6) -> np.ndarray:
    """
    Visualize attention rollout as overlay on original image.
    
    Based on show_mask_on_image() function from vit_rollout.py.
    Creates heatmap visualization of attention rollout overlaid on input image.
    
    Args:
        image: Original input image [height, width, 3]
        attention_map: Attention rollout map [height, width]
        save_path: Optional path to save visualization
        alpha: Blending factor for overlay (0.0 = only heatmap, 1.0 = only image)
        
    Returns:
        visualization: Combined image with attention overlay [height, width, 3]
    """
    # Ensure image is float32 in range [0, 1]
    img = np.float32(image) / 255
    
    # Resize attention map to match image size
    attention_resized = cv2.resize(attention_map, (img.shape[1], img.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Overlay heatmap on image
    cam = alpha * img + (1 - alpha) * heatmap
    cam = cam / np.max(cam)
    
    # Convert back to uint8
    visualization = np.uint8(255 * cam)
    
    if save_path is not None:
        cv2.imwrite(save_path, visualization)
    
    return visualization


def extract_high_response_regions(rollout_scores: torch.Tensor, top_k_ratio: float = 0.1) -> torch.Tensor:
    """
    Extract indices of high-response regions from attention rollout.
    
    Used by GLCA to select top R% patches with highest attention scores
    for local discriminative feature learning. R=10% for FGVC, R=30% for Re-ID.
    
    Args:
        rollout_scores: CLS attention scores [batch_size, num_patches]
        top_k_ratio: Ratio of top patches to select (R in the paper)
        
    Returns:
        high_response_indices: Indices of selected patches [batch_size, num_selected]
    """
    if rollout_scores is None:
        return None
        
    B, num_patches = rollout_scores.shape
    top_k = int(num_patches * top_k_ratio)
    
    # Get indices of top-k patches with highest attention scores
    _, high_response_indices = torch.topk(rollout_scores, top_k, dim=-1, largest=True)
    
    return high_response_indices


def compute_attention_statistics(attention_weights: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics of attention patterns across layers.
    
    Provides insights into attention distribution and can help with
    model analysis and debugging.
    
    Args:
        attention_weights: List of attention matrices from all layers
        
    Returns:
        stats: Dictionary with attention statistics (entropy, max values, etc.)
    """
    if not attention_weights:
        return {}
    
    stats = {}
    all_attentions = torch.cat([attn.flatten() for attn in attention_weights])
    
    # Basic statistics
    stats['mean'] = all_attentions.mean().item()
    stats['std'] = all_attentions.std().item()
    stats['min'] = all_attentions.min().item()
    stats['max'] = all_attentions.max().item()
    
    # Entropy statistics
    entropies = []
    for attn in attention_weights:
        # Compute entropy for each attention head and layer
        attn_flat = attn.view(-1, attn.size(-1))
        entropy = -(attn_flat * torch.log(attn_flat + 1e-8)).sum(dim=-1)
        entropies.append(entropy.mean().item())
    
    stats['mean_entropy'] = np.mean(entropies)
    stats['std_entropy'] = np.std(entropies)
    
    return stats


class AttentionHook:
    """
    Hook class for collecting attention weights during forward pass.
    
    Utility class to register hooks on attention modules and collect
    attention weights for rollout computation and visualization.
    """
    
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """
        Register hooks on specified attention layers.
        
        Args:
            model: Model to register hooks on
            layer_names: Names of layers to hook into
        """
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.attention_weights:
                    self.attention_weights[name] = []
                
                # Extract attention weights from output
                if hasattr(output, 'attention_weights'):
                    self.attention_weights[name].append(output.attention_weights)
                elif isinstance(output, tuple) and len(output) > 1:
                    self.attention_weights[name].append(output[1])
            return hook
        
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(get_hook(name))
                self.hooks.append(hook)
    
    def get_attention_weights(self) -> Dict[str, List[torch.Tensor]]:
        """
        Get collected attention weights from all hooked layers.
        
        Returns:
            attention_weights: Dictionary mapping layer names to attention weights
        """
        return self.attention_weights
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}

