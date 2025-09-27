"""
Visualization Tools for Dual Cross-Attention Learning

This module provides comprehensive visualization capabilities for:
1. Attention maps and attention rollout visualization
2. Model performance analysis and comparison
3. Training progress monitoring
4. Feature embedding visualization  

Based on visualization techniques from vit_rollout.py and extended for
dual cross-attention analysis.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class AttentionVisualizer:
    """
    Comprehensive attention visualization for dual cross-attention models.
    
    Provides tools to visualize:
    - Self-attention maps from SA branch
    - Global-local cross-attention maps from GLCA branch  
    - Attention rollout across all layers
    - Comparison between different attention mechanisms
    
    Based on visualization methods from vit_rollout.py but extended for
    multiple attention types and cross-attention analysis.
    """
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_attention_rollout(self, image: np.ndarray, attention_map: np.ndarray,
                                  title: str = "Attention Rollout", save_name: Optional[str] = None,
                                  colormap: str = "jet", alpha: float = 0.6) -> np.ndarray:
        """
        Visualize attention rollout as heatmap overlay.
        
        Based on show_mask_on_image() from vit_rollout.py but with enhanced
        customization options for different attention types.
        
        Args:
            image: Original image [H, W, 3]
            attention_map: Attention rollout map [H, W]  
            title: Title for the visualization
            save_name: Optional filename to save visualization
            colormap: OpenCV colormap for heatmap
            alpha: Blending factor (0.0=only heatmap, 1.0=only image)
            
        Returns:
            visualization: Combined image with attention overlay [H, W, 3]
        """
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Normalize attention map to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Resize attention map to match image size
        if attention_map.shape != image.shape[:2]:
            attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), getattr(cv2, f'COLORMAP_{colormap.upper()}'))
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Combine image and heatmap
        visualization = alpha * image + (1 - alpha) * heatmap
        visualization = np.clip(visualization, 0, 1)
        
        # Convert back to uint8
        visualization = (visualization * 255).astype(np.uint8)
        
        # Save if requested
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(save_path, visualization)
        
        return visualization
    
    def compare_attention_mechanisms(self, image: np.ndarray,
                                   sa_attention: np.ndarray, glca_attention: np.ndarray,
                                   save_name: Optional[str] = None) -> np.ndarray:
        """
        Compare SA and GLCA attention maps side by side.
        
        Creates visualization showing how different attention mechanisms
        focus on different regions of the same image.
        
        Args:
            image: Original image [H, W, 3]
            sa_attention: Self-attention rollout map [H, W]
            glca_attention: GLCA attention map [H, W]
            save_name: Optional filename to save comparison
            
        Returns:
            comparison: Side-by-side attention comparison
        """
        # Create visualizations for each attention type
        sa_vis = self.visualize_attention_rollout(image, sa_attention, colormap="jet")
        glca_vis = self.visualize_attention_rollout(image, glca_attention, colormap="plasma")
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original image
        original_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        comparison[:, :w] = original_uint8
        
        # SA attention
        comparison[:, w:2*w] = sa_vis
        
        # GLCA attention  
        comparison[:, 2*w:3*w] = glca_vis
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "SA Attention", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "GLCA Attention", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save if requested
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(save_path, comparison)
        
        return comparison
    
    def visualize_attention_evolution(self, image: np.ndarray, 
                                    attention_maps: List[np.ndarray],
                                    layer_names: List[str],
                                    save_name: Optional[str] = None) -> np.ndarray:
        """
        Show attention evolution across transformer layers.
        
        Visualizes how attention patterns change from early to late layers,
        helping understand the hierarchical attention learning process.
        
        Args:
            image: Original image [H, W, 3]
            attention_maps: List of attention maps from different layers
            layer_names: Names of the layers  
            save_name: Optional filename to save evolution plot
            
        Returns:
            evolution_plot: Multi-panel attention evolution visualization
        """
        num_layers = len(attention_maps)
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if num_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (attn_map, layer_name) in enumerate(zip(attention_maps, layer_names)):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Create attention visualization
            vis = self.visualize_attention_rollout(image, attn_map)
            
            ax.imshow(vis)
            ax.set_title(layer_name)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_layers, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        evolution_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        evolution_plot = evolution_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return evolution_plot
    
    def visualize_local_regions(self, image: np.ndarray, attention_rollout: np.ndarray,
                              top_k_ratio: float = 0.1, save_name: Optional[str] = None) -> np.ndarray:
        """
        Visualize high-response local regions selected by GLCA.
        
        Shows which patches are selected as discriminative local regions
        based on attention rollout scores.
        
        Args:
            image: Original image [H, W, 3]
            attention_rollout: Attention rollout scores [H, W]
            top_k_ratio: Ratio of top patches to highlight (R in paper)
            save_name: Optional filename to save visualization
            
        Returns:
            local_regions: Image with highlighted local regions
        """
        # Assume image is divided into 16x16 patches (ViT standard)
        patch_size = 16
        h, w = image.shape[:2]
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        
        # Flatten attention scores and get top-k
        attention_flat = attention_rollout.flatten()
        num_top_patches = max(1, int(len(attention_flat) * top_k_ratio))
        top_indices = np.argpartition(attention_flat, -num_top_patches)[-num_top_patches:]
        
        # Create mask for selected patches
        mask = np.zeros_like(attention_rollout, dtype=bool)
        for idx in top_indices:
            patch_h = idx // num_patches_w
            patch_w = idx % num_patches_w
            
            # Handle boundary cases
            start_h = min(patch_h * patch_size, h - patch_size)
            end_h = min(start_h + patch_size, h)
            start_w = min(patch_w * patch_size, w - patch_size)
            end_w = min(start_w + patch_size, w)
            
            mask[start_h:end_h, start_w:end_w] = True
        
        # Create visualization
        local_regions = image.copy()
        if image.max() > 1.0:
            local_regions = local_regions.astype(np.float32) / 255.0
        
        # Highlight selected regions with colored border
        highlight_color = np.array([1.0, 0.0, 0.0])  # Red
        
        # Create border mask
        border_mask = mask.astype(np.uint8)
        border_mask = cv2.dilate(border_mask, np.ones((3, 3), np.uint8), iterations=2)
        border_mask = border_mask.astype(bool) & ~mask
        
        # Apply highlight
        for c in range(3):
            local_regions[:, :, c] = np.where(border_mask, highlight_color[c], local_regions[:, :, c])
        
        # Convert back to uint8
        if local_regions.max() <= 1.0:
            local_regions = (local_regions * 255).astype(np.uint8)
        
        # Save if requested
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            cv2.imwrite(save_path, local_regions)
        
        return local_regions
    
    def create_attention_statistics_plot(self, attention_stats: Dict[str, List[float]],
                                       save_name: Optional[str] = None) -> plt.Figure:
        """
        Create statistical plots of attention patterns.
        
        Visualizes attention statistics across layers, heads, and training steps
        to understand model behavior and training dynamics.
        
        Args:
            attention_stats: Dictionary with attention statistics
            save_name: Optional filename to save plot
            
        Returns:
            figure: Matplotlib figure with attention statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Plot different statistics
        plot_configs = [
            ("attention_entropy", "Attention Entropy", "Epoch", "Entropy"),
            ("attention_max", "Max Attention Weight", "Epoch", "Max Weight"),
            ("attention_std", "Attention Std Dev", "Epoch", "Std Dev"),
            ("rollout_concentration", "Rollout Concentration", "Epoch", "Concentration")
        ]
        
        for i, (key, title, xlabel, ylabel) in enumerate(plot_configs):
            if key in attention_stats:
                axes[i].plot(attention_stats[key])
                axes[i].set_title(title)
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def plot_attention_maps(images: Union[np.ndarray, List[np.ndarray]], 
                       attention_maps: Union[np.ndarray, List[np.ndarray]],
                       titles: Optional[List[str]] = None,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot multiple attention maps in a grid layout.
    
    Utility function for creating clean attention visualizations for papers
    and presentations.
    
    Args:
        images: Original images or list of images
        attention_maps: Corresponding attention maps
        titles: Optional titles for each subplot
        save_path: Path to save the figure
        figsize: Figure size (width, height)
        
    Returns:
        figure: Matplotlib figure with attention grid
    """
    # Convert single items to lists
    if not isinstance(images, list):
        images = [images]
    if not isinstance(attention_maps, list):
        attention_maps = [attention_maps]
    
    num_samples = len(images)
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, (image, attn_map) in enumerate(zip(images, attention_maps)):
        # Original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Original {i+1}" if titles is None else titles[i])
        axes[0, i].axis('off')
        
        # Attention map
        visualizer = AttentionVisualizer()
        vis = visualizer.visualize_attention_rollout(image, attn_map)
        axes[1, i].imshow(vis)
        axes[1, i].set_title(f"Attention {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_attention_heatmaps(attention_maps: Dict[str, np.ndarray], 
                          original_image: np.ndarray, save_dir: str):
    """
    Save individual attention heatmaps for different mechanisms.
    
    Saves separate files for SA, GLCA, and combined attention visualizations
    for detailed analysis and paper figures.
    
    Args:
        attention_maps: Dictionary mapping attention types to maps
        original_image: Original input image
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    visualizer = AttentionVisualizer(save_dir)
    
    for attention_type, attention_map in attention_maps.items():
        save_name = f"{attention_type}_heatmap.png"
        visualizer.visualize_attention_rollout(
            original_image, 
            attention_map,
            title=attention_type.upper(),
            save_name=save_name
        )


class PerformanceVisualizer:
    """
    Performance analysis and comparison visualization.
    
    Creates plots and charts for:
    - Training curves (loss, accuracy over time)
    - Model comparison across different configurations  
    - Dataset-specific performance analysis
    - Ablation study results
    """
    
    def __init__(self, save_dir: str = "./performance_plots"):
        pass
    
    def plot_training_curves(self, training_logs: Dict[str, List[float]],
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            training_logs: Dictionary with training metrics over time
            save_name: Optional filename to save plot
            
        Returns:
            figure: Training curves figure
        """
        pass
    
    def plot_loss_components(self, loss_logs: Dict[str, List[float]],
                           uncertainty_weights: Dict[str, List[float]],
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot individual loss components and their uncertainty weights.
        
        Shows how SA, GLCA, and PWCA losses evolve during training
        and how the uncertainty-based weighting adapts.
        
        Args:
            loss_logs: Individual loss components over time
            uncertainty_weights: Uncertainty parameters over time
            save_name: Optional filename to save plot
            
        Returns:
            figure: Loss components figure
        """
        pass
    
    def create_model_comparison(self, results_dict: Dict[str, Dict[str, float]],
                              metrics: List[str], save_name: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart comparing different models/configurations.
        
        Args:
            results_dict: Dictionary mapping model names to their metrics
            metrics: List of metrics to compare
            save_name: Optional filename to save plot
            
        Returns:
            figure: Model comparison figure
        """
        pass
    
    def plot_ablation_results(self, ablation_results: Dict[str, Dict[str, float]],
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot ablation study results showing contribution of each component.
        
        Args:
            ablation_results: Results for different component combinations
            save_name: Optional filename to save plot
            
        Returns:
            figure: Ablation study figure
        """
        pass


class FeatureVisualizer:
    """
    Feature embedding and representation visualization.
    
    Tools for analyzing learned feature representations:
    - t-SNE and PCA embeddings
    - Feature similarity analysis
    - Class clustering visualization
    """
    
    def __init__(self):
        pass
    
    def visualize_feature_embeddings(self, features: np.ndarray, labels: np.ndarray,
                                   method: str = "tsne", save_name: Optional[str] = None) -> plt.Figure:
        """
        Visualize high-dimensional features in 2D space.
        
        Args:
            features: Feature embeddings [num_samples, feature_dim]
            labels: Class/identity labels [num_samples]
            method: Dimensionality reduction method ("tsne" or "pca")
            save_name: Optional filename to save plot
            
        Returns:
            figure: Feature embedding visualization
        """
        pass
    
    def plot_feature_similarity_matrix(self, features: np.ndarray, labels: np.ndarray,
                                     save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot feature similarity matrix between different classes.
        
        Args:
            features: Feature embeddings [num_samples, feature_dim]
            labels: Class labels [num_samples]
            save_name: Optional filename to save plot
            
        Returns:
            figure: Similarity matrix heatmap
        """
        pass
    
    def analyze_attention_head_specialization(self, attention_weights: List[torch.Tensor],
                                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Analyze how different attention heads specialize on different patterns.
        
        Args:
            attention_weights: List of multi-head attention weights
            save_name: Optional filename to save analysis
            
        Returns:
            figure: Attention head specialization analysis
        """
        pass


def create_paper_figures(results_dict: Dict[str, any], save_dir: str = "./paper_figures"):
    """
    Create publication-ready figures matching the paper.
    
    Generates all the visualizations shown in the dual cross-attention paper:
    - Attention rollout comparisons
    - Performance comparison tables
    - Ablation study results
    - Qualitative attention visualizations
    
    Args:
        results_dict: Comprehensive results from experiments
        save_dir: Directory to save paper figures
    """
    pass


def interactive_attention_explorer(model, dataset_loader, save_path: str = "attention_explorer.html"):
    """
    Create interactive attention exploration tool.
    
    Generates an interactive HTML page for exploring attention patterns
    across different images, layers, and attention mechanisms.
    
    Args:
        model: Trained dual cross-attention model
        dataset_loader: Dataset for exploration
        save_path: Path to save interactive HTML
    """
    pass

