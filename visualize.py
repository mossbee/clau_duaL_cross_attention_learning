"""
Attention Visualization Script for Dual Cross-Attention Learning

Specialized script for generating comprehensive attention visualizations
and analysis for the dual cross-attention model. Creates publication-ready
figures showing attention rollout, cross-attention patterns, and model behavior.

Usage:
    # Generate attention maps for sample images
    python visualize.py --checkpoint path/to/model.pth --dataset cub --images path/to/images
    
    # Create paper figures
    python visualize.py --checkpoint path/to/model.pth --dataset cub --paper_figures
    
    # Interactive attention explorer
    python visualize.py --checkpoint path/to/model.pth --dataset cub --interactive
    
    # Compare different attention mechanisms
    python visualize.py --checkpoint path/to/model.pth --dataset cub --compare_attention
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dual_cross_attention.models import DualCrossAttentionViT
from dual_cross_attention.utils import (
    AttentionRollout, AttentionVisualizer, rollout_attention_maps,
    plot_attention_maps, save_attention_heatmaps
)
from dual_cross_attention.datasets import get_transform_factory
from dual_cross_attention.configs import get_fgvc_config, get_reid_config


class AttentionAnalyzer:
    """
    Comprehensive attention analysis and visualization tool.
    
    Provides tools for:
    - Attention rollout computation and visualization  
    - Cross-attention pattern analysis
    - Layer-wise attention evolution
    - Publication-ready figure generation
    - Interactive attention exploration
    
    Args:
        model: Trained dual cross-attention model
        config: Task configuration
        device: Device for computation
    """
    
    def __init__(self, model: nn.Module, config, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        
        # Setup transform for image preprocessing
        from dual_cross_attention.datasets import get_transform_factory
        self.transform = get_transform_factory(config.task_type, config.dataset_name, is_training=False)
        
        # Initialize visualizer
        self.visualizer = AttentionVisualizer()
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess image for model input.
        
        Args:
            image_path: Path to input image
            
        Returns:
            input_tensor: Preprocessed tensor for model
            original_image: Original image array for visualization
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Preprocess for model
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return input_tensor, original_image
    
    def compute_attention_rollout(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute attention rollout for all attention mechanisms.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            rollout_maps: Dictionary with rollout maps for SA and GLCA
        """
        with torch.no_grad():
            # Forward pass to get attention maps
            outputs = self.model(input_tensor, inference_mode=True)
            attention_maps = self.model.get_attention_maps()
        
        rollout_maps = {}
        
        # Compute SA rollout using attention rollout utility
        if 'sa_attention' in attention_maps and attention_maps['sa_attention']:
            sa_rollout_scores = rollout_attention_maps(
                self.model, input_tensor, discard_ratio=0.9, head_fusion="mean"
            )
            if 'sa_rollout' in sa_rollout_scores:
                # Convert to 2D spatial map
                rollout_computer = AttentionRollout(discard_ratio=0.9, head_fusion="mean")
                img_size = self.config.input_size if hasattr(self.config, 'input_size') else 224
                rollout_maps['sa'] = rollout_computer.get_cls_attention_map(
                    sa_rollout_scores['sa_rollout'], 
                    (img_size, img_size),
                    patch_size=16
                )
        
        return rollout_maps
    
    def visualize_single_image(self, image_path: str, save_dir: str = "./attention_vis"):
        """
        Generate comprehensive attention visualization for a single image.
        
        Creates:
        - SA attention rollout
        - GLCA attention map
        - Side-by-side comparison
        - High-response local regions overlay
        
        Args:
            image_path: Path to input image
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and preprocess image
        input_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        # Compute attention rollout
        rollout_maps = self.compute_attention_rollout(input_tensor)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Visualize SA attention if available
        if 'sa' in rollout_maps:
            save_path = os.path.join(save_dir, f"{base_name}_sa_attention.png")
            from dual_cross_attention.utils.attention_rollout import visualize_attention_rollout
            visualize_attention_rollout(
                original_image, 
                rollout_maps['sa'],
                save_path=save_path,
                alpha=0.6
            )
            print(f"Saved SA attention to {save_path}")
        
        # Visualize GLCA attention if available
        if 'glca' in rollout_maps:
            save_path = os.path.join(save_dir, f"{base_name}_glca_attention.png")
            from dual_cross_attention.utils.attention_rollout import visualize_attention_rollout
            visualize_attention_rollout(
                original_image,
                rollout_maps['glca'],
                save_path=save_path,
                alpha=0.6
            )
            print(f"Saved GLCA attention to {save_path}")
        
        # Compare attention mechanisms if both available
        if 'sa' in rollout_maps and 'glca' in rollout_maps:
            save_path = os.path.join(save_dir, f"{base_name}_comparison.png")
            visualizer = AttentionVisualizer(save_dir)
            visualizer.compare_attention_mechanisms(
                original_image,
                rollout_maps['sa'],
                rollout_maps['glca'],
                save_name=f"{base_name}_comparison.png"
            )
            print(f"Saved attention comparison to {save_path}")
    
    def compare_attention_mechanisms(self, image_paths: List[str], 
                                   save_path: str = "./attention_comparison.png"):
        """
        Compare SA and GLCA attention patterns across multiple images.
        
        Creates a grid visualization showing how different attention
        mechanisms focus on different regions.
        
        Args:
            image_paths: List of image paths to analyze
            save_path: Path to save comparison figure
        """
        num_images = min(len(image_paths), 6)  # Limit to 6 for clean visualization
        fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 4))
        
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for i, image_path in enumerate(image_paths[:num_images]):
            # Load and process image
            input_tensor, original_image = self.load_and_preprocess_image(image_path)
            rollout_maps = self.compute_attention_rollout(input_tensor)
            
            # Original image
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis('off')
            
            # SA attention
            if 'sa' in rollout_maps:
                from dual_cross_attention.utils.attention_rollout import visualize_attention_rollout
                sa_vis = visualize_attention_rollout(original_image, rollout_maps['sa'])
                axes[i, 1].imshow(sa_vis)
                axes[i, 1].set_title("SA Attention")
                axes[i, 1].axis('off')
            
            # GLCA attention
            if 'glca' in rollout_maps:
                from dual_cross_attention.utils.attention_rollout import visualize_attention_rollout
                glca_vis = visualize_attention_rollout(original_image, rollout_maps['glca'])
                axes[i, 2].imshow(glca_vis)
                axes[i, 2].set_title("GLCA Attention")
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved attention comparison to {save_path}")
    
    def analyze_layer_evolution(self, image_path: str, 
                              save_path: str = "./layer_evolution.png"):
        """
        Visualize attention evolution across transformer layers.
        
        Shows how attention patterns develop from early to late layers,
        revealing the hierarchical attention learning process.
        
        Args:
            image_path: Path to input image
            save_path: Path to save evolution plot
        """
        # Load and preprocess image
        input_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        # Get layer-wise attention
        with torch.no_grad():
            outputs = self.model(input_tensor, inference_mode=True)
            attention_maps = self.model.get_attention_maps()
        
        if 'sa_attention' not in attention_maps or not attention_maps['sa_attention']:
            print("No attention maps available for layer evolution")
            return
        
        # Compute rollout at each layer incrementally
        from dual_cross_attention.utils.attention_rollout import AttentionRollout
        rollout_computer = AttentionRollout(discard_ratio=0.9, head_fusion="mean")
        
        layer_maps = []
        layer_names = []
        img_size = self.config.input_size if hasattr(self.config, 'input_size') else 224
        
        # Sample every few layers for visualization
        sample_layers = [0, 3, 6, 9, 11]  # Early, mid, late layers
        for layer_idx in sample_layers:
            if layer_idx < len(attention_maps['sa_attention']):
                # Compute rollout up to this layer
                rollout_scores = rollout_computer.rollout(
                    attention_maps['sa_attention'][:layer_idx+1]
                )
                if rollout_scores is not None:
                    attn_map = rollout_computer.get_cls_attention_map(
                        rollout_scores, (img_size, img_size), patch_size=16
                    )
                    layer_maps.append(attn_map)
                    layer_names.append(f"Layer {layer_idx+1}")
        
        # Create evolution visualization
        if layer_maps:
            visualizer = AttentionVisualizer()
            visualizer.visualize_attention_evolution(
                original_image, layer_maps, layer_names, save_name=os.path.basename(save_path)
            )
            print(f"Saved layer evolution to {save_path}")
    
    def visualize_local_region_selection(self, image_path: str,
                                       save_path: str = "./local_regions.png"):
        """
        Visualize which regions are selected as discriminative local patches.
        
        Shows the top-k patches selected by attention rollout for GLCA
        cross-attention computation.
        
        Args:
            image_path: Path to input image  
            save_path: Path to save visualization
        """
        # Load and preprocess image
        input_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        # Compute attention rollout
        rollout_maps = self.compute_attention_rollout(input_tensor)
        
        if 'sa' in rollout_maps:
            # Determine top_k_ratio from config
            top_k_ratio = getattr(self.config, 'top_k_ratio', 0.1)
            
            # Visualize local regions
            visualizer = AttentionVisualizer()
            visualizer.visualize_local_regions(
                original_image,
                rollout_maps['sa'],
                top_k_ratio=top_k_ratio,
                save_name=os.path.basename(save_path)
            )
            print(f"Saved local region visualization to {save_path}")
    
    def generate_paper_figures(self, sample_images: List[str], save_dir: str = "./paper_figures"):
        """
        Generate publication-ready figures matching the paper.
        
        Creates all attention visualizations shown in the dual cross-attention paper:
        - Figure showing attention rollout comparison
        - Local region selection visualization  
        - Layer-wise attention evolution
        - Cross-attention pattern analysis
        
        Args:
            sample_images: List of representative images
            save_dir: Directory to save paper figures
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating paper figures...")
        
        # 1. Attention mechanism comparison across multiple images
        self.compare_attention_mechanisms(
            sample_images, 
            os.path.join(save_dir, "fig_attention_comparison.png")
        )
        
        # 2. Layer evolution for a representative image
        if sample_images:
            self.analyze_layer_evolution(
                sample_images[0],
                os.path.join(save_dir, "fig_layer_evolution.png")
            )
        
        # 3. Local region selection visualization
        for i, img_path in enumerate(sample_images[:3]):
            self.visualize_local_region_selection(
                img_path,
                os.path.join(save_dir, f"fig_local_regions_{i+1}.png")
            )
        
        print(f"Paper figures saved to {save_dir}")
    
    def create_attention_statistics(self, image_paths: List[str]) -> Dict[str, any]:
        """
        Compute statistics about attention patterns.
        
        Analyzes:
        - Average attention distribution
        - Attention entropy across layers
        - Variability in attention patterns
        - Class-specific attention statistics
        
        Args:
            image_paths: List of images to analyze
            
        Returns:
            statistics: Dictionary with attention statistics
        """
        from dual_cross_attention.utils.attention_rollout import compute_attention_statistics
        
        all_stats = []
        
        for image_path in image_paths:
            # Load and process image
            input_tensor, _ = self.load_and_preprocess_image(image_path)
            
            # Get attention weights
            with torch.no_grad():
                outputs = self.model(input_tensor, inference_mode=True)
                attention_maps = self.model.get_attention_maps()
            
            if 'sa_attention' in attention_maps and attention_maps['sa_attention']:
                stats = compute_attention_statistics(attention_maps['sa_attention'])
                all_stats.append(stats)
        
        # Aggregate statistics
        if all_stats:
            aggregated_stats = {
                'mean_attention_mean': np.mean([s['mean'] for s in all_stats]),
                'mean_attention_std': np.mean([s['std'] for s in all_stats]),
                'mean_entropy': np.mean([s['mean_entropy'] for s in all_stats]),
                'std_entropy': np.mean([s['std_entropy'] for s in all_stats]),
                'num_samples': len(all_stats)
            }
        else:
            aggregated_stats = {}
        
        return aggregated_stats
    
    def interactive_attention_explorer(self, image_paths: List[str],
                                     save_path: str = "./attention_explorer.html"):
        """
        Create interactive HTML visualization for exploring attention.
        
        Generates an interactive tool allowing users to:
        - Browse through different images
        - Toggle between attention mechanisms
        - Adjust visualization parameters
        - Export specific visualizations
        
        Args:
            image_paths: List of images for exploration
            save_path: Path to save interactive HTML
        """
        print("Interactive attention explorer requires plotly. Creating static gallery instead.")
        
        # Create a simple static HTML gallery as fallback
        html_content = ["<html><head><title>Attention Visualization Gallery</title></head><body>"]
        html_content.append("<h1>Dual Cross-Attention Visualization Gallery</h1>")
        
        for i, image_path in enumerate(image_paths):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            html_content.append(f"<h2>Image {i+1}: {base_name}</h2>")
            
            # Generate visualizations
            save_dir = os.path.dirname(save_path)
            self.visualize_single_image(image_path, save_dir)
            
            # Add images to HTML
            html_content.append(f'<img src="{base_name}_sa_attention.png" width="400">')
            html_content.append(f'<img src="{base_name}_comparison.png" width="800"><br>')
        
        html_content.append("</body></html>")
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        print(f"Saved attention gallery to {save_path}")


def load_model(checkpoint_path: str, config) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        
    Returns:
        model: Loaded model in evaluation mode
    """
    
    # Create model
    model = DualCrossAttentionViT(
        num_classes=config.num_classes,
        **config.get_model_config()
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_sample_images(dataset_name: str, data_root: str, num_samples: int = 8) -> List[str]:
    """
    Get sample images from dataset for visualization.
    
    Args:
        dataset_name: Name of dataset
        data_root: Root directory of dataset
        num_samples: Number of samples to select
        
    Returns:
        image_paths: List of selected image paths
    """
    import glob
    import random
    
    # Try common dataset structures
    possible_paths = [
        os.path.join(data_root, "**/*.jpg"),
        os.path.join(data_root, "**/*.png"),
        os.path.join(data_root, "**/*.JPEG"),
        os.path.join(data_root, "test", "**/*.jpg"),
        os.path.join(data_root, "val", "**/*.jpg"),
    ]
    
    image_paths = []
    for pattern in possible_paths:
        image_paths.extend(glob.glob(pattern, recursive=True))
        if len(image_paths) >= num_samples:
            break
    
    if not image_paths:
        print(f"Warning: No images found in {data_root}")
        return []
    
    # Sample random images
    if len(image_paths) > num_samples:
        image_paths = random.sample(image_paths, num_samples)
    
    return image_paths


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dual Cross-Attention Visualization")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (cub, cars, aircraft, market1501, etc.)")
    
    # Input specification
    parser.add_argument("--images", type=str, nargs='+',
                       help="Specific image paths to visualize")
    parser.add_argument("--image_dir", type=str,
                       help="Directory containing images to visualize")
    parser.add_argument("--data_root", type=str,
                       help="Dataset root directory (for automatic sample selection)")
    parser.add_argument("--num_samples", type=int, default=8,
                       help="Number of samples for automatic selection")
    
    # Visualization modes
    parser.add_argument("--single_image", action="store_true",
                       help="Generate detailed visualization for single images")
    parser.add_argument("--compare_attention", action="store_true",
                       help="Compare SA vs GLCA attention patterns")
    parser.add_argument("--layer_evolution", action="store_true",
                       help="Show attention evolution across layers")
    parser.add_argument("--local_regions", action="store_true",
                       help="Visualize local region selection")
    parser.add_argument("--paper_figures", action="store_true",
                       help="Generate publication-ready figures")
    parser.add_argument("--interactive", action="store_true",
                       help="Create interactive attention explorer")
    parser.add_argument("--statistics", action="store_true",
                       help="Compute attention statistics")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "pdf"],
                       help="Output format for figures")
    
    # Visualization settings
    parser.add_argument("--colormap", type=str, default="jet",
                       help="Colormap for attention heatmaps")
    parser.add_argument("--alpha", type=float, default=0.6,
                       help="Transparency for attention overlay")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8],
                       help="Figure size (width, height)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for computation")
    
    return parser.parse_args()


def main():
    """Main visualization function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}...")
    
    # Determine task type from dataset name
    fgvc_datasets = ["cub", "cars", "aircraft"]
    reid_datasets = ["market1501", "duke", "msmt17", "veri776"]
    
    if args.dataset in fgvc_datasets:
        config = get_fgvc_config(args.dataset)
        task_type = "fgvc"
    elif args.dataset in reid_datasets:
        config = get_reid_config(args.dataset)
        task_type = "reid"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Override config with arguments
    if args.data_root:
        config.data_root = args.data_root
    
    # Load model
    model = load_model(args.checkpoint, config)
    model = model.to(args.device)
    
    # Initialize analyzer
    analyzer = AttentionAnalyzer(model, config, args.device)
    
    # Get image paths
    if args.images:
        image_paths = args.images
    elif args.image_dir:
        image_paths = [str(p) for p in Path(args.image_dir).glob("*.jpg")]
        image_paths.extend([str(p) for p in Path(args.image_dir).glob("*.png")])
        image_paths = image_paths[:args.num_samples]
    elif args.data_root:
        image_paths = get_sample_images(args.dataset, args.data_root, args.num_samples)
    else:
        raise ValueError("Must specify --images, --image_dir, or --data_root")
    
    print(f"Analyzing {len(image_paths)} images...")
    
    # Generate visualizations based on requested modes
    if args.single_image or not any([args.compare_attention, args.layer_evolution, 
                                   args.local_regions, args.paper_figures, args.interactive]):
        # Default: single image analysis
        print("Generating single image visualizations...")
        for image_path in image_paths:
            print(f"Processing {image_path}...")
            analyzer.visualize_single_image(image_path, args.output_dir)
    
    if args.compare_attention:
        print("Generating attention mechanism comparison...")
        analyzer.compare_attention_mechanisms(
            image_paths, os.path.join(args.output_dir, "attention_comparison.png")
        )
    
    if args.layer_evolution:
        print("Generating layer evolution visualization...")
        for image_path in image_paths[:3]:  # Limit to first 3 images for clarity
            analyzer.analyze_layer_evolution(
                image_path, os.path.join(args.output_dir, f"layer_evolution_{Path(image_path).stem}.png")
            )
    
    if args.local_regions:
        print("Visualizing local region selection...")
        for image_path in image_paths:
            analyzer.visualize_local_region_selection(
                image_path, os.path.join(args.output_dir, f"local_regions_{Path(image_path).stem}.png")
            )
    
    if args.paper_figures:
        print("Generating paper figures...")
        analyzer.generate_paper_figures(image_paths[:6], args.output_dir)
    
    if args.interactive:
        print("Creating interactive attention explorer...")
        analyzer.interactive_attention_explorer(
            image_paths, os.path.join(args.output_dir, "attention_explorer.html")
        )
    
    if args.statistics:
        print("Computing attention statistics...")
        stats = analyzer.create_attention_statistics(image_paths)
        
        # Save statistics
        import json
        with open(os.path.join(args.output_dir, "attention_statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)
    
    print(f"Visualization complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

